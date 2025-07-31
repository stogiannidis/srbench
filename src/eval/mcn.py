import re
from typing import List, Dict, Optional, Union
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.utils import logging as transformers_logging
import logging

# Suppress transformers warnings for cleaner output
transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class MultipleChoiceNormalizer:
    """
    Offline inference system using Hugging Face Transformers for normalizing text to multiple-choice answers.
    """
    
    def __init__(
        self,
        model_name: str = "internlm/internlm3-8b-instruct",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the Hugging Face Transformers-based multiple choice normalizer.
        
        Args:
            model_name: HuggingFace model name or path to local model
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision settings
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not (load_in_8bit or load_in_4bit) and self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Default generation parameters for normalization
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=50,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Common multiple choice patterns
        self.choice_patterns = [
            r'^[A-E]\)',  # A), B), C), D), E)
            r'^[A-E]\.',  # A. B. C. D. E.
            r'^[A-E]$',   # A, B, C, D, E
            r'^[1-5]\)',  # 1), 2), 3, 4), 5)
            r'^[1-5]\.',  # 1. 2. 3. 4. 5.
            r'^[1-5]$',   # 1, 2, 3, 4, 5
        ]
    
    def normalize_to_multiple_choice(
        self,
        texts: Union[str, List[str]],
        choices: List[str] = None,
        prompt_template: str = None,
        use_few_shot: bool = True
    ) -> Union[str, List[str]]:
        """
        Normalize input text(s) to multiple-choice format.
        
        Args:
            texts: Single text or list of texts to normalize
            choices: List of possible choices (A, B, C, D, E or 1, 2, 3, 4, 5)
            prompt_template: Custom prompt template for normalization
            use_few_shot: Whether to use few-shot examples in the prompt
            
        Returns:
            Normalized choice(s) as string or list of strings
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if choices is None:
            choices = ['A', 'B', 'C', 'D', 'E']
        
        if prompt_template is None:
            prompt_template = self._get_default_prompt_template(choices, use_few_shot)
        
        # Generate prompts
        prompts = []
        for text in texts:
            prompt = prompt_template.format(
                text=text.strip(),
                choices=', '.join(choices)
            )
            prompts.append(prompt)
        
        # Generate responses using Hugging Face Transformers
        generated_texts = self._generate_batch(prompts)
        
        # Extract and normalize answers
        normalized_answers = []
        for generated_text in generated_texts:
            normalized_answer = self._extract_choice(generated_text, choices)
            normalized_answers.append(normalized_answer)
        
        return normalized_answers[0] if single_input else normalized_answers
    
    def batch_normalize(
        self,
        texts: List[str],
        choices: List[str] = None,
        batch_size: int = 8
    ) -> List[str]:
        """
        Batch normalize multiple texts efficiently.
        
        Args:
            texts: List of texts to normalize
            choices: List of possible choices
            batch_size: Size of batches for processing (reduced default for HF)
            
        Returns:
            List of normalized choices
        """
        if choices is None:
            choices = ['A', 'B', 'C', 'D', 'E']
        
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.normalize_to_multiple_choice(batch_texts, choices)
            all_results.extend(batch_results)
        
        return all_results
    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a batch of prompts using Hugging Face Transformers.
        
        Args:
            prompts: List of prompts to generate responses for
            
        Returns:
            List of generated text responses
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                use_cache=True
            )
        
        # Decode responses (only the newly generated tokens)
        generated_texts = []
        for i, output in enumerate(outputs):
            # Skip the input tokens and decode only the generated part
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def _get_default_prompt_template(self, choices: List[str], use_few_shot: bool) -> str:
        """Get default prompt template for normalization."""
        choices_str = ', '.join(choices)
        if use_few_shot:
            few_shot_examples = """
Example 1:
Text: "The capital of France is Paris."
Choices: A) London, B) Berlin, C) Paris, D) Madrid, E) Rome
Answer: C

Example 2:
Text: "Water boils at 100 degrees Celsius."
Choices: A) 90, B) 100, C) 110, D) 120, E) 130
Answer: B
"""
            return f"""{few_shot_examples}
Given the following text, extract or determine the most appropriate multiple-choice answer from the options: {choices_str}

Text: "{{text}}"

Respond with only the letter/number of the correct choice ({choices_str}):"""
        else:
            return """Given the following text, extract or determine the most appropriate multiple-choice answer from the options: {choices}

Text: "{text}"

Respond with only the letter/number of the correct choice ({choices}):"""
    
    def _extract_choice(self, generated_text: str, choices: List[str]) -> str:
        """
        Extract the choice from generated text using pattern matching.
        
        Args:
            generated_text: Generated text from the model
            choices: List of valid choices
            
        Returns:
            Extracted choice or first choice if no match found
        """
        text = generated_text.upper().strip()
        
        # First, try to find patterns like "A. Description", "B. Something", etc.
        choice_with_dot_pattern = r'^([A-E])\.'
        match = re.search(choice_with_dot_pattern, text)
        if match:
            letter = match.group(1)
            if letter in [c.upper() for c in choices]:
                return letter
        
        # Try to find patterns like "A)", "B)", etc.
        choice_with_paren_pattern = r'^([A-E])\)'
        match = re.search(choice_with_paren_pattern, text)
        if match:
            letter = match.group(1)
            if letter in [c.upper() for c in choices]:
                return letter
        
        # Direct match with choices (single letters)
        for choice in choices:
            choice_upper = choice.upper()
            # Look for the choice as a standalone letter
            if re.search(r'\b' + re.escape(choice_upper) + r'\b', text):
                return choice_upper
        
        # Look for choice at the beginning of the text
        for choice in choices:
            if text.startswith(choice.upper()):
                return choice.upper()
        
        # Pattern matching for various formats
        for pattern in self.choice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                match = matches[0].upper()
                # Extract just the letter/number
                clean_match = re.sub(r'[^A-E0-9]', '', match)
                if clean_match in [c.upper() for c in choices]:
                    return clean_match
        
        # Last resort: look for any single letter that matches our choices
        single_letters = re.findall(r'\b([A-E])\b', text)
        for letter in single_letters:
            if letter in [c.upper() for c in choices]:
                return letter
        
        # Fallback: return first choice
        return choices[0].upper()
    
    def update_generation_config(self, **kwargs):
        """Update generation parameters."""
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
    
    def save_results(self, texts: List[str], results: List[str], filepath: str):
        """Save normalization results to JSON file."""
        data = {
            'model': self.model_name,
            'results': [
                {'input': text, 'normalized_choice': result}
                for text, result in zip(texts, results)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)