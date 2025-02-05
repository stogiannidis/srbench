from transformers import pipeline
import glob

models = glob.glob("bin/models/llms/*")


def run_text_generation_pipeline(model_name):
    # Create a text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device_map="auto"  # Automatically place model on available device
    )
    print(f"Running model {model_name}")
    # Prepare the prompt by combining the system and user messages
    # Here, we mimic the chat template by concatenating the messages.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    try:
        output = generator(messages)
        print(output)
    except Exception as e:
        print(f"Error with model {model_name}: {e}")


# for model in models:
# model = "bin/models/llms/google_gemma-2-27b-it"
# run_text_generation_pipeline(model)

def supports_system_prompt(model_name: str) -> bool:
    """
    A simple heuristic to decide if the model supports chat-style prompts with a system message.
    In this example, if the model id contains "gemma" (case-insensitive) we assume it does.
    """
    return not "gemma" in model_name.lower()

for model in models:
    print(f"Model {model} supports system prompt: {supports_system_prompt(model)}")