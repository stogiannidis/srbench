from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

class TITTDataset(Dataset):
    def __init__(self, data, tokenizer, vlm):
        self.data = data
        self.tokenizer = tokenizer
        self.vlm = vlm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        text = self.vlm.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)

        inputs = self.vlm.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)

        return inputs, answer
