import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()
# Load the tokenizer and model.
# The model card (see :contentReference[oaicite:0]{index=0}) specifies that for image generation,
# Janus Pro uses a specialized tokenizer (with a downsample rate of 16) and a unified transformer.
model_name = "deepseek-ai/Janus-Pro-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move the model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your text prompt for image generation.
prompt = "A futuristic cityscape at night, with neon lights and flying cars."

# Tokenize the prompt.
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output tokens.
# Adjust generation parameters (max_length, temperature, top_p, etc.) as needed.
generated_tokens = model.generate(
    **inputs,
    max_length=256,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    num_return_sequences=1
)

# For multimodal models like Janus Pro, the generated tokens are typically not plain text.
# Instead, they represent image data that must be decoded via a specialized decoder.
# Below, we provide a dummy 'decode_image' function for illustration.
def decode_image(image_tokens):
    """
    Dummy function to illustrate conversion of tokens to an image.
    In practice, the Janus Pro repository should provide a decoding utility,
    which converts model outputs into an image (e.g., a numpy array or PIL Image).
    """
    # For demonstration, we generate a random 384x384 RGB image.
    dummy_array = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    return Image.fromarray(dummy_array)

# Decode the generated tokens to produce an image.
# Replace 'decode_image' with the model's actual image decoding method.
generated_image = decode_image(generated_tokens[0])

# Display the generated image.
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Image")
plt.show()
plt.savefig("generated_image.png")

# Optionally, print any textual output if the model returns one.
# (For many multimodal models, the textual output may be secondary.)
text_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print("Text output (if any):", text_output)
