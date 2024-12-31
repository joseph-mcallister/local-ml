# https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

image = Image.open("vision/sample_img.jpg")

prompt_msg = """
You are operating a small robot. 
Your job is to summarize what you see and then navigate the robot as directed. 
You are able to go forward, or turn up to 90 degrees in either direction. 
<action> can be <forward>, <turn>, or <stop>
<distance> can be any integer between 0 and 24 and measured in inches.
<direction> can be <left> or <right>.
<degrees> can be any integer between 0 and 90.

Reply in the following format:
Summary: <summary>
Action: <action>

For example, you can reply:
Summary: There is stool in the left and a fridge to the right. There is nothing directly in front of me.
Action: <forward> 12

Another response can be
Summary: There is a stool directly in front of me.
Action: <turn> <left> 90

Describe what you see in the following format and then try to find and navigate to the bathroom.
"""
# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_msg}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
