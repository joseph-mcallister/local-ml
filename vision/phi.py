# https://huggingface.co/microsoft/Phi-3.5-vision-instruct

from PIL import Image 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cpu", # "cuda"
  trust_remote_code=True, 
  torch_dtype="auto",
  _attn_implementation='eager'  
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 

placeholder = ""
images = []
images.append(Image.open("vision/sample_img.jpg"))
placeholder += f"<|image_1|>\n"

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
messages = [
    {"role": "user", "content": placeholder + prompt_msg },
] 

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt")

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print(response)
