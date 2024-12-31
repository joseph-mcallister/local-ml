import ollama

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
Summary: <summary>
Action: <action>
"""

# model = "moondream"
model = "llava"
# model = "llama3.2-vision"
# model = "llama3.3" # 70B (runs on mac but not rpi)

res = ollama.chat(
	model=model,
	messages=[
		{
			'role': 'user',
			'content': prompt_msg,
			'images': ["./vision/sample_img.jpg"]
		}
	]
)

print(res['message']['content'])