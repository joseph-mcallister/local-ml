# https://github.com/coqui-ai/TTS

import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# List available 🐸TTS models
print(TTS().list_models())

# Initialize model
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)

speakers = tts.speakers
tts.tts_to_file(text="this is a test", file_path="/tmp/output.wav", speaker=speakers[0])