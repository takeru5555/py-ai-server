from TTS.api import TTS
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = TTS().list_models()
print(models.models_dict)
