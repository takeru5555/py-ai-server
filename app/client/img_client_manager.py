from typing import Union, Dict
import os
from py_api.args import Args
from py_api.client.base_manager import BaseAIManager
from py_api.client.img import ImgClient_Diffusers

ClientUnion = ImgClient_Diffusers
ClientDict = Dict[str, ClientUnion]

EXTENSIONS = ['.safetensors']

class ImgManager(BaseAIManager):
	_instance = None
	clients: ClientDict = {
		'diffusers': ImgClient_Diffusers.instance,
	}
	loader: Union[ClientUnion, None] = None

	def __init__(self):
		self.clients = {
			'diffusers': ImgClient_Diffusers.instance,
		}
		# self.default_model = Args['img_model']
		# self.models_dir = Args['img_models_dir']

	def pick_client(self, model_name: str):
		return 'diffusers'

	def get_models_dir(self) -> str:
		return Args['img_models_dir']

	def get_default_model(self):
		return Args['img_model']

	def list_models(self) -> list[str]:
		models = []
		for filename in os.listdir(self.get_models_dir()):
			path = os.path.join(self.get_models_dir(), filename)
			if not os.path.isfile(path):
				continue
			if not any([filename.endswith(ext) for ext in EXTENSIONS]):
				continue
			if 'inpaint' in filename:
				continue
			models.append(filename)
		return models

	def list_samplers(self) -> list[str]:
		if self.loader is None:
			return []
		return self.loader.list_samplers()

	def txt2img(self, gen_options):
		if self.loader is None:
			raise Exception('No model loaded.')
		return self.loader.txt2img(gen_options)

	# def img2img(self, gen_options):
