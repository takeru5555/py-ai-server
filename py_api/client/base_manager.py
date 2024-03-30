from typing import Union

class BaseAIManager:
	_instance = None
	clients = {}
	model_name: Union[str, None] = None
	loader = None
	loader_name: Union[str, None] = None
	default_model: Union[str, None] = None
	models_dir: Union[str, None] = None

	@classmethod
	@property
	def instance(cls):
		if not cls._instance:
			cls._instance = cls()
		return cls._instance

	def pick_client(self, model_name: str) -> str:
		raise NotImplementedError()

	def get_models_dir(self) -> str:
		raise NotImplementedError()

	def get_default_model(self) -> str:
		raise NotImplementedError()

	def load_model(self, model_name: Union[str, None]):
		if model_name is None or model_name == '':
			# if self.default_model is None:
			# 	raise Exception('No default model set')
			d = self.get_default_model()
			if d is None:
				raise Exception('No default model set')
			model_name = d
		if model_name == self.model_name:
			return
		print('Loading model', model_name)
		client_key = self.pick_client(model_name)
		client_instance = self.clients.get(client_key)
		if client_instance:
			self.loader_name = client_key
			self.loader = client_instance
			self.loader.load_model(model_name)
			self.model_name = model_name

	def unload_model(self):
		if self.loader:
			self.loader.unload_model()
		self.loader = None
		self.loader_name = None
		self.model_name = None

	def list_models(self):
		raise NotImplementedError()
