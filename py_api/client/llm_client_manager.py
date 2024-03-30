from typing import Union, Dict, TypeVar
import logging, os
from py_api.args import Args
from py_api.client.base_manager import BaseAIManager
from py_api.client.llm import LLMClient_LlamaCppPython, LLMClient_Exllamav2, LLMClient_OpenAI, LLMClient_Transformers
from py_api.models.llm.llm_api import CompletionReturn
from py_api.models.llm.client import CompletionOptions, CompletionOptions_LlamaCppPython, CompletionOptions_Exllamav2, CompletionOptions_Transformers

ClientUnion = Union[LLMClient_LlamaCppPython,
										LLMClient_Exllamav2, LLMClient_OpenAI,
										LLMClient_Transformers]
ClientDict = Dict[str, ClientUnion]

EXTENSIONS = ['.gguf', '.ggml', '.safetensor']
logger = logging.getLogger(__name__)

class LLMManager(BaseAIManager):
	clients: ClientDict = {
		'llamacpp': LLMClient_LlamaCppPython.instance,
		'exllamav2': LLMClient_Exllamav2.instance,
		'openai': LLMClient_OpenAI.instance,
		'transformers': LLMClient_Transformers.instance,
	}
	loader: Union[ClientUnion, None] = None

	def __init__(self):
		self.clients = {
			'llamacpp': LLMClient_LlamaCppPython.instance,
			'exllamav2': LLMClient_Exllamav2.instance,
			'openai': LLMClient_OpenAI.instance,
			'transformers': LLMClient_Transformers.instance,
		}
		self.default_model = Args['llm_model']
		self.models_dir = Args['llm_models_dir']

	def get_loader_model(self):
		if self.loader == LLMClient_LlamaCppPython.instance:
			return CompletionOptions_LlamaCppPython
		if self.loader == LLMClient_Exllamav2.instance:
			return CompletionOptions_Exllamav2
		if self.loader == LLMClient_Transformers.instance:
			return CompletionOptions_Transformers

	def list_local_models(self) -> list[str]:
		models_dir = self.get_models_dir()
		if not models_dir or not os.path.isdir(models_dir):
			return []
		model_names = []
		for filename in os.listdir(models_dir):
			path = os.path.join(models_dir, filename)
			if os.path.isfile(path) and os.path.splitext(filename)[
				1] in EXTENSIONS:
				model_names.append(filename)
			if os.path.isdir(path):
				f = filename.lower()
				if 'awq' in f or 'gptq' in f or 'exl2' in f:
					model_names.append(filename)
					continue
				for subfilename in os.listdir(
					os.path.join(models_dir, filename)
				):
					path = os.path.join(models_dir, filename, subfilename)
					if os.path.isfile(path) and os.path.splitext(
						subfilename
					)[1] in EXTENSIONS:
						model_names.append(
							os.path.join(filename, subfilename)
						)
		return model_names

	def list_models(self) -> list[str]:
		models = self.list_local_models()
		models.extend(LLMClient_OpenAI.instance.list_models())
		models.sort()
		return models

	def pick_client(self, model_name: str):
		# open openai models with openai
		# open GPTQ and EXL2 models with exllamav2
		# open AWQ models with transformers
		# open everything else with llamacpp
		models_dir = self.get_models_dir()
		p = os.path.join(models_dir, model_name)
		model_name = model_name.lower()
		if 'openai:' in model_name:
			return 'openai'
		if os.path.isdir(p):
			if 'gptq' in model_name or 'exl2' in model_name:
				return 'exllamav2'
			if 'awq' in model_name:
				return 'transformers'
		return 'llamacpp'

	def get_models_dir(self):
		return Args['llm_models_dir']

	def get_default_model(self):
		return Args['llm_model']

	def load_model(self, model_name: Union[str, None]):
		if model_name is None or model_name == '':
			return super().load_model(model_name)
		if 'openai:' in model_name:
			self.model_name = model_name
			self.loader_name = 'openai'
			return

		return super().load_model(model_name)

	def generate(self, gen_options: CompletionOptions):
		if not self.loader:
			self.load_model(None)
			if not self.loader:
				raise Exception('Model not loaded.')
		options = self.loader.convert_options(gen_options)
		loader_model = self.get_loader_model()
		assert loader_model is not None
		assert isinstance(options, loader_model)
		return self.loader.generate(options)  # type: ignore

	def complete(self, gen_options: CompletionOptions):
		model = gen_options.model
		if model is None or model == '':
			model = self.model_name or Args['llm_model']
			gen_options.model = model

		if 'openai:' in model:
			OpenAI = LLMClient_OpenAI.instance
			opt = OpenAI.convert_options(gen_options)
			result = LLMClient_OpenAI.instance.complete(opt)
		else:
			if not self.loader:
				self.load_model(model or None)
			if not self.loader:
				raise Exception('Model not loaded.')
			options = self.loader.convert_options(gen_options)
			loader_model = self.get_loader_model()
			assert loader_model is not None
			assert isinstance(options, loader_model)
			result = self.loader.complete(options)  # type: ignore

		try:
			validated_result = CompletionReturn.model_validate(result)
		except Exception as e:
			logger.error(
				"Validation failed for model %s with options %s. Error: %s",
				model, gen_options, str(e)
			)
			return None

		return validated_result
