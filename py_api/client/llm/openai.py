from typing import Generator, List, Dict, Union
import requests
from py_api.models.llm.llm_api import CompletionResult
from py_api.models.llm.client import CompletionOptions, CompletionOptions_LlamaCppPython, CompletionOptions_Exllamav2, CompletionOptions_OpenAI
from .base import LLMClient_Base
from ._utils import text_completion
from py_api.settings import OPENAI_API_KEY

# so, with this there is no loading models, you pick which model when calling complete
class LLMClient_OpenAI(LLMClient_Base):
	_instance = None
	cache = None
	config = None
	device = None
	tokenizer = None
	generator = None
	loaded = False
	model = None
	model_name: Union[str, None] = None
	model_abspath: Union[str, None] = None
	key = OPENAI_API_KEY

	OPTIONS_MAP: dict[str, str] = {
		'temp': 'temperature',
	}

	def _api_post(self, url: str, body: dict) -> dict:
		headers = {
			"Authorization": f"Bearer {self.key}",
			"Content-Type": "application/json"
		}
		response = requests.post(url, json=body, headers=headers)
		# response.raise_for_status()
		return response.json()

	def _api_get(self, url: str) -> dict:
		headers = {"Authorization": f"Bearer {self.key}"}
		response = requests.get(url, headers=headers)
		# response.raise_for_status()
		return response.json()

	def list_models(self) -> list[str]:
		url = "https://api.openai.com/v1/models"
		res = self._api_get(url)
		models = []
		prefix = 'openai:'
		for model in res['data']:
			if 'gpt' not in model['id']:
				continue
			models.append(prefix + model['id'])
		return models

	# TODO temporary
	def hasKey(self):
		return self.key is not None and self.key != ''

	def convert_options(
		self, options: CompletionOptions
	) -> CompletionOptions_OpenAI:
		"""Convert options from common names to model-specific names and values."""
		new_options = self.map_options_from_model(
			options, CompletionOptions_OpenAI
		)
		if 'openai:' in options.model:
			new_options.model = options.model.split('openai:')[1]
		else:
			new_options.model = options.model
		assert isinstance(new_options, CompletionOptions_OpenAI)
		return new_options

	def load_model(self, model_name: str):
		raise NotImplementedError()

	def unload_model(self):
		raise NotImplementedError()

	def generate(
		self, options: Union[CompletionOptions_LlamaCppPython,
													CompletionOptions_Exllamav2]
	) -> Generator:
		"""(TODO) Generate text from a prompt. Returns a Generator."""
		raise NotImplementedError()

	def complete(self, options: CompletionOptions_OpenAI):
		"""Generate text from a prompt. Returns a string."""
		if not self.hasKey():
			raise Exception('OpenAI API key not set.')

		url = "https://api.openai.com/v1/chat/completions"
		msgs = []
		for msg in options.messages:
			msgs.append(msg.model_dump())
		body = {
			"model": options.model,
			"messages": msgs,
			"max_tokens": options.max_tokens,
			"temperature": options.temperature,
			"top_p": options.top_p,
			"stop": options.stop,
		}
		res = self._api_post(url, body)
		choice = res['choices'][0]
		r = text_completion(
			choice['message']['content'],
			model_name=options.model,
			finish_reason=choice['finish_reason'],
		)
		return {
			'result': r,
			'params': options.model_dump(),
		}

	def chat(
		self, messages: List[Dict],
		options: Union[CompletionOptions_LlamaCppPython,
										CompletionOptions_Exllamav2]
	):
		# not implemented anywhere
		raise NotImplementedError()
