from typing import Generator, List, Dict, Union, Any
from py_api.models.llm.llm_api import CompletionReturn
from py_api.models.llm.client import CompletionOptions, CompletionOptions_LlamaCppPython, CompletionOptions_Exllamav2
from py_api.settings import DEVICE_MAP

class LLMClient_Base:
	_instance = None
	cache = None
	config = None
	device = DEVICE_MAP['llm']
	tokenizer = None
	generator = None
	loaded = False
	model = None
	model_name: Union[str, None] = None
	model_abspath: Union[str, None] = None

	OPTIONS_MAP: dict[str, str] = {}

	@classmethod
	@property
	def instance(cls):
		if not cls._instance:
			cls._instance = cls()
		return cls._instance

	def map_options_from_model(
		self, options: CompletionOptions, model: Any
	) -> Union[CompletionOptions_LlamaCppPython,
							CompletionOptions_Exllamav2]:
		if options is None or (
			options.prompt is None and len(options.messages) == 0
		):
			raise Exception('No prompt provided.')
		obj = {}
		if options.prompt is not None:
			obj['prompt'] = options.prompt
			obj['messages'] = []
		if options.messages is not None:
			obj['messages'] = options.messages
			obj['prompt'] = ''
			obj['model'] = options.model
		new_options = model.model_validate(obj)
		newOptions = {}
		keys = list(options.model_fields.keys())
		newkeys = list(new_options.model_fields.keys())

		optObj = options.model_dump(exclude_unset=True)
		optObj = {k: v for k, v in optObj.items() if v is not None}
		schema = model.schema()
		for key in keys:
			if key in self.OPTIONS_MAP:
				remappedkey = self.OPTIONS_MAP[key]
				if remappedkey in newkeys:
					value = optObj[key]
					newOptions[remappedkey] = value

			elif key in newkeys:
				# common option
				if key in optObj:
					newOptions[key] = optObj[key]
				else:
					newOptions[key] = schema['properties'][key]['default']

		new_options = model.model_validate(newOptions)

		return new_options

	def convert_options(self, options: CompletionOptions) -> Any:
		"""Convert options from common names to model-specific names and values."""
		raise NotImplementedError()

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

	def complete(
		self, options: Union[CompletionOptions_LlamaCppPython,
													CompletionOptions_Exllamav2]
	) -> CompletionReturn:
		"""Generate text from a prompt. Returns a string."""
		raise NotImplementedError()

	def chat(
		self, messages: List[Dict],
		options: Union[CompletionOptions_LlamaCppPython,
										CompletionOptions_Exllamav2]
	):
		# not implemented anywhere
		raise NotImplementedError()
