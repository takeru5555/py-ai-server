from typing import Generator, List, Dict, Union, Any
from pydantic import BaseModel
from py_api.models.img.img_client import Txt2ImgOptions, Txt2ImgResponse
from py_api.settings import DEVICE_MAP

class ImgClient_Base:
	_instance = None
	config = None
	device = DEVICE_MAP['img']
	loaded = False
	model = None
	model_name: Union[str, None] = None
	model_abspath: Union[str, None] = None

	# OPTIONS_MAP: dict[str, str] = {}

	@classmethod
	@property
	def instance(cls):
		if not cls._instance:
			cls._instance = cls()
		return cls._instance

	# def map_options_from_moodel(self, options: CompletionOptions, model: Any) -> Union[CompletionOptions_LlamaCppPython, CompletionOptions_Exllamav2]:
	# 	if options is None or options.prompt is None:
	# 		raise Exception('No prompt provided.')
	# 	new_options = model.model_validate({'prompt': options.prompt})
	# 	newOptions = {}
	# 	keys = list(options.model_fields.keys())
	# 	newkeys = list(new_options.model_fields.keys())

	# 	optObj = options.model_dump()
	# 	for key in keys:
	# 		if key in self.OPTIONS_MAP:
	# 			remappedkey = self.OPTIONS_MAP[key]
	# 			if remappedkey in newkeys:
	# 				value = optObj[key]
	# 				newOptions[remappedkey] = value

	# 		elif key in newkeys:
	# 			# common option
	# 			newOptions[key] = optObj[key]
	# 			# print(f'setting {key} to {opt[key]}')

	# 	new_options = model.model_validate(newOptions)

	# 	return new_options

	# def convert_options(self, options: CompletionOptions) -> Any:
	# 	"""Convert options from common names to model-specific names and values."""
	# 	raise NotImplementedError()

	def list_samplers(self) -> List[str]:
		raise NotImplementedError()

	def get_sampler(self, sampler_name: str) -> Any:
		raise NotImplementedError()

	def load_model(self, model_name: str):
		raise NotImplementedError()

	def unload_model(self):
		raise NotImplementedError()

	def txt2img(
		self, gen_options: Txt2ImgOptions
	) -> Txt2ImgResponse:
		raise NotImplementedError()

	# def img2img(self, gen_options):
