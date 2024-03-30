from typing import Union
from py_api.models.tts.tts_client import SpeakOptions, SpeakToFileOptions, SpeakResponse, SpeakToFileResponse
from py_api.settings import DEVICE_MAP

# TODO i guess do we need a Base class for these Base classes?
#   could have helper for converting options in diff ways
#   have common methods load/unload then inheriting Base classes can implement their own methods to run inference
#   also `list_models`, which would only be implemented by clients that use different source for models (e.g. llm/openai)
# TODO this base class should have:
# `list_voices` for TTS clients that support it

class TTSClient_Base:
	_instance = None
	# cache = None
	config = None
	device = DEVICE_MAP['tts']
	# tokenizer = None
	# generator = None
	loaded = False
	model = None
	model_name: Union[str, None] = None
	# model_abspath: Union[str, None] = None

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

	def load_model(self, model_name: str):
		raise NotImplementedError()

	def unload_model(self):
		raise NotImplementedError()

	def speak(self, options: SpeakOptions) -> SpeakResponse:
		"""Generate audio from a prompt. Returns wav file data."""
		raise NotImplementedError()

	def speak_to_file(
		self, options: SpeakToFileOptions
	) -> SpeakToFileResponse:
		"""Generate audio from a prompt and save to file on server."""
		raise NotImplementedError()

	# def add_voice(self, voice: str, voice_path: str):
	# voice_path should be speaker file already saved to disk
	# (api should handle saving the upload)
	# 2nd thought, maybe this doesn't belong here: once uploaded we just need to pass the file for the speaker_wav (we're not fine tuning)
	# 3rd thought, maybe this does belong here: other tts clients could have voice cloning, so maybe we should have voice management stuff in this base class
