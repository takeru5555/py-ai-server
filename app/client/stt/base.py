from py_api.models.stt.stt_client import TranscribeOptions, TranscribeResponse
from py_api.settings import DEVICE_MAP

class STTClient_Base:
	_instance = None
	device = DEVICE_MAP['stt']

	@classmethod
	@property
	def instance(cls):
		if not cls._instance:
			cls._instance = cls()
		return cls._instance

	def load_model(self, model_name: str | None):
		raise NotImplementedError()

	def transcribe(
		self, options: TranscribeOptions
	) -> TranscribeResponse:
		raise NotImplementedError()
