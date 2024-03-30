from typing import Union
from py_api.client.base_manager import BaseAIManager
from py_api.client.stt import STTClient_WhisperCpp, STTClient_WhisperX
from py_api.models.stt.stt_client import TranscribeOptions, TranscribeResponse

class STTManager(BaseAIManager):
	clients = {
		'whispercpp': STTClient_WhisperCpp.instance,
		'whisperx': STTClient_WhisperX.instance
	}
	model = None  # either whispercpp or whisperx

	def load_model(self, model_name: str | None):
		if self.model:
			self.unload_model()
		try:
			if model_name == 'whispercpp':
				self.model = 'whispercpp'
			elif model_name == 'whisperx':
				self.model = 'whisperx'
				self.clients['whisperx'].load_model(model_name)
		except Exception as e:
			print('Failed to load model', e)
			self.unload_model()
			self.model = None

	def unload_model(self):
		if self.model == 'whispercpp':
			pass
		elif self.model == 'whisperx':
			self.clients['whisperx'].unload_model()
		self.model = None

	def transcribe(
		self,
		file_path: str,
		diarize: bool = False,
		result_format: str = 'json'
	) -> TranscribeResponse:
		if not self.model:
			self.load_model('whisperx')
			if not self.model:
				raise Exception('No model loaded')
		client = self.clients[self.model]
		assert isinstance(
			client, STTClient_WhisperCpp
		) or isinstance(client, STTClient_WhisperX)
		return client.transcribe(
			TranscribeOptions.model_validate({
				'file_path':
				file_path,
				'diarize':
				diarize,
				'result_format':
				result_format
			})
		)
