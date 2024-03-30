import base64, logging, os, time
from py_api.args import Args
from py_api.models.tts.tts_client import SpeakOptions, SpeakToFileOptions, SpeakResponse, SpeakToFileResponse
from .base import TTSClient_Base
from TTS.api import TTS
import torch

logger = logging.getLogger('XTTS-client')
DEFAULT_VOICE = 'jaiden-10s.wav'

# might need to think of something for supporting bark model (tts_to_file takes voice_dir with bark, but not other models?)
# see https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/bark.md

class TTSClient_Coqui(TTSClient_Base):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def get_voice_path(self, voice: str) -> str:
		if voice is None or voice == '':
			raise Exception('No voice provided.')
		if '.wav' not in voice:
			voice += '.wav'
		return os.path.join(Args['tts_voices_dir'], voice)

	def load_model(self, model_name: str):
		if self.model_name == model_name:
			return
		start = time.time()
		if self.model_name is not None:
			self.unload_model()
		voices_dir = Args['tts_voices_dir']
		if not os.path.exists(voices_dir):
			os.makedirs(voices_dir)
		logger.debug(model_name)
		try:
			self.model = TTS(model_name
												).to(self.device, non_blocking=True)
			self.model_name = model_name
			self.loaded = True
		except Exception as e:
			self.model = None
			self.model_name = None
			self.loaded = False
			raise e
		logger.debug(
			f'Loaded model in {time.time() - start} seconds.'
		)

	def unload_model(self):
		start = time.time()
		if self.model_name is None:
			return
		self.model = None
		self.model_name = None
		self.loaded = False
		torch.cuda.empty_cache()
		logger.debug(
			f'Unloaded model in {time.time() - start} seconds.'
		)

	def speak(self, gen_options: SpeakOptions) -> SpeakResponse:
		if not self.loaded or self.model is None:
			raise Exception('Model not loaded.')
		if gen_options.text is None:
			raise Exception('No text provided.')
		start = time.time()
		tmp_name = f'tmp-{time.time()}.wav'
		tmp_name = os.path.join(Args['tts_output_dir'], tmp_name)
		voice = gen_options.voice if gen_options.voice != '' else 'default'
		if voice is not None and voice != 'default':
			voice = self.get_voice_path(voice)
		else:
			voice = self.get_voice_path(DEFAULT_VOICE)
		try:
			audio = self.model.tts_to_file(
				text=gen_options.text,
				language=gen_options.language,
				file_path=tmp_name,
				speaker_wav=voice
			)
		except Exception as e:
			msg = str(e)
			print(msg)
		end = time.time()
		# read and return file content
		with open(tmp_name, 'rb') as f:
			audio = f.read()
		os.remove(tmp_name)
		b64 = base64.b64encode(audio).decode('utf-8')
		return SpeakResponse.model_validate({
			'audio': b64,
			'time': end - start
		})

	def speak_to_file(
		self, gen_options: SpeakToFileOptions
	) -> SpeakToFileResponse:
		if not self.loaded or self.model is None:
			raise Exception('Model not loaded.')
		if gen_options.text is None:
			raise Exception('No text provided.')
		start = time.time()
		file = gen_options.file if gen_options.file != '' else f'tts-{time.time()}.wav'
		file_path = os.path.join(Args['tts_output_dir'], file)
		voice = gen_options.voice if gen_options.voice != '' else 'default'
		if voice is not None and voice != 'default':
			voice = self.get_voice_path(voice)
		else:
			voice = self.get_voice_path(DEFAULT_VOICE)
		self.model.tts_to_file(
			text=gen_options.text,
			language=gen_options.language,
			file_path=file_path,
			speaker_wav=voice
		)
		end = time.time()
		return SpeakToFileResponse.model_validate({
			'file_name':
			gen_options.file,
			'time':
			end - start
		})
