import re, subprocess
from .base import STTClient_Base
from py_api.models.stt.stt_client import TranscribeOptions, TranscribeResponse

class STTClient_WhisperCpp(STTClient_Base):
	whisper_cpp_path = '/home/user/whisper.cpp'

	def transcribe(
		self, options: TranscribeOptions
	) -> TranscribeResponse:
		try:
			whisper_cmd = [
				f"{self.whisper_cpp_path}/main", "-m",
				f"{self.whisper_cpp_path}/models/ggml-base.bin", "-f",
				options.file_path, "-ml", "16"
			]

			process = subprocess.Popen(
				whisper_cmd,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True
			)

			stdout, stderr = process.communicate()

			if process.returncode != 0:
				raise RuntimeError(f"Error in transcription: {stderr}")

			parts = self.parse_transcription(stdout)
			return TranscribeResponse.model_validate({'parts': parts})

		except Exception as e:
			raise RuntimeError(f"Error in transcription: {e}")

	def parse_transcription(self, transcription_output):
		parts = []
		lines = transcription_output.strip().split('\n')
		for line in lines:
			match = re.match(
				r"\[([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}) --> ([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3})\](.*)",
				line
			)
			if match:
				start, end, speech = match.groups()
				speech = speech.strip()
				parts.append({
					'start': start,
					'end': end,
					'speech': speech
				})
		return parts
