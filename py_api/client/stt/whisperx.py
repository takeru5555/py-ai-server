import datetime, os, re, subprocess
import whisperx
from .base import STTClient_Base
from py_api.models.stt.stt_client import TranscribeOptions, TranscribeResponse
from py_api.settings import HF_TOKEN

class STTClient_WhisperX(STTClient_Base):
	raw_model = None
	align_model = None
	metadata = None
	diarize_model = None
	models = {'raw': None, 'align': None, 'diarize': None}

	def load_model(self, model_name: str | None):
		compute_type = 'float16' if self.device == 'cuda' else 'int8'
		# self.raw_model = whisperx.load_model(
		# 	'large-v2',
		# 	device=self.device,
		# 	compute_type=compute_type,
		# 	language='en'
		# )
		if not self.raw_model:
			self.raw_model = whisperx.load_model(
				'large-v2',
				device=self.device,
				compute_type=compute_type,
				language='en'
			)
		if not self.align_model:
			model_a, metadata = whisperx.load_align_model(
				language_code='en', device=self.device
			)
			self.align_model = model_a
			self.metadata = metadata
			del model_a, metadata
		if HF_TOKEN and not self.diarize_model:
			self.diarize_model = whisperx.DiarizationPipeline(
				use_auth_token=HF_TOKEN, device=self.device
			)

	def unload_model(self):
		import gc
		import torch
		print('Unloading model')
		self.raw_model = None
		self.align_model = None
		self.metadata = None
		self.diarize_model = None
		gc.collect()
		torch.cuda.empty_cache()

	def transcribe(
		self, options: TranscribeOptions
	) -> TranscribeResponse:
		file_path = options.file_path
		exists = os.path.exists(file_path)
		if not exists:
			raise Exception('File does not exist')
		assert self.raw_model
		assert self.align_model
		assert self.metadata

		audio = whisperx.load_audio(options.file_path)
		raw_transcript = self.raw_model.transcribe(
			audio,
			batch_size=16,
			language='en'  # adjust batch_size to fit VRAM
		)
		align_transcript = whisperx.align(
			raw_transcript["segments"],
			self.align_model,
			self.metadata,
			audio,
			device=self.device,
			return_char_alignments=False
		)

		# choose transcript to return
		# if diarize, then diarize and return
		# else return align_transcript
		transcript = align_transcript if not options.diarize else None
		if not transcript:
			if not HF_TOKEN:
				raise Exception('HF_TOKEN not set')
			assert self.diarize_model
			diarize_segments = self.diarize_model(audio)
			diarize_transcript = whisperx.assign_word_speakers(
				diarize_segments, align_transcript
			)
			transcript = diarize_transcript

		if options.result_format == 'text':
			transcript = self.format_str(transcript['segments'])
			return TranscribeResponse(result=transcript)

		parts = self.parse(transcript)
		return TranscribeResponse(result=parts)

	def format_str(self, segments):
		s = ''
		merged_segments = []
		i = 0
		# TODO merge lines from same speaker within 1 second
		while i < len(segments):
			segment = segments[i]
			merged_segments.append(segment)
			i += 1

		hours = False
		last = merged_segments[-1]
		_start = datetime.timedelta(seconds=last['start'])
		_end = datetime.timedelta(seconds=last['end'])
		if _start.seconds > 3600:
			hours = True

		for segment in merged_segments:
			segment['start'] = round(segment['start'], 0)
			segment['end'] = round(segment['end'], 0)
			_start = datetime.timedelta(seconds=segment['start'])
			_end = datetime.timedelta(seconds=segment['end'])

			if not hours:
				_start = str(_start).split(':')[1:]
				_start = ':'.join(_start)
				_end = str(_end).split(':')[1:]
				_end = ':'.join(_end)
			speaker = segment['speaker']
			speech = segment['text']
			s += f"[{_start}-{_end}] {speaker}: {speech}\n"
		return s

	def parse(self, transcript):
		parts = []
		for segment in transcript['segments']:
			part = {
				'start': segment['start'],
				'end': segment['end'],
				'speech': segment['text']
			}
			if 'speaker' in segment:
				part['speaker'] = segment['speaker']
			parts.append(part)
		return parts
