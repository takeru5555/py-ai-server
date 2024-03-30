from pydantic import BaseModel
from typing import Union

class TranscribeOptions(BaseModel):
	"""Options for transcribing audio."""
	file_path: str
	diarize: bool = False
	# json or text
	result_format: str = 'json'

class TranscribeChunk(BaseModel):
	"""Chunk of transcription."""
	start: float
	end: float
	speech: str
	speaker: str = ''

class TranscribeResponse(BaseModel):
	"""Response for transcribing audio."""
	# result: list[TranscribeChunk] = []
	result: Union[list[TranscribeChunk], str] = []
