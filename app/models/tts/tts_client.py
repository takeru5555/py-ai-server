from typing import Optional, List, Any
from pydantic import BaseModel, Field

class SpeakOptions(BaseModel):
	"""Generic options to generate TTS audio."""
	text: str = Field(
		..., description='Text to generate audio from.'
	)
	split_sentences: bool = Field(
		True,
		description='Whether to split sentences when text is too long.'
	)
	language: str = Field('en', description='Language to use.')

	# TODO?
	voice: str = Field('default', description='Voice to use.')

class SpeakResponse(BaseModel):
	"""Generic response to generate TTS audio."""
	audio: str = Field(
		..., description='Audio data encoded in base64.'
	)
	time: float = Field(
		..., description='Time taken to generate audio in seconds.'
	)

class SpeakToFileOptions(SpeakOptions):
	"""Generic options to generate and save TTS audio to file on server."""
	file: str = Field('', description='File to save audio to.')

class SpeakToFileResponse(BaseModel):
	"""Generic response to generate and save TTS audio to file on server."""
	file_name: str = Field(
		..., description='File name of saved audio on server.'
	)
	time: float = Field(
		..., description='Time taken to generate audio in seconds.'
	)
