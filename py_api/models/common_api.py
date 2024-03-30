from typing import Union
from pydantic import BaseModel, Field

class GetModelResponse(BaseModel):
	"""Get currently loaded AI model."""
	model: str = Field(..., description='Model name.')

class ListModelsResponse(BaseModel):
	"""List available AI models."""
	models: list[str] = Field(..., description='Available models.')

class LoadModelRequest(BaseModel):
	model: str = Field(
		...,
		description='Model to load.',
		examples=[
			'username/model_name[:branch]', 'model.safetensors'
		]
	)

class LoadModelResponse(BaseModel):
	"""Load AI model."""
	status: str = Field(
		...,
		description='Model status.',
		examples=['Loaded', 'Error']
	)
	model: str = Field(..., description='Model name.')
	time: float = Field(..., description='Time to load model.')
	error: Union[
		str, None] = Field(None, description='Error message.')

class UnloadModelResponse(BaseModel):
	"""Unload AI model."""
	status: str = Field('Unloaded', description='Model status.')
	model: str = Field(..., description='Model name.')
	time: float = Field(..., description='Time to unload model.')
