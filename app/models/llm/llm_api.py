from typing import Union, Any
from pydantic import BaseModel, Field
from py_api.models.llm.client import CompletionOptions, MessageObject

class PromptPart(BaseModel):
	use: bool = Field(
		True, description='Whether to use prompt part.'
	)
	val: str = Field(..., description='Prompt part value.')
	pre: str = Field(
		'', description='Prefix to add to prompt part.'
	)
	suf: str = Field(
		'', description='Suffix to add to prompt part.'
	)

class PromptParts(BaseModel):
	user: list[PromptPart] = Field(
		..., description='User prompt parts.'
	)
	system: list[PromptPart] = Field(
		[], description='System prompt parts.'
	)
	prior_msgs: list[MessageObject] = Field(
		[], description='Prior messages to add to prompt.'
	)

class CompletionRequest(CompletionOptions):
	prompt: str = Field('', description='Prompt to feed to model.')
	parts: PromptParts = Field(
		None, description='Prompt parts to construct prompt from.'
	)
	messages: list[MessageObject] = Field(
		[], description='Messages to feed to model.'
	)
	prefix_response: str = Field(
		'', description='Prefix to add to prompt.'
	)
	model: str = Field(
		'',
		description=
		'Model to use. If local, model will be loaded (current model will be unloaded). If blank, current (local) model will be used.',
		examples=['mistral-7b.Q4.gguf', 'openai:gpt-3.5-turbo']
	)

class CompletionUsage(BaseModel):
	prompt_tokens: int = Field(
		0, description='Number of tokens in prompt.'
	)
	completion_tokens: int = Field(
		0, description='Number of tokens in completion.'
	)
	total_tokens: int = Field(
		0, description='Total number of tokens.'
	)

class CompletionChoice(BaseModel):
	text: str = Field(..., description='Completion text.')
	index: int = Field(..., description='Completion index.')
	# logprobs: Union[dict, None] = Field(..., description='Completion logprobs.')
	finish_reason: str = Field(
		...,
		description='Completion finish reason.',
		examples=['stop', 'length']
	)
	tool_calls: list[Any] = Field(
		[], description='Completion tool calls.'
	)

class CompletionResult(BaseModel):
	id: str = Field(..., description='Completion ID.')
	object: str = Field(..., description='Completion object.')
	created: int = Field(
		..., description='Completion creation timestamp.'
	)
	model: str = Field(..., description='Completion model.')
	choices: list[CompletionChoice] = Field(
		..., description='Completion choices.'
	)
	usage: CompletionUsage = Field(
		..., description='Completion usage.'
	)

class CompletionResponse(BaseModel):
	result: CompletionResult = Field(
		..., description='Completion result.'
	)
	params: dict[
		str, Any] = Field(..., description='Completion parameters.')

class CompletionReturn(BaseModel):
	result: CompletionResult = Field(
		..., description='Completion result.'
	)
	params: dict[
		str, Any] = Field(..., description='Completion parameters.')

class DownloadModelRequest(BaseModel):
	model: str = Field(
		...,
		description='Model to download.',
		examples=['username/model_name[:branch]']
	)

class DownloadModelResponse(BaseModel):
	status: str = Field(
		...,
		description='Model status.',
		examples=['Downloaded', 'Error']
	)
	model: str = Field(
		...,
		description='Model name.',
		examples=['username/model_name[:branch]']
	)
	time: float = Field(..., description='Time to download model.')
	error: Union[
		str, None] = Field(None, description='Error message.')

class GetModelResponse(BaseModel):
	model: str = Field(
		...,
		description='Model name.',
		examples=['username/model_name[:branch]']
	)
	loader_name: str = Field(
		...,
		description='Loader name.',
		examples=['llamacpp', 'exllamav2', 'transformers']
	)

class LoadModelRequest(BaseModel):
	model: str = Field(
		...,
		description='Model to load.',
		examples=['username/model_name[:branch]']
	)

class LoadModelResponse(BaseModel):
	status: str = Field(
		...,
		description='Model status.',
		examples=['Loaded', 'Error']
	)
	model: str = Field(
		...,
		description='Model name.',
		examples=['username/model_name[:branch]']
	)
	loader_name: Union[str, None] = Field(
		None,
		description='Loader name.',
		examples=['llamacpp', 'exllamav2', 'transformers']
	)
	time: float = Field(..., description='Time to load model.')
	error: Union[
		str, None] = Field(None, description='Error message.')

class UnloadModelRequest(BaseModel):
	status: str = Field('Unloaded', description='Model status.')
	model: str = Field(
		...,
		description='Model name.',
		examples=['username/model_name[:branch]']
	)
	time: float = Field(..., description='Time to unload model.')

class ListModelsResponse(BaseModel):
	models: list[str] = Field(..., description='List of models.')
