from typing import Any, Optional
from pydantic import BaseModel, Field

class MessageObject(BaseModel):
	role: str = Field(
		...,
		description='Message role.',
		examples=['system', 'user', 'assistant']
	)
	content: str = Field(..., description='Message content.')

class CompletionOptions_OpenAI(BaseModel):
	model: str = Field(
		...,
		description=
		'Model to use. If local, model will be loaded (current model will be unloaded). If blank, current (local) model will be used.',
		examples=['mistral-7b.Q4.gguf', 'openai:gpt-3.5-turbo']
	)
	messages: list[MessageObject] = Field(
		..., description='Messages to feed to model.'
	)
	frequency_penalty: Optional[float] = Field(
		0.0, description='Frequency penalty (-2.0 to 2.0).'
	)
	max_tokens: Optional[int] = Field(
		56, description='Maximum number of tokens to generate.'
	)
	presence_penalty: Optional[float] = Field(
		0.0, description='Presence penalty (-2.0 to 2.0).'
	)
	seed: Optional[int] = Field(-1, description='Seed.')
	stop: Optional[list[str]] = Field([],
																		description='Stop strings.')
	temperature: Optional[float] = Field(
		0.8, description='Temperature for sampling.'
	)
	top_p: Optional[float] = Field(
		0.95, description='Top-p (nucleus) sampling cutoff.'
	)
