from typing import Any, Optional
from pydantic import BaseModel, Field

class CompletionOptions_LlamaCppPython(BaseModel):
	prompt: str = Field(
		..., description='Prompt to feed to model.'
	)
	temperature: Optional[float] = Field(
		0.8, description='Temperature for sampling.'
	)
	max_tokens: Optional[int] = Field(
		16, description='Maximum number of tokens to generate.'
	)
	top_p: Optional[float] = Field(
		0.95, description='Top-p (nucleus) sampling cutoff.'
	)
	min_p: Optional[float] = Field(
		0.05, description='Minimum probability for top-p sampling.'
	)
	typical_p: Optional[float] = Field(
		1.0, description='Typical_p'
	)
	stop: Optional[list[str]] = Field([],
																		description='Stop strings.')
	frequency_penalty: Optional[float] = Field(
		0.0, description='Frequency penalty.'
	)
	presence_penalty: Optional[float] = Field(
		0.0, description='Presence penalty.'
	)
	repeat_penalty: Optional[float] = Field(
		1.1, description='Repetition penalty.'
	)
	top_k: Optional[int] = Field(
		40, description='Top-k sampling cutoff.'
	)
	seed: Optional[int] = Field(-1, description='Seed.')
	tfs_z: Optional[float] = Field(
		1.0, description='Tail free sampling.'
	)
	mirostat_mode: Optional[int] = Field(
		0, description='Whether to use mirostat.'
	)
	mirostat_tau: Optional[float] = Field(
		5.0, description='Mirostat tau.'
	)
	mirostat_eta: Optional[float] = Field(
		0.1, description='Mirostat eta.'
	)
	grammar: Optional[Any] = Field(
		None, description='Instance of LlamaGrammar.'
	)
