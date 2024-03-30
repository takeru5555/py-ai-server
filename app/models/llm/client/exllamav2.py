from typing import Any
from pydantic import BaseModel, Field

class CompletionOptions_Exllamav2(BaseModel):
	prompt: str = Field(
		..., description='Prompt to feed to model.'
	)
	max_tokens: int = Field(
		128, description='Maximum number of tokens to generate.'
	)
	token_repetition_penalty: float = Field(
		1.15, description='Token repetition penalty.'
	)
	token_repetition_range: int = Field(
		-1, description='Token repetition range.'
	)
	token_repetition_decay: int = Field(
		0, description='Token repetition decay.'
	)
	temperature: float = Field(
		0.9, description='Temperature for sampling.'
	)
	top_k: int = Field(40, description='Top-k sampling cutoff.')
	top_p: float = Field(
		0.9, description='Top-p (nucleus) sampling cutoff.'
	)
	stop: list[str] = Field([], description='Stop strings.')
	min_p: float = Field(
		0.0, description='Minimum probability for top-p sampling.'
	)
	tfs: float = Field(0.0, description='Tail free sampling.')
	typical: float = Field(0.0, description='Typical_p')
	temperature_last: bool = Field(
		False, description='Whether to use temperature last.'
	)
	mirostat: bool = Field(
		False, description='Whether to use mirostat.'
	)
	mirostat_tau: float = Field(1.5, description='Mirostat tau.')
	mirostat_eta: float = Field(0.1, description='Mirostat eta.')
	mirostat_mu: Any = Field(None, description='Mirostat mu.')
