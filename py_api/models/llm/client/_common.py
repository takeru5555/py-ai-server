from typing import Optional, Any
from pydantic import BaseModel, Field

class CompletionOptions(BaseModel):
	"""Generic model, to be passed to manager, which will convert to model-specific options."""
	# Common options
	model: str = Field(
		'',
		description=
		'Model to use. If local, model will be loaded (current model will be unloaded). If blank, current (local) model will be used.'
	)
	prompt: str = Field(
		..., description='Prompt to feed to model.'
	)
	messages: list[Any] = Field(
		[], description='Messages to feed to (openai) model.'
	)
	temp: float = Field(
		0.7, description='Temperature for sampling.'
	)
	max_tokens: int = Field(
		128, description='Maximum number of tokens to generate.'
	)
	top_p: float = Field(
		0.9, description='Top-p (nucleus) sampling cutoff.'
	)
	min_p: float = Field(
		0.0, description='Minimum probability for top-p sampling.'
	)
	top_k: int = Field(40, description='Top-k sampling cutoff.')
	typical: float = Field(0.01, description='Typical_p')
	tfs: float = Field(0.0, description='Tail free sampling.')
	repeat_pen: float = Field(
		1.05, description='Repetition penalty.'
	)
	mirostat: bool = Field(
		False,
		description=
		'Whether to use mirostat. Will use appropriate value for current loader.'
	)
	mirostat_tau: float = Field(0.0, description='Mirostat tau.')
	mirostat_eta: float = Field(0.0, description='Mirostat eta.')
	stop: list[str] = Field([], description='Stop strings.')

	# LlamaCppPython options
	frequency_penalty: Optional[float] = Field(
		None, description='Frequency penalty. LlamaCpp only.'
	)
	presence_penalty: Optional[float] = Field(
		None, description='Presence penalty. LlamaCpp only.'
	)
	seed: Optional[int] = Field(
		None, description='Seed. LlamaCpp only.'
	)
	grammar: Optional[str] = Field(
		None,
		description=
		'Grammar to use for constrained sampling. LlamaCpp only.'
	)

	# Exllamav2 options
	token_repetition_range: Optional[int] = Field(
		-1, description='Token repetition range. Exllamav2 only.'
	)
	token_repetition_decay: Optional[float] = Field(
		0.0, description='Token repetition decay. Exllamav2 only.'
	)
	temperature_last: Optional[bool] = Field(
		False,
		description='Whether to use temperature last. Exllamav2 only.'
	)
	mirostat_mu: Optional[Any] = Field(
		None, description='Mirostat mu. Exllamav2 only.'
	)

	# TODO
	# Transformers options
