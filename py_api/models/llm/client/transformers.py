from typing import Any, List, Optional
from pydantic import BaseModel, Field

class CompletionOptions_Transformers(BaseModel):
	max_new_tokens: int = Field(
		0, description='Maximum number of tokens to generate.'
	)
	min_new_tokens: int = Field(
		0, description='Minimum number of tokens to generate.'
	)
	max_time: int = Field(
		10, description='Maximum time for generation in seconds.'
	)
	do_sample: bool = Field(
		True,
		description=
		'Whether or not to use sampling; uses greedy decoding otherwise.'
	)
	use_cache: bool = Field(
		True,
		description=
		'Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.'
	)
	temperature: float = Field(
		1.0,
		description=
		'Value used to module the next token probabilities.'
	)
	top_k: int = Field(
		50,
		description=
		'The number of highest probability vocabulary tokens to keep for top-k-filtering.'
	)
	top_p: float = Field(
		1.0,
		description=
		'If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
	)
	typical_p: float = Field(
		1.0,
		description=
		'Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation.'
	)
	repetition_penalty: float = Field(
		1.0, description='Repetition penalty. 1 = Off'
	)
	encoder_repetition_penalty: float = Field(
		1.0,
		description=
		'An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.'
	)
	guidance_scale: float = Field(
		1.0,
		description=
		'The guidance scale for classifier free guidance (CFG). CFG is enabled by setting guidance_scale > 1. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality.'
	)

	prompt: str = Field(
		..., description='Prompt to feed to model.'
	)
	# token_repetition_range: int = Field(-1, description='Token repetition range.')
	# token_repetition_decay: int = Field(0, description='Token repetition decay.')
	# stop: list[str] = Field([], description='Stop strings.')
	# tfs: float = Field(0.0, description='Tail free sampling.')
	# temperature_last: bool = Field(False, description='Whether to use temperature last.')
	# mirostat: bool = Field(False, description='Whether to use mirostat.')
	# mirostat_tau: float = Field(1.5, description='Mirostat tau.')
	# mirostat_eta: float = Field(0.1, description='Mirostat eta.')
	# mirostat_mu: Any = Field(None, description='Mirostat mu.')
