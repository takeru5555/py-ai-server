import logging, os, time
from py_api.args import Args
from py_api.models.llm.client import CompletionOptions, CompletionOptions_Exllamav2
from .base import LLMClient_Base
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
import torch
from ._utils import text_completion

logger = logging.getLogger('Exllamav2-client')

class LLMClient_Exllamav2(LLMClient_Base):
	device = torch.device(
		'cuda' if torch.cuda.is_available() else 'cpu'
	)
	OPTIONS_MAP = {
		'temp': 'temperature',
		'repeat_pen': 'token_repetition_penalty',
	}

	def convert_options(
		self, options: CompletionOptions
	) -> CompletionOptions_Exllamav2:
		new_options = self.map_options_from_model(
			options, CompletionOptions_Exllamav2
		)
		assert isinstance(new_options, CompletionOptions_Exllamav2)
		return new_options

	def load_model(self, model_name=''):
		model_name = model_name or Args['llm_model']
		models_dir = Args['llm_models_dir']

		# model_name should be name of directory in models_dir
		is_dir = os.path.isdir(os.path.join(models_dir, model_name))
		if not is_dir:
			raise Exception(
				f'Model {model_name} not found in {models_dir}.'
			)
		self.model_abspath = os.path.join(models_dir, model_name)
		self.model_name = model_name

		if self.loaded or self.model is not None:
			self.unload_model()

		if self.config is None:
			cfg = ExLlamaV2Config()
			cfg.model_dir = self.model_abspath
			cfg.prepare()
			self.config = cfg

		logger.debug(f'Loading model {self.model_name}...')
		start = time.time()
		self.model = ExLlamaV2(self.config, lazy_load=True)
		self.cache = ExLlamaV2Cache(self.model, lazy=True)
		self.model.load_autosplit(self.cache)

		self.tokenizer = ExLlamaV2Tokenizer(self.config)
		self.generator = ExLlamaV2StreamingGenerator(
			self.model, self.cache, self.tokenizer
		)
		self.generator.warmup()

		end = time.time()
		logger.debug(
			f'Loaded model {self.model_name} in {end - start}s'
		)
		self.loaded = True

	def unload_model(self):
		if self.model is None:
			return
		self.model.unload()
		logger.debug('Unloaded model.')
		self.model = None
		self.model_name = None
		self.model_abspath = None
		self.cache = None
		self.config = None
		self.tokenizer = None
		self.generator = None
		self.loaded = False
		torch.cuda.empty_cache()

	def generate(self, options: CompletionOptions_Exllamav2):
		if not self.loaded or self.model is None or self.generator is None or self.tokenizer is None or self.cache is None:
			raise Exception('Re-load model.')

		settings = ExLlamaV2Sampler.Settings()
		settings.temperature = options.temperature
		settings.top_p = options.top_p
		settings.token_repetition_penalty = options.token_repetition_penalty
		settings.token_repetition_range = options.token_repetition_range
		settings.token_repetition_decay = options.token_repetition_decay
		settings.disallow_tokens(
			self.tokenizer, []
		)  # would ban eos token here?

		start = time.time()

		input_ids = self.tokenizer.encode(options.prompt)
		input_ids.to(self.device)

		if options.stop:
			self.generator.set_stop_conditions(options.stop)
		self.generator.begin_stream(input_ids, settings)

		generated_tokens = 0
		while True:
			chunk, eos, _ = self.generator.stream()
			generated_tokens += 1
			if eos or generated_tokens >= options.max_tokens:
				break
			yield chunk

		end = time.time()
		logger.debug(f'Generated text in {end - start}s')

	def complete(self, options: CompletionOptions_Exllamav2):
		if not self.loaded or self.model is None:
			self.load_model()
		assert isinstance(options, CompletionOptions_Exllamav2)
		result = ''
		tokens = 0
		for chunk in self.generate(options):
			result += chunk
			tokens += 1

		r = text_completion(
			result,
			model_name=self.model_name or '',
			tokens=tokens,
			max_tokens=options.max_tokens
		)
		return {
			'result': r,
			'params': options.model_dump(),
		}
