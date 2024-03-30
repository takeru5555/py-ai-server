import logging, os, time
from py_api.args import Args
from py_api.models.llm.client import CompletionOptions, CompletionOptions_Transformers
from .base import LLMClient_Base
from ._utils import text_completion
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logger = logging.getLogger('Transformers-client')

class LLMClient_Transformers(LLMClient_Base):
	device = torch.device(
		'cuda:0' if torch.cuda.is_available() else 'cpu'
	)
	OPTIONS_MAP = {
		'temp': 'temperature',
		'max_tokens': 'max_new_tokens',
		'typical': 'typical_p',
		'repeat_pen': 'repetition_penalty',
	}

	def convert_options(
		self, options: CompletionOptions
	) -> CompletionOptions_Transformers:
		new_options = self.map_options_from_model(
			options, CompletionOptions_Transformers
		)
		assert isinstance(
			new_options, CompletionOptions_Transformers
		)
		return new_options

	def load_model(self, model_name=''):
		model_name = model_name or Args['llm_model']
		models_dir = Args['llm_models_dir']

		is_dir = os.path.isdir(os.path.join(models_dir, model_name))
		if not is_dir:
			raise Exception(
				f'Model {model_name} not found in {models_dir}.'
			)
		self.model_abspath = os.path.join(models_dir, model_name)
		self.model_name = model_name

		if self.loaded or self.model is not None:
			self.unload_model()

		self.tokenizer = AutoTokenizer.from_pretrained(
			self.model_abspath
		)
		if self.config is None:
			cfg = {
				'model_name_or_path': self.model_abspath,
				'low_cpu_mem_usage': True,
				'device_map': self.device,
			}
			self.config = cfg

		logger.debug(f'Loading model {self.model_name}...')
		start = time.time()
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_abspath,
			low_cpu_mem_usage=True,
			device_map=self.device,
		)

		end = time.time()
		logger.debug(
			f'Loaded model {self.model_name} in {end - start}s'
		)
		self.loaded = True

	def unload_model(self):
		if self.model is None:
			return
		self.model = None
		self.model_name = None
		self.model_abspath = None
		self.cache = None
		self.config = None
		self.tokenizer = None
		self.generator = None
		self.loaded = False
		torch.cuda.empty_cache()
		logger.debug('Unloaded model.')

	def complete(self, options: CompletionOptions_Transformers):
		if not self.loaded or self.model is None:
			self.load_model()
		assert isinstance(options, CompletionOptions_Transformers)
		opt = options.model_dump()
		del opt['prompt']
		pipe = pipeline(
			'text-generation',
			model=self.model,
			tokenizer=self.tokenizer,
			**opt
		)

		result = pipe(options.prompt)
		result = result[0]['generated_text']  # type: ignore
		# remove prompt from result
		result = result[len(options.prompt):]
		assert isinstance(result, str)

		r = text_completion(
			result,
			model_name=self.model_name or '',
		)
		return {
			'result': r,
			'params': options.model_dump(),
		}
