from typing import Union
import logging, os, time
from py_api.args import Args
from py_api.models.llm.client import CompletionOptions, CompletionOptions_LlamaCppPython
from py_api.utils.llm_models import parse_size_and_quant
from .base import LLMClient_Base
from ._utils import text_completion
from llama_cpp import Llama, LlamaGrammar, LlamaCache
import torch

logger = logging.getLogger('LlamaCpp-client')

def LlamaCppConfig(model_path):
	model_file_name = os.path.basename(model_path)
	size, quant = parse_size_and_quant(model_file_name)

	if size != '' and quant != '':
		logger.debug(f'Model Size: {size} | Quant: {quant}')
	return {
		'model_path': model_path,
		'n_threads': 8,
		'n_gpu_layers': 35,  # TODO calc from size and quant
		'n_ctx':
		4096,  # TODO: docs say 0 = from model -- does it work?
		'verbose': False,  # setting LLM_VERBOSE ?
	}

def LlamaCppCompletionConfig(
	prompt: str, max_tokens: int, temperature: Union[int, float],
	top_p: int, repetition_penalty: Union[int, float], seed: int,
	grammar: str, stop: list
):
	config = {
		'prompt': prompt,
		'max_tokens': max_tokens,
		'temperature': temperature,
		'top_p': top_p,
		'repeat_penalty': repetition_penalty,
		'seed': seed,
		'stop': stop
	}
	if grammar != '' and grammar != None:
		# TODO: LlamaGrammar.from_json_schema ???
		# NOTE: from_string seems to fix grammar (like using '+' which leads to issues (i guess because recursion))
		config['grammar'] = LlamaGrammar.from_string(grammar, False)
	return config

class LLMClient_LlamaCppPython(LLMClient_Base):
	OPTIONS_MAP = {
		'temp': 'temperature',
		'typical': 'typical_p',
		'tfs': 'tfs_z',
		'repeat_pen': 'repeat_penalty',
	}

	def convert_options(
		self, options: CompletionOptions
	) -> CompletionOptions_LlamaCppPython:
		new_options = self.map_options_from_model(
			options, CompletionOptions_LlamaCppPython
		)
		assert isinstance(
			new_options, CompletionOptions_LlamaCppPython
		)
		if options.mirostat:
			new_options.mirostat_mode = 1
		if options.grammar:
			new_options.grammar = LlamaGrammar.from_string(
				options.grammar, False
			)
		return new_options

	def load_model(self, model_name: str):
		model_name = model_name or Args['llm_model']
		models_dir = Args['llm_models_dir']

		# model_name should be name of file or directory in models_dir
		is_dir = os.path.isdir(os.path.join(models_dir, model_name))
		is_file = os.path.isfile(
			os.path.join(models_dir, model_name)
		)
		if not is_dir and not is_file:
			raise Exception(
				f'Model {model_name} not found in {models_dir}.'
			)
		self.model_abspath = os.path.join(models_dir, model_name)
		self.model_name = model_name
		self.cache = LlamaCache()
		self.config = LlamaCppConfig(self.model_abspath)

		if self.loaded or self.model is not None:
			self.unload_model()

		logger.debug(f'Loading model {self.model_name}...')
		start = time.time()
		self.model = Llama(**self.config)
		self.model.set_cache(self.cache)
		end = time.time()
		logger.debug(
			f'Loaded model {self.model_name} in {end - start}s'
		)
		self.loaded = True

	def unload_model(self):
		torch.cuda.empty_cache()
		logger.debug('Unloaded model.')
		self.model = None
		self.model_name = None
		self.model_abspath = None
		self.config = None
		self.loaded = False

	def complete(self, options: CompletionOptions_LlamaCppPython):
		if not self.loaded or self.model is None:
			raise Exception('No model loaded.')
		assert isinstance(options, CompletionOptions_LlamaCppPython)
		start = time.time()
		o = options.model_dump()
		o = {k: v for k, v in o.items() if v is not None}

		result = self.model.create_completion(**o)
		end = time.time()
		logger.debug(f'Generated text in {end - start}s')
		return {
			'result': result,
			'params': o,
		}
