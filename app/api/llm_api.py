import logging, os, time
from fastapi import FastAPI, HTTPException, Path, WebSocket
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from py_api.args import Args
from py_api.client import llm_client_manager
from py_api.client.llm import LLMClient_OpenAI
from py_api.models.llm.llm_api import CompletionRequest, CompletionResponse, DownloadModelRequest, DownloadModelResponse, ListModelsResponse, GetModelResponse, LoadModelResponse, UnloadModelRequest, LoadModelRequest
from py_api.models.llm.client import CompletionOptions, MessageObject
from py_api.utils import prompt_format

# supported extensions
EXTENSIONS = ['.gguf', '.ggml', '.safetensor']
logger = logging.getLogger(__name__)

def llm_api(app: FastAPI):
	manager = llm_client_manager.LLMManager.instance
	openai = LLMClient_OpenAI.instance

	def modelName():
		if manager.model_name is not None:
			return manager.model_name
		else:
			return 'None'

	def get_model() -> GetModelResponse:
		return GetModelResponse.model_validate({
			'model':
			modelName(),
			'loader_name':
			manager.loader_name
		})

	def load_model(model_name: str) -> LoadModelResponse:
		# TODO: instead of model name, accept 'features' dict:
		#   grammar (bool): model supports grammar
		#   model_type ('instruct' or 'chat'): the preferred model type
		#   cfg (bool): model supports cfg sampler
		#   ctx (int)? (maybe not): preferred n_ctx, probably not trivial (e.g. a bigger model at same ctx might not fit in vram/etc)
		start = time.time()
		if manager.model_name is not None:
			if manager.model_name == model_name:  # already loaded
				return LoadModelResponse.model_validate({
					'status':
					'Loaded',
					'model':
					modelName(),
					'loader_name':
					manager.loader_name,
					'time':
					time.time() - start
				})
			manager.unload_model()
		logger.debug(model_name)
		try:
			manager.load_model(model_name)
		except Exception as e:
			return LoadModelResponse.model_validate({
				'status':
				'Error',
				'model':
				modelName(),
				'time':
				time.time() - start,
				'error':
				str(e)
			})
		return LoadModelResponse.model_validate({
			'status':
			'Loaded',
			'model':
			modelName(),
			'loader_name':
			manager.loader_name,
			'time':
			time.time() - start
		})

	def unload_model() -> UnloadModelRequest:
		start = time.time()
		if manager.model_name is None:
			return UnloadModelRequest.model_validate({
				'status':
				'Unloaded',
				'model':
				modelName(),
				'time':
				time.time() - start
			})
		manager.unload_model()
		return UnloadModelRequest.model_validate({
			'status':
			'Unloaded',
			'model':
			modelName(),
			'time':
			time.time() - start
		})

	def list_models() -> ListModelsResponse:
		return ListModelsResponse.model_validate({
			'models':
			manager.list_models()
		})

	def download_model(model_name: str) -> DownloadModelResponse:
		# TODO support links to e.g. *.gguf files (download them to llm_models_dir)
		model = None
		branch = 'main'
		is_right_format = False
		# check that model name is format like 'username/model_name[:branch]'
		if '/' in model_name:
			split = model_name.split('/')
			model = split[1]
			if ':' in model:
				split = model.split(':')
				model = split[0]
				branch = split[1]
			is_right_format = True
		if not is_right_format:
			raise Exception(
				'model_name must be in format like "username/model_name[:branch]"'
			)
		dir_name = model_name.replace('/', '_')
		dir_name = dir_name.replace(':', '--')
		start = time.time()
		try:
			snapshot_download(
				repo_id=model_name,
				revision=branch,
				cache_dir=Args['llm_models_dir']
			)
		except Exception as e:
			return DownloadModelResponse.model_validate({
				'status':
				'Error',
				'model':
				dir_name,
				'time':
				time.time() - start,
				'error':
				str(e)
			})
		return DownloadModelResponse.model_validate({
			'status':
			'Downloaded',
			'model':
			dir_name,
			'time':
			time.time() - start
		})

	def complete(req: CompletionRequest):
		# TODO accept a "json_format" which can take some kind of specfor json
		#   we'll use whatever json-constraints the current loader exposes if any
		#   llamacpp has grammar (& LlamaGrammar has a from_json method)
		#   thought exllama has something to constrain to json (but not general grammar)
		#   transformers probably has something
		# this option takes precedence over grammar and will override it
		prompt = req.prompt
		parts = req.parts
		messages = req.messages
		prefix_response = req.prefix_response
		model = req.model
		if model == '':
			model = manager.model_name or ''

		# TODO create default stop strings for models/formats

		# TODO wrapper for hanndling prompt/parts/messages
		if prompt is None and parts is None and len(messages) == 0:
			raise Exception('Prompt or parts or messages is required.')
		if parts is not None:
			try:
				if 'gpt-' in model:
					m = prompt_format.parts_to_messages(
						parts, prefix_response=prefix_response
					)
					# reconstruct models from message objects
					messages = []
					for msg in m:
						messages.append(MessageObject.model_validate(msg))
				else:
					prompt = prompt_format.parts_to_prompt(
						parts, model, prefix_response
					)
			except Exception as e:
				# raise Exception(
				#     'Internal server error: Unknown model when detecting format: ' +
				#     modelName())
				raise e
		else:
			prompt = str(prompt)
		req.prompt = prompt
		req.messages = messages
		options = CompletionOptions.model_validate(req.model_dump())
		result = manager.complete(options)
		assert result is not None
		res: dict = {
			'result': result.result,
			'params': result.params
		}
		return CompletionResponse(**res)

	@app.websocket('/llm/v1/ws')
	async def llm_ws(websocket: WebSocket):
		await websocket.accept()

		await websocket.send_json({
			'type':
			'list_models',
			'data':
			list_models().model_dump_json()
		})
		await websocket.send_json({
			'type':
			'get_model',
			'data':
			get_model().model_dump_json()
		})
		while True:
			try:
				data = await websocket.receive_json()
			except Exception as e:
				logger.error(e)
				await websocket.close()
				break
			if data['type'] == 'complete':
				req = CompletionRequest(**data['data'])
				try:
					res = complete(req).model_dump_json()
				except Exception as e:
					res = {'error': str(e)}
				await websocket.send_json(res)
			elif data['type'] == 'load_model':
				req = data['data']
				try:
					res = load_model(req.model_name).model_dump_json()
				except Exception as e:
					res = {'error': str(e)}
				await websocket.send_json({
					'type': 'load_model',
					'data': res
				})
			elif data['type'] == 'unload_model':
				await websocket.send_json({
					'type':
					'unload_model',
					'data':
					unload_model().model_dump_json()
				})
			elif data['type'] == 'get_model':
				await websocket.send_json({
					'type':
					'get_model',
					'data':
					get_model().model_dump_json()
				})
			elif data['type'] == 'list_models':
				await websocket.send_json({
					'type':
					'list_models',
					'data':
					list_models().model_dump_json()
				})
			elif data['type'] == 'download_model':
				req = DownloadModelRequest(**data['data'])
				try:
					res = download_model(req.model).model_dump_json()
				except Exception as e:
					res = {'error': str(e)}
				await websocket.send_json({
					'type': 'download_model',
					'data': res
				})

	@app.post(
		'/llm/v1/complete',
		response_model=CompletionResponse,
		tags=['llm']
	)
	async def llm_complete(req: CompletionRequest):
		"""Generate text from a prompt, or an array of PromptParts. Prompt should be in proper format (unless using `parts`), it's fed directly to the model. If both are provided then `prompt` is overwritten by constructing prompt from `parts`."""
		if manager.model_name is None and req.model is None:
			raise HTTPException(
				status_code=500, detail='Model not loaded.'
			)
		# TODO add 'messages' prop for openai (converted to prompt if not using openai)
		if req.prompt is None and req.parts is None:
			raise HTTPException(
				status_code=400, detail='Prompt or parts is required.'
			)
		return complete(req)

	@app.get(
		'/llm/v1/model',
		response_model=GetModelResponse,
		tags=['llm']
	)
	async def llm_get_model():
		"""Get currently-loaded model"""
		return JSONResponse(content=get_model().model_dump())

	@app.get(
		'/llm/v1/list-models',
		response_model=ListModelsResponse,
		tags=['llm']
	)
	async def llm_list_models():
		"""Get list of models (using relative filenames) in llm_models_dir"""
		return JSONResponse(content=list_models().model_dump())

	@app.post(
		'/llm/v1/model/load',
		response_model=LoadModelResponse,
		tags=['llm']
	)
	async def llm_load_model(body: LoadModelRequest):
		"""Load a model by filename from llm_models_dir"""
		model_name = body.model
		return JSONResponse(
			content=load_model(model_name).model_dump()
		)

	@app.get(
		'/llm/v1/model/unload',
		response_model=UnloadModelRequest,
		tags=['llm']
	)
	async def llm_unload_model():
		"""Unload currently-loaded model"""
		return JSONResponse(content=unload_model().model_dump())

	@app.post(
		'/llm/v1/download-model',
		response_model=DownloadModelResponse,
		tags=['llm']
	)
	async def llm_download_model(body: DownloadModelRequest):
		"""Download a model from the HuggingFace Hub"""
		return JSONResponse(
			content=download_model(body.model).model_dump()
		)
