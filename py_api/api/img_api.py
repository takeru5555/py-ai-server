import logging, os, time
from typing import List, Union
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from py_api.args import Args
from py_api.client import img_client_manager
from py_api.models.common_api import GetModelResponse, ListModelsResponse, LoadModelRequest, LoadModelResponse, UnloadModelResponse
from py_api.models.img.img_client import Txt2ImgOptions, Txt2ImgResponse

EXTENSIONS = []
logger = logging.getLogger(__name__)

def img_api(app: FastAPI):
	manager = img_client_manager.ImgManager.instance

	def modelName():
		if manager.model_name is not None:
			return manager.model_name
		else:
			return 'None'

	def get_model() -> GetModelResponse:
		return GetModelResponse.model_validate({
			'model': modelName()
		})

	def load_model(model_name: str) -> LoadModelResponse:
		start = time.time()
		if manager.model_name is not None:
			if manager.model_name == model_name:  # already loaded
				return LoadModelResponse.model_validate({
					'status':
					'Loaded',
					'model':
					modelName(),
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
			'time':
			time.time() - start
		})

	def unload_model() -> UnloadModelResponse:
		start = time.time()
		if manager.model_name is None:
			return UnloadModelResponse.model_validate({
				'status':
				'Unloaded',
				'model':
				modelName(),
				'time':
				time.time() - start
			})
		manager.unload_model()
		return UnloadModelResponse.model_validate({
			'status':
			'Unloaded',
			'model':
			modelName(),
			'time':
			time.time() - start
		})

	def list_models() -> ListModelsResponse:
		models = manager.list_models()
		return ListModelsResponse.model_validate({'models': models})

	def list_samplers() -> List[str]:
		return manager.list_samplers()

	def txt2img(gen_options: Txt2ImgOptions) -> Txt2ImgResponse:
		if manager.model_name is None:
			raise Exception('No model loaded.')
		res = manager.txt2img(gen_options)
		return Txt2ImgResponse.model_validate(res)

	@app.websocket('/img/v1/ws')
	async def img_ws(websocket: WebSocket):
		await websocket.accept()

		async def send_json(data):
			return await websocket.send_json(data)

		await send_json({
			'type': 'list_models',
			'data': list_models().model_dump_json()
		})
		await send_json({
			'type': 'get_model',
			'data': get_model().model_dump_json()
		})

		while True:
			try:
				data = await websocket.receive_json()
			except Exception as e:
				logger.error(e)
				await websocket.close()
				break
			if data['type'] == 'load_model':
				req = data['data']
				try:
					res = load_model(req.model_name).model_dump_json()
				except Exception as e:
					res = {'error': str(e)}
				await send_json({'type': 'load_model', 'data': res})
			elif data['type'] == 'unload_model':
				await send_json({
					'type': 'unload_model',
					'data': unload_model().model_dump_json()
				})
			elif data['type'] == 'get_model':
				await send_json({
					'type': 'get_model',
					'data': get_model().model_dump_json()
				})
			elif data['type'] == 'list_models':
				await send_json({
					'type': 'list_models',
					'data': list_models().model_dump_json()
				})

	@app.get(
		'/img/v1/model',
		response_model=GetModelResponse,
		tags=['img']
	)
	async def img_get_model():
		"""Get currently-loaded model_name"""
		return JSONResponse(content=get_model().model_dump())

	@app.get(
		'/img/v1/list-models',
		response_model=ListModelsResponse,
		tags=['img']
	)
	async def img_list_models():
		"""Get list of models (using relative filenames) in llm_models_dir"""
		return JSONResponse(content=list_models().model_dump())

	@app.post(
		'/img/v1/model/load',
		response_model=LoadModelResponse,
		tags=['img']
	)
	async def img_load_model(body: LoadModelRequest):
		"""Load a model by filename from img_models_dir"""
		model_name = body.model
		return JSONResponse(
			content=load_model(model_name).model_dump()
		)

	@app.get(
		'/img/v1/model/unload',
		response_model=UnloadModelResponse,
		tags=['img']
	)
	async def img_unload_model():
		"""Unload currently-loaded model"""
		return JSONResponse(content=unload_model().model_dump())

	# POST /img/v1/txt2img
	@app.post(
		'/img/v1/txt2img',
		response_model=Txt2ImgResponse,
		tags=['img']
	)
	async def img_txt2img(gen_options: Txt2ImgOptions):
		"""Generate an image from text"""
		return JSONResponse(
			content=txt2img(gen_options).model_dump()
		)

	@app.get('/img/v1/list-samplers', tags=['img'])
	async def img_list_samplers():
		"""Get list of available samplers."""
		return JSONResponse(content={'samplers': list_samplers()})
