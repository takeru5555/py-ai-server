import logging, os, time
from fastapi import FastAPI, HTTPException, Path, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from py_api.args import Args
from py_api.client import tts_client_manager
from py_api.models.common_api import GetModelResponse, ListModelsResponse, LoadModelResponse, UnloadModelResponse
from py_api.models.tts.tts_api import SpeakRequest, SpeakToFileRequest, ListVoicesResponse
from py_api.models.tts.tts_client import SpeakResponse, SpeakToFileResponse

EXTENSIONS = []
logger = logging.getLogger(__name__)

def tts_api(app: FastAPI):
	manager = tts_client_manager.TTSManager.instance

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
		return ListModelsResponse.model_validate({
			'models':
			manager.list_models()
		})

	def speak(req: SpeakRequest) -> SpeakResponse:
		try:
			res = manager.speak(req)
		except Exception as e:
			res = {'error': str(e)}
		return SpeakResponse.model_validate(res)

	def speak_to_file(
		req: SpeakToFileRequest
	) -> SpeakToFileResponse:
		try:
			res = manager.speak_to_file(req)
		except Exception as e:
			res = {'error': str(e)}
		return SpeakToFileResponse.model_validate(res)

	def list_voices() -> ListVoicesResponse:
		voices = []
		voices_dir = Args['tts_voices_dir']
		# list only *.wav in voices_dir/
		for file in os.listdir(voices_dir):
			if file.endswith('.wav'):
				voices.append(file)
		return ListVoicesResponse.model_validate({'voices': voices})

	@app.websocket('/tts/v1/ws')
	async def tts_ws(websocket: WebSocket):
		await websocket.accept()

		await websocket.send_json({
			'type': 'list_models',
			'data': list_models().model_dump()
		})
		await websocket.send_json({
			'type': 'get_model',
			'data': get_model().model_dump()
		})

		async def send_json(data):
			return websocket.send_json(data)

		while True:
			try:
				data = await websocket.receive_json()
			except Exception as e:
				logger.error(e)
				await websocket.close()
				break
			if data['type'] == 'speak':
				req = SpeakRequest.model_validate(data['data'])
				res = speak(req).model_dump()
				await send_json({'type': 'speak', 'data': res})
			elif data['type'] == 'speak_to_file':
				req = SpeakToFileRequest.model_validate(data['data'])
				res = speak_to_file(req).model_dump()
				await send_json({'type': 'speak_to_file', 'data': res})
			elif data['type'] == 'load_model':
				req = data['data']
				try:
					res = load_model(req.model_name).model_dump()
				except Exception as e:
					res = {'error': str(e)}
				await send_json({'type': 'load_model', 'data': res})
			elif data['type'] == 'unload_model':
				await send_json({
					'type': 'unload_model',
					'data': unload_model().model_dump()
				})
			elif data['type'] == 'get_model':
				await send_json({
					'type': 'get_model',
					'data': get_model().model_dump()
				})
			elif data['type'] == 'list_models':
				await send_json({
					'type': 'list_models',
					'data': list_models().model_dump()
				})
			elif data['type'] == 'list_voices':
				await send_json({
					'type': 'list_voices',
					'data': list_voices().model_dump()
				})
			elif data['type'] == 'play':
				# TODO
				pass

	@app.post(
		'/tts/v1/speak', response_model=SpeakResponse, tags=['tts']
	)
	async def tts_speak(req: SpeakRequest):
		"""Generate TTS audio from text."""
		if manager.model_name is None:
			raise HTTPException(
				status_code=500, detail='Model not loaded.'
			)
		return speak(req).model_dump()

	@app.post(
		'/tts/v1/speak-to-file',
		response_model=SpeakToFileResponse,
		tags=['tts']
	)
	async def tts_speak_to_file(req: SpeakToFileRequest):
		"""Generate and save TTS audio to file on server."""
		if manager.model_name is None:
			raise HTTPException(
				status_code=500, detail='Model not loaded.'
			)
		return speak_to_file(req).model_dump()

	@app.get(
		'/tts/v1/model',
		response_model=GetModelResponse,
		tags=['tts']
	)
	async def tts_get_model():
		"""Get currently-loaded model_name"""
		return JSONResponse(content=get_model().model_dump())

	@app.get(
		'/tts/v1/list-models',
		response_model=ListModelsResponse,
		tags=['tts']
	)
	async def tts_list_models():
		"""Get list of models (using relative filenames) in tts_models_dir"""
		return JSONResponse(content=list_models().model_dump())

	@app.get(
		'/tts/v1/list-voices',
		response_model=ListVoicesResponse,
		tags=['tts']
	)
	async def tts_list_voices():
		"""Get list of voices (using relative filenames) in tts_voices_dir"""
		return JSONResponse(content=list_voices().model_dump())

	@app.get(
		'/tts/v1/model/load',
		response_model=LoadModelResponse,
		tags=['tts']
	)
	async def tts_load_model(model_name: str):
		"""Load a model by filename from tts_models_dir"""
		return JSONResponse(
			content=load_model(model_name).model_dump()
		)

	@app.get(
		'/tts/v1/model/unload',
		response_model=UnloadModelResponse,
		tags=['tts']
	)
	async def tts_unload_model():
		"""Unload currently-loaded model"""
		return JSONResponse(content=unload_model().model_dump())

	@app.get('/tts/v1/play', tags=['tts'])
	async def tts_play(file: str):
		"""Play a file from tts_output_dir"""
		# read and return audio
		return FileResponse(
			os.path.join(Args['tts_output_dir'], file)
		)
