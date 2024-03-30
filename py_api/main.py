import torch

torch.cuda.empty_cache()

import argparse, logging, os
import uvicorn, fastapi
from fastapi.middleware.cors import CORSMiddleware

# from py_api.api import img_api, llm_api, stt_api, tts_api, utils_api
from py_api.args import Args
# from py_api.client import llm_client_manager, tts_client_manager, stt_client_manager, img_client_manager
from py_api.settings import HOST, PORT, LLM_MODELS_DIR, LLM_MODEL, TTS_MODELS_DIR, TTS_MODEL, TTS_OUTPUT_DIR, TTS_VOICES_DIR, STT_INPUT_DIR, IMG_MODELS_DIR, IMG_MODEL
from py_api.utils import prompt_format

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Starts the API server.'
	)

	# Group for host and port
	server_group = parser.add_argument_group('server')
	server_group.add_argument(
		'--host',
		type=str,
		default=HOST,
		help='The host to bind to.'
	)
	server_group.add_argument(
		'--port',
		type=int,
		default=PORT,
		help='The port to bind to.'
	)

	# Group for LLM
	llm_group = parser.add_argument_group('llm')
	llm_group.add_argument(
		'--no-llm', action='store_true', help='Disable LLM.'
	)
	llm_group.add_argument(
		'--llm-models-dir',
		#  type=argparse.FileType('r'),
		default=LLM_MODELS_DIR,
		help='The directory to load LLM models from.'
	)
	llm_group.add_argument(
		'--llm-model',
		type=str,
		default=LLM_MODEL,
		help='The LLM model to load.'
	)

	# Group for TTS
	tts_group = parser.add_argument_group('tts')
	tts_group.add_argument(
		'--no-tts', action='store_true', help='Disable TTS.'
	)
	tts_group.add_argument(
		'--tts-models-dir',
		#  type=argparse.FileType('r'),
		default=TTS_MODELS_DIR,
		help='The directory to load TTS models from.'
	)
	tts_group.add_argument(
		'--tts-model',
		type=str,
		default=TTS_MODEL,
		help='The TTS model to load.'
	)
	tts_group.add_argument(
		'--tts-output-dir',
		#  type=argparse.FileType('r'),
		default=TTS_OUTPUT_DIR,
		help='The directory to save TTS output to.'
	)
	tts_group.add_argument(
		'--tts-voices-dir',
		#  type=argparse.FileType('r'),
		default=TTS_VOICES_DIR,
		help='The directory to load TTS voices from.'
	)

	# Group for STT
	stt_group = parser.add_argument_group('stt')
	stt_group.add_argument(
		'--no-stt', action='store_true', help='Disable STT.'
	)
	stt_group.add_argument(
		'--stt-input-dir',
		#  type=argparse.FileType('r'),
		default=STT_INPUT_DIR,
		help='The directory to load STT input from.'
	)

	# Group for Img
	img_group = parser.add_argument_group('img')
	img_group.add_argument(
		'--no-img', action='store_true', help='Disable Img AI.'
	)
	img_group.add_argument(
		'--img-models-dir',
		#  type=argparse.FileType('r'),
		default=IMG_MODELS_DIR,
		help='The directory to load Img AI models from.'
	)
	img_group.add_argument(
		'--img-model',
		type=str,
		default=IMG_MODEL,
		help='The Img AI model to load.'
	)

	# Log level
	parser.add_argument(
		'--log-level',
		type=str,
		default='INFO',
		choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
		help='The log level to use.'
	)

	args = parser.parse_args()
	Args['host'] = args.host
	Args['port'] = args.port
	Args['llm_models_dir'] = args.llm_models_dir
	Args['llm_model'] = args.llm_model
	Args['tts_models_dir'] = args.tts_models_dir
	Args['tts_model'] = args.tts_model
	Args['tts_output_dir'] = args.tts_output_dir
	Args['tts_voices_dir'] = args.tts_voices_dir
	Args['stt_input_dir'] = args.stt_input_dir
	Args['img_models_dir'] = args.img_models_dir
	Args['img_model'] = args.img_model

	logging.basicConfig(level=args.log_level)
	logger = logging.getLogger(__name__)

	totalmem = torch.cuda.get_device_properties(0).total_memory
	totalmem /= 1024**3
	usedmem = torch.cuda.memory_allocated(0)
	usedmem /= 1024**3
	freemem = totalmem - usedmem
	logger.debug(f'Total GPU memory: {totalmem:.2f} GB')
	if freemem < 1:
		raise RuntimeError('Not enough GPU memory to run LLM.')

	llm_model = Args['llm_model']
	tts_model = Args['tts_model']

	app = fastapi.FastAPI()
	app.add_middleware(
		CORSMiddleware,
		allow_origins=[
			"http://localhost:8085",
		],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	if not args.no_llm:
		from py_api.api.llm_api import llm_api
		from py_api.client.llm_client_manager import LLMManager
		llmManager = LLMManager.instance
		if llm_model is not None:
			fmt = prompt_format.get_model_format(llm_model)
			logger.info(f'Loading LLM model: {llm_model}')
			logger.info(f'Detected model prompt format: {fmt}')
		llmManager.load_model(llm_model)
		llm_api(app)

	if not args.no_tts:
		from py_api.api.tts_api import tts_api
		from py_api.client.tts_client_manager import TTSManager
		ttsManager = TTSManager.instance
		ttsManager.load_model(tts_model)
		tts_api(app)

	if not args.no_stt:
		from py_api.api.stt_api import stt_api
		from py_api.client.stt_client_manager import STTManager
		sttManager = STTManager.instance
		try:
			sttManager.load_model('whisperx')
		except Exception:
			sttManager.unload_model()
			logger.warning('Failed to load STT model. Out of memory?')
		stt_api(app)

	if not args.no_img:
		from py_api.api.img_api import img_api
		from py_api.client.img_client_manager import ImgManager
		imgManager = ImgManager.instance
		imgManager.load_model('')
		img_api(app)

	from py_api.api.utils_api import utils_api
	utils_api(app)

	uvicorn.run(app, host=args.host, port=args.port)
	logger.info(f'API server started on {args.host}:{args.port}.')
