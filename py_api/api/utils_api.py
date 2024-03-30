# TODO
# convert audio to wav
#   pass format (if wav then pick some defaults like 16000 hz, 16 bit, mono)
#   allow options for trimming, etc.
# remove noise from audio (with facebook denoiser)

import logging, os, time
from typing import Any
import magic
import subprocess as sp
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from py_api.args import Args
from py_api.models.utils_api import GetVRAMResponse
from py_api.utils import audio

mime = magic.Magic(mime=True)
logger = logging.getLogger(__name__)

def utils_api(app: FastAPI):
	@app.get('/utils/v1/vram', tags=['utils'])
	async def get_vram() -> GetVRAMResponse:
		"""
		Get VRAM usage.
		"""
		command = "nvidia-smi --query-gpu=gpu_name,memory.free,memory.total --format=csv"
		memory_info = sp.check_output(
			command.split()
		).decode('ascii').split('\n')[:-1][1:]
		memory_values = memory_info[0].split(', ')
		gpu_name = memory_values[0]
		mem_free = int(memory_values[1].split()[0])
		mem_total = int(memory_values[2].split()[0])
		mem_used = mem_total - mem_free

		return GetVRAMResponse(
			gpu_name=gpu_name,
			used=mem_used / 1024,
			total=mem_total / 1024,
			free=mem_free / 1024
		)

	@app.post('/utils/v1/media-to-wav', tags=['utils'])
	async def media_to_wav(
		file: UploadFile = File(...),
		trim_start: float = Query(None),
		trim_end: float = Query(None),
	):
		"""
		Convert media file to wav, for use with STT.
		"""
		# TODO flag for downsampling (for transcription) - call it 'sample_for_stt'?
		if file.filename is None or file.filename == '':
			raise HTTPException(
				status_code=400, detail="No file provided"
			)
		file_path = os.path.join(
			Args['stt_input_dir'], file.filename
		)

		with open(file_path, 'wb') as f:
			f.write(file.file.read())

		file_type = mime.from_file(file_path)
		if 'audio/wav' not in file_type:
			file_path = audio.convert_to_wav(file_path)
		return FileResponse(file_path, media_type='audio/wav')

