import subprocess
import os

def convert_to_wav(input_file_path: str) -> str:
	"""
	Converts an audio file to WAV format using ffmpeg.
	:param input_file_path: Path to the input audio file.
	:return: Path to the converted WAV file.
	"""
	output_file_path = input_file_path + '.wav'

	try:
		subprocess.run([
			'ffmpeg', '-i', input_file_path, '-f', 'wav', '-ar',
			'16000', '-ac', '1', output_file_path, '-y'
		],
										check=True,
										stdout=subprocess.DEVNULL,
										stderr=subprocess.DEVNULL)
		return output_file_path
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Audio conversion failed: {e}")
