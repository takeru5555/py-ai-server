from py_api.client.img.base import ImgClient_Base
from py_api.models.img.img_client import Txt2ImgOptions, Txt2ImgResponse
from PIL import Image
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import base64, time
from io import BytesIO
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

import torch

from typing import Union, List, Any, Dict
from py_api.args import Args
import os, json

SamplersUnion = Union[DPMSolverMultistepScheduler,
											EulerAncestralDiscreteScheduler]
SamplersDict = Dict[str, Union[SamplersUnion, None]]

samplers: SamplersDict = {
	'dpm++ 2m': None,
	'dpm++ 2m karras': None,
	'euler a': None
}

class ImgClient_Diffusers(ImgClient_Base):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model: Union[StableDiffusionPipeline,
								StableDiffusionXLPipeline, None] = None
	pipeline: Union[DiffusionPipeline, None] = None
	default_config: Any = {}
	current_sampler: str = ''

	def load_model(self, model_name: str):
		if model_name is None or model_name == '':
			model_name = Args['img_model']
		if model_name == self.model_name:
			return
		self.model_name = model_name
		self.abspath = os.path.join(
			Args['img_models_dir'], model_name
		)
		if 'xl' in model_name:
			self.pipeline = StableDiffusionXLPipeline.from_single_file(
				self.abspath,
				variant="fp16",
				load_safety_checker=False,
				torch_dtype=torch.float16
			)
			print('Using SDXL pipeline')
		else:
			self.pipeline = StableDiffusionPipeline.from_single_file(
				self.abspath,
				variant="fp16",
				load_safety_checker=False,
				torch_dtype=torch.float16,
				# scheduler_type='dpm'
			)
		self.pipeline.to(self.device)
		# self.pipeline.unet.set_default_attn_processor()

		# use vae from model
		vae = AutoencoderKL.from_single_file(
			self.abspath, variant="fp16", torch_dtype=torch.float16
		).to(self.device)
		self.pipeline.vae = vae

		self.default_config = self.pipeline.scheduler.config
		samplers['dpm++ 2m'] = DPMSolverMultistepScheduler.from_config( # type: ignore
			self.default_config
		)
		self.pipeline.scheduler = samplers['dpm++ 2m']
		self.current_sampler = 'DPM++ 2M'

		self.model = AutoPipelineForText2Image.from_pipe(
			self.pipeline
		)
		self.loaded = True

	def unload_model(self):
		self.model = None
		self.model_name = None
		self.model_abspath = None
		self.loaded = False
		torch.cuda.empty_cache()

	def list_samplers(self) -> List[str]:
		return ['DPM++ 2M', 'DPM++ 2M Karras', 'Euler a']

	def get_sampler(self, sampler_name: str) -> Any:
		sampler_name = sampler_name.lower()
		if samplers[sampler_name] is None:
			sampler = None
			if sampler_name == 'dpm++ 2m':
				sampler = DPMSolverMultistepScheduler.from_config(
					self.default_config
				)
			elif sampler_name == 'dpm++ 2m karras':
				sampler = DPMSolverMultistepScheduler.from_config(
					self.default_config, use_karras_sigmas=True
				)
			elif sampler_name == 'euler a':
				sampler = EulerAncestralDiscreteScheduler.from_config(
					self.default_config
				)
			samplers[sampler_name] = sampler  # type: ignore
		return samplers[sampler_name]

	def txt2img(
		self, gen_options: Txt2ImgOptions
	) -> Txt2ImgResponse:
		if not self.loaded:
			self.load_model('')
			if not self.loaded:
				raise Exception('Model not loaded.')
		if gen_options is None or gen_options.prompt is None:
			raise Exception('No prompt provided.')
		if gen_options.prompt == '':
			raise Exception('No prompt provided.')
		assert self.model is not None
		assert self.pipeline is not None
		prompt = gen_options.prompt
		prompt = prompt.replace('\n', ' ')
		prompt = prompt.replace('\r', ' ')
		prompt = prompt.replace('\t', ' ')
		prompt = prompt.replace('  ', ' ')
		prompt = prompt.strip()

		sampler_name = gen_options.sampler_name
		if sampler_name is None or sampler_name == '':
			sampler_name = 'Euler a'
		if sampler_name != self.current_sampler:
			sampler = self.get_sampler(sampler_name)
			self.pipeline.scheduler = sampler
			self.model = AutoPipelineForText2Image.from_pipe(
				self.pipeline
			)
			self.current_sampler = sampler_name

		seed = -1
		if gen_options.seed is not None:
			seed = gen_options.seed

		generator = torch.Generator(device=self.device)
		if seed >= 0:
			generator = generator.manual_seed(seed)
		else:
			ran = int(torch.randint(0, 2**32, (1, )).item())
			generator = generator.manual_seed((ran))
			seed = ran

		# TODO option to enable freeu
		self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

		opt = gen_options.model_dump()
		res = self.model(generator=generator, **opt)

		self.pipeline.disable_freeu()
		img = res.images[0]  # type: ignore
		assert isinstance(img, Image.Image)
		imgb64 = 'data:image/png;base64,'
		with BytesIO() as buffer:
			img.save(buffer, 'png')
			imgb64 += base64.b64encode(buffer.getvalue()).decode()

		torch.cuda.empty_cache()

		options = {
			'prompt': prompt,
			'negative_prompt': gen_options.negative_prompt,
			'num_inference_steps': gen_options.num_inference_steps,
			'guidance_scale': gen_options.guidance_scale,
			'width': gen_options.width,
			'height': gen_options.height,
			'clip_skip': gen_options.clip_skip,
			'seed': seed,
			'sampler_name': sampler_name
		}
		return Txt2ImgResponse.model_validate({
			'images': [imgb64],
			'nsfw_content_detected': [False],
			'info':
			Txt2ImgOptions.model_validate(options)
		})

	# def img2img(
	# 	self, gen_options: Img2ImgOptions
	# ) -> Img2ImgResponse:
	# 	raise NotImplementedError()
