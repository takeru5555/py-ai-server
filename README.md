# py-ai-server

## Overview

This is an api server to support my [`ai-experiments`](https://github.com/parsehex/ai-experiments) collection of demos/prototypes.

Instead of finding and using a growing number of server projects to serve separate apis, I made an api server that does what I need.

This was/is mostly a learning exercise to introduce myself to lower-level inference on AI models.

The rest of this README and likely other places is out of date. I had the project working but I haven't used it for at least a few months so this is currently unmaintained.

## Setup

`../setup-api.sh`

## Structure Info

For each of LLM, TTS or STT (or any other AI model types in the future), the following structure is used to (hopefully) manage the variety of the AI ecosystem:

- `api/` (e.g. `llm_api`) - The API routes for that AI. The endpoints use the `manager` for that AI to do the work.
- `client/[AI]_client_manager.py` - The manager for that AI. The manager exposes methods to power the API and handles which client to use.
- `client/[AI]/` - The clients for that AI. There's a `base.py` class that all clients inherit from, and then a client for each client. The client has the specific code for working with the model.
- `models/` - contains the pydantic models that the API routes and clients use.

As of now, the LLM API is the only one with more than 1 client for the manager to use. TODO clients:

- [ ] TTS/XTTS - xtts actually supports other models too (yourtts is one that does voice cloning)
- [ ] TTS/Silero - Is a fast TTS that can be used for testing or when low on VRAM.
- [ ] STT/OpenAI - Supposed to be no difference but you wouldn't have to run the model, uses the API obviously

The manager handles converting options to the correct format for the client. Currently this only happens with LLM.

### Models, Clients/Loaders, Sources

This is an attempt by me to treat different AI types/projects in a similar way

Every type of AI has a single Manager, as well as 1 or more Clients (or Loader). Through the manager, a model is picked and Loaded, and the Model is what the appropriate Client uses to actually run inference (the "generate" in Generative AI).

Models are a bit of a mix. Different clients may support models that other clients also support. To solve this, there are Sources, which are simply different sources for models for that AI.

Example: GGUF models can be used by either LlamaCppPython or Transformers.

Further, certain AI like TTS might have additional assets (like Voices) which can be used across clients or models (for TTS voices, they'd be supported by any voice cloning model).

## Ideas

- `--hot-[llm|tts|...]` - hot-load the model (using websockets?)

## Resources

### LLM

- Exllamav2: https://github.com/turboderp/exllamav2
- LlamaCppPython | Docs: https://llama-cpp-python.readthedocs.io/
- Transformers | Docs: https://huggingface.co/docs/transformers/index

### SD (rename? maybe /img)

- Diffusers: https://huggingface.co/docs/diffusers/index
  - Tutorial to use SD with Diffusers: https://thepythoncode.com/article/generate-images-from-text-stable-diffusion-python#stable-diffusion-pipeline
  - Loading LoRAs: https://huggingface.co/docs/diffusers/using-diffusers/other-formats or https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters#lora

### TTS

- XTTS/Coqui TTS: https://github.com/coqui-ai/TTS | Docs: https://docs.coqui.ai/en/dev/index.html
- StyleTTS2: https://github.com/yl4579/StyleTTS2
  - Python pkg: https://github.com/sidharthrajaram/StyleTTS2

### STT

- WhisperCppPython | pypi: https://pypi.org/project/whisper-cpp-python/
  - (Can't find a github)
  - **TODO**: Use this instead of what we're doing now
