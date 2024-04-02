#!/bin/bash

# Default values
MODEL="/media/user/ML/LLaMA/openchat-3.5-1210.Q5_K_M.gguf"
GPU_LAYERS=35
CACHE=true

# Parse command line arguments
while (("$#")); do
	case "$1" in
	--model)
		MODEL=$2
		shift 2
		;;
	--n_gpu_layers)
		GPU_LAYERS=$2
		shift 2
		;;
	--cache)
		CACHE=$2
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

# Run the command
python3 -m llama_cpp.server --model $MODEL --n_gpu_layers $GPU_LAYERS --cache $CACHE
