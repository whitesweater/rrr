import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
# print(torch.cuda.is_available())
# print(torch.__version__)
# exit(-1)


os.environ["http_proxy"] = "http://127.0.0.1:12637"
os.environ["https_proxy"] = "http://127.0.0.1:12637"

base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "checkpoint-500/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# disable safety checks for avoiding false positives of NSFW
pipe.safety_checker = lambda images, clip_input: (images, None)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()

# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("merge")
prompt = "terrain map, grayscale"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, generator=generator, image=control_image, num_inference_steps=20
).images[0]
image.save("output.png")
