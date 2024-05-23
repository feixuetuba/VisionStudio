import logging
import os.path
import re

import diffusers
import torch


import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    HeunDiscreteScheduler, LMSDiscreteScheduler


def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character = ",",
    device = torch.device("cpu")
):
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        print("prompt", prompt)
        print("negative_prompt", negative_prompt)
        input_ids = pipe.tokenizer(
            prompt, return_tensors = "pt", truncation = False,
            padding = "max_length",
        ).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length,
            return_tensors = "pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        print("negative_prompt", negative_prompt)
        print("prompt", prompt)
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors = "pt", truncation = False,
            padding = "max_length",
        ).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

def load_pipeline(model_path:str,  safety_check=True, sd_based="1.5", varient="fp16", original_config_file=None):
    """
    加载sd模型，
    :param model_path: 模型路径
    :param sd_based: sd_based基模，不同基模对应不同的加载器
    :return:
    """
    if varient == "fp16":
        torch_type = torch.float16,
    else:
        torch_type = torch.float32

    if sd_based == "xdl":
        diffusion = diffusers.StableDiffusionXLPipeline
    else:
        diffusion = diffusers.StableDiffusionPipeline
    args = {
        "torch_type": torch_type,
        "varient": varient
    }
    if not safety_check:
        args["safety_checker"] = None
    if os.path.isfile(model_path):
        pipe = diffusion.from_single_file(
            model_path,
            use_safetensor = ".safetensor" in model_path,
            original_config_file = original_config_file,
            **args
            )
    else:
        pipe = diffusion.from_pretrained(model_path,**args )
    # pipe.unet = torch.compile(pipe.unet, model="reduce-overhead",fullgraph=True,original_config_file=None)
    pipe.enable_model_cpu_offload()
    return pipe


def set_scheduler(stage, scheduler_name):
    #ref https://stablediffusionapi.com/docs/a1111schedulers/
    name = scheduler_name.lower()
    if "2s a" in name:
        logging.warning("Use 2M to replace '2S a'")
        name = name.replace("2s a", "2m")
    if name == 'ddpm':
        scheduler = DDPMScheduler.from_config(stage.scheduler.config)
    elif "dpm++ 2m" in name:
        scheduler = DPMSolverMultistepScheduler.from_config(stage.scheduler.config) #DPM++ 2M
        if "sde" in name:
            scheduler = DPMSolverMultistepScheduler.from_config(stage.scheduler.config)
            scheduler.algorithm_type="sde-dpmsolver+++"
        if 'karras' in name:
            scheduler.use_karras_sigmas="yes"
    elif "dpm++" in name:
        scheduler = DPMSolverSinglestepScheduler.from_config(stage.scheduler.config)
        if "karras" in name:
            scheduler.use_karras_sigmas = "yes"
        if "sde" in name:
            scheduler.config.algorithm_type = 'sde-dpmsolver++'
        else:
            scheduler.config.algorithm_type = 'dpmsolver++'
    elif "dpm2" in name:
        if "dpm2 a" in name:
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(stage.scheduler.config)
        else:
            scheduler = KDPM2DiscreteScheduler.from_config(stage.scheduler.config)
        if "karras" in name:
            scheduler.use_karras_sigmas = "yes"
    elif name == "euler":
        scheduler = EulerDiscreteScheduler.from_config(stage.scheduler.config)
    elif name == "euler a":
        scheduler = EulerAncestralDiscreteScheduler.from_config(stage.scheduler.config)
    elif name == "heun":
        scheduler = HeunDiscreteScheduler.from_config(stage.scheduler.config)
    elif name == "lms":
        scheduler = LMSDiscreteScheduler.from_config(stage.scheduler.config)
        if "karras" in name:
            scheduler.use_karras_sigmas = "yes"
    stage.scheduler = scheduler
    return stage

def load_lora(pipeline, lora_path, lora_w):
    state_dict, network_alphas = pipeline.lora_state_dict(
        lora_path
    )

    for key in network_alphas:
        network_alphas[key] = network_alphas[key] * lora_w

    pipeline.load_lora_into_unet(
        state_dict=state_dict
        , network_alphas=network_alphas
        , unet=pipeline.unet
    )

    pipeline.load_lora_into_text_encoder(
        state_dict=state_dict
        , network_alphas=network_alphas
        , text_encoder=pipeline.text_encoder
    )
    return pipeline


def _size(value=str):
    return [int(_) for _ in value.lower().strip("x")]

class Param(dict):
    def __init__(self, params:str, **kwargs):
        for p in params.split(","):
            name, value = p.strip().split(":")
            self.__setitem__(name.strip(), value.strip())

    def get(self, key, value, dtype=None):
        if key in self:
            v = self.__getitem__(key)
            if dtype is not None:
                return dtype(v)
        else:
            return value

def run_from_generate_data(pipeline, generate_data, device_name="cuda", torch_dtype=torch.float32):

    if isinstance(generate_data, str):
        keys = ["prompt", "negative", "params"]
        data = {}
        i = 0
        for line in generate_data.split("\n"):
            line = line.strip()
            if len(line) > 0:
                data[keys[i]] = line
                i+=1
        generate_data = data
    pattern = re.compile("<lora:([\w|_]*:\d+\.*\d*)>")
    prompt = generate_data.get("prompt","").replace("Prompt: ", "")
    loras = []
    for lora in pattern.findall(prompt):
        name, alpha = lora.split(":")
        alpha = float(alpha)
        loras.append((name, alpha))
    prompt = pattern.sub("", prompt).strip()

    neg_prompt = generate_data.get("negative","").replace("Negative prompt: ", "")
    params = Param(generate_data["params"])

    steps = params.get("step", 30, int)
    size = params.get("size", (512,512), _size)
    seed = params.get("seed", -1, int)
    model_id = params.get("model","models/stablediffussion/runwayml/stable-diffusion-v1-5", str)
    smapler = params.get("Sampler","DDPM 2M", str)
    CFG_SCALE = params.get("CFG scale", 7, int)
    clip_skip = params.get("Clip skip", 9, int)
    hires_steps = 0
    hires_upscale = 1.0
    denoising_strength = 0
    original_config_file = None

    if clip_skip > 1:
        clip_layers = pipeline.text_encoder.text_model.encoder.layers
        if clip_skip > 0:
            pipeline.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    scheduler = set_scheduler(pipeline, smapler)
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(pipeline, prompt, neg_prompt, device=torch.device(device_name))

    if False: #hires_upscale > 1.0:
        latent_images = pipeline(prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=size[0], height=size[1], scheduler=scheduler,
            generator=generator, num_inference_steps=steps,
            guidance_scale=CFG_SCALE,
            clip_skip=clip_skip,
            strength=denoising_strength,
           output_type='latent'
        )
        upScale = StableDiffusionLatentUpscalePipeline(**pipeline.components)
        latent= upScale(
            prompt=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=latent_images,
            strength=denoising_strength,
            num_inference_steps=hires_steps,
            guidance_scale=CFG_SCALE,
            num_images_per_prompt=1,
            generator=generator
        )

        img2img = StableDiffusionImg2ImgPipeline(**pipeline.components)
        images = img2img(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=latent,
            strength=denoising_strength,
            num_inference_steps=hires_steps,
            guidance_scale=CFG_SCALE,
            num_images_per_prompt=1,
            generator=generator
        ).images
    else:
        images = pipeline(prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            width=size[0], height=size[1], scheduler=scheduler,
            generator=generator, num_inference_steps=steps,
            guidance_scale=CFG_SCALE,
            clip_skip=clip_skip,
            strength=denoising_strength, use_safetensors=True
        ).images
    return images


if __name__ == "__main__":
    generate_data = """
    overweight, (chubby:1.25), sweaty, woman sitting on a couch, surrounded by empty bags of chips and candy wrappers. She is wearing a stained t-shirt and sweatpants, and she looks content and relaxed, (best quality), (masterpiece:1.2), 4k ,(ultra detailed:1.2)
    Negative prompt: (low quality:1.2), (worst quality:1.2), (bad anatomy), (deformed), disfigured, long neck, bad hands, poorly drawn face, watermark, text, poorly drawn hands, username, EasyNegative, bad-hands-5, bad_prompt_version2, lowres
    Steps: 20, CFG scale: 7, Sampler: DPM++ 2M Karras, Seed: 4039756984, Clip skip: 2
    """
    # pipe = load_pipeline(r"E:\virtualmachine\shared\stablediffusion\majicmixRealistic-v7.safetensors",
    pipe = load_pipeline(r"D:\codes\stable_diffussion_refernece\my_diffussion\models\stablediffussion\chilloutmix_Ni\chilloutmix_NiPrunedFp32Fix.safetensors",
                         safety_check=False,
                         varient="fp32")
    images = run_from_generate_data(pipe, generate_data,)
    images[0].save("./result.jpg")