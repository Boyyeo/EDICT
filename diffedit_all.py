import PIL
import requests
import torch
from io import BytesIO
from PIL import Image 
import random 
import numpy as np
import os 
from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler,DDIMInverseScheduler
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
   


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


save_path = 'DIFFEDIT_RESULT'
os.makedirs(save_path, exist_ok=True)

pipe = StableDiffusionDiffEditPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipe = pipe.to("cuda:6")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)



folder_name = 'ORIGINAL_IMAGES/'
image_path_list = ['fruitbowl_scaled.jpg','owl.jpg','car.jpg','bird.jpg','donut.jpg','dog2.png','horse_scaled.jpg','plate_steak.jpeg']
mask_prompt_list = ['Bowl of fruit','An owl','A car','A bird','A donut','a photo of a dog','A photo of a horse','A steak on plate']
prompt_list = ['Bowl of Grapes','A bald eagle','A police car','A origami bird','A sandwich','A photo of a leopard','a photo of a donkey','a car toy on plate']
guidance_scale_list = [7.5,7.5,7.5,3,3,6,6,4]
seeds_list = [0,1,0,0,1,0,0,0]

for i, seed in enumerate(seeds_list):
    set_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed) 
    init_image = Image.open(folder_name + image_path_list[i]).convert('RGB').resize((512,512))
    mask_prompt = mask_prompt_list[i]
    prompt = prompt_list[i]
    mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
    image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
    image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents, generator=generator, guidance_scale=guidance_scale_list[i]).images[0]
    image.save('{}/{}.jpg'.format(save_path,prompt))
    
df = pd.DataFrame(
    {'image path':image_path_list,
     'base prompt':mask_prompt_list,
     'edit prompt':prompt_list,
     'seed': seeds_list,
     'guidance scale': guidance_scale_list,
    })

df.to_csv('diffedit_params.csv',index=False)