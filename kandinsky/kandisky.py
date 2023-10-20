
from kandisky_pipeline import MyKandinskyPipeline
from diffusers import DiffusionPipeline
import torch

gpu_id = 2
pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda:{}".format(gpu_id))

t2i_pipe = MyKandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda:{}".format(gpu_id))

prompt = "A meowtwo eating a burger, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality"

image_embeds, negative_image_embeds = pipe_prior(prompt, negative_prompt, guidance_scale=1.0).to_tuple()


image = t2i_pipe(
    prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768
).images[0]
image.save("cheeseburger_monster.png")