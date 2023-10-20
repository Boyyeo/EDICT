#from diffusers import DiffusionPipeline
from ldm_pipeline import MyLDMTextToImagePipeline

gpu_id = 0

# load model and scheduler
ldm = MyLDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
ldm = ldm.to("cuda:{}".format(gpu_id))
print("vae:{} unet:{}".format(ldm.vqvae,ldm.unet))
# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

# save images
for idx, image in enumerate(images):
    image.save(f"squirrel-{idx}.png")