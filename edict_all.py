from edict_functions import *
import os 
import pandas as pd 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
   

save_path = 'EDICT_RESULT'
os.makedirs(save_path, exist_ok=True)

###################################### EDITING #####################################################################




folder_name = 'ORIGINAL_IMAGES/'
image_path_list = ['two_birds.png','cat1.png','woman_blue_hair.png','sea.png','sky.jpg','yellow_cat.jpeg','white_horse.png','cat_chair.png']
base_prompt_list = ['Two birds sitting on a branch','A photo of a cat','A girl with blue hair','A sea','A sky','A cat','A photo of a horse','A cat on chair']
edit_prompt_list = ['Two lego birds sitting on a branch','A photo of a tiger','A girl with black hair','A boat on a sea','A sky in cartoon style','A cat in watercolor painting','A photo of a zebra','A ferret on chair']
guidance_scale_list = [3,3,5,7.5,4,5,4,4]
seeds_list = [1,1,1,1,1,1,1,1]
mix_weight_list =[0.93,0.93,0.93,0.93,0.90,0.90,0.90,0.90]
init_image_list = [0.8,0.8,0.7,0.7,0.7,0.75,0.8,0.8]



for i,seed in enumerate(seeds_list):
    set_seed(seed)
    im_path = folder_name + image_path_list[i]
    base_prompt = base_prompt_list[i]
    edit_prompt = edit_prompt_list[i]
    mix_weight = mix_weight_list[i]
    init_image_strength = init_image_list[i]
    guidance_scale = guidance_scale_list[i]
    
    edited_image = EDICT_editing(im_path,base_prompt, edit_prompt, steps=50, mix_weight=mix_weight,init_image_strength=init_image_strength,guidance_scale=guidance_scale,seed=seed)[0]
    edited_image.save('{}/{}.jpg'.format(save_path,edit_prompt))


df = pd.DataFrame(
    {'image path':image_path_list,
     'base prompt':base_prompt_list,
     'edit prompt':edit_prompt_list,
     'seed': seeds_list,
     'guidance scale': guidance_scale_list,
     'mix weight': mix_weight_list,
     'init image strength':init_image_strength,
    })

df.to_csv('edict_params.csv',index=False)
