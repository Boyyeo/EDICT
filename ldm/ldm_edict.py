from edict_functions_ldm import *
import os 

def plot_EDICT_outputs(im_tuple):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im_tuple[0])
    ax1.imshow(im_tuple[1])
    plt.show()

path = 'edit_result_edict_ldm'
os.makedirs(path, exist_ok=True)


###################################### Reconstruction #####################################################################

im = load_im_into_format_from_path('../experiment_images/church.jpg')
prompt = 'A church'
run_baseline = False

latents = coupled_stablediffusion(prompt,
                               reverse=True,
                                init_image=im,
                                run_baseline=run_baseline,
                               )
if run_baseline:
    latents = latents[0]
recon = coupled_stablediffusion(prompt,
                               reverse=False,
                                fixed_starting_latent=latents,
                                run_baseline=run_baseline,
                               )
recon = recon[0]

fig, (ax0, ax1) = plt.subplots(1,2)
ax0.imshow(im)
ax0.set_title("Original")
ax1.imshow(recon)
ax1.set_title("Recon")
plt.tight_layout()
plt.savefig('{}/recon_kd.jpg'.format(path))


###################################### EDITING #####################################################################


im_path = '../experiment_images/imagenet_cake.jpg'
base_prompt = 'A cupcake'
#display(load_im_into_format_from_path(im_path))
for edit_prompt in ['An Easter cupcake',
                   'A hedgehog cupcake',
                   'An England Union Jack cupcake',
                   'A Chinese New Year cupcake',
                   'A rainbow cupcake']:
    print(edit_prompt)
    edited_image = EDICT_editing(im_path,base_prompt, edit_prompt, steps=50, mix_weight=0.93,init_image_strength=0.8,guidance_scale=3)[0]
    edited_image.save('{}/edit_{}.jpg'.format(path,edit_prompt.replace(' ','_')))


for i in range(1, 8):
    im_path = f'../experiment_images/imagenet_dog_{i}.jpg'
    base_prompt = 'A dog' # poodle, dalmatian, lab, german shepherd
    print("Original")
    load_im_into_format_from_path(im_path)
    for breed in ['golden retriever', 'chihuahua', 'poodle', 'dalmatian', 'german shepherd', 'husky']:
        print(i, breed)
        edit_prompt = f'A {breed}'
        edited_image = EDICT_editing(im_path, base_prompt, edit_prompt, steps=50, mix_weight=0.93,init_image_strength=0.8,guidance_scale=3)[0]
        edited_image.save('{}/edit_{}_{}.jpg'.format(path,edit_prompt.replace(' ','_'),i))
