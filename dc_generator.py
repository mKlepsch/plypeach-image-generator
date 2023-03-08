import random
import os
import glob
##pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
## others
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# dc_gan
from dc_gan import Generator,Discriminator,weights_init,save_checkpoint
from utils import newest_file,make_animation_from_folder


# Set random seed for reproducibility
#manualSeed = np.random.randint(low=10,high=200)
manualSeed = 42
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
datasource = "/home/max/Schreibtisch/max_processed_rgba"
dataload_workers = 8
output_dir = "/mnt/data/spiced/final_project/dcgan_ply_peach"
model_dir = f"{output_dir}/checkpoints/"
training_dir = f"{output_dir}/training_pics/"
batch_size = 64
number_of_color_channels = 3
latent_vector_size = 64
feature_map_generator = 64
feature_map_discriminator = 64
number_epochs = 3000
learning_rate = 0.0002
beta_adam = 0.5
number_of_gpus = 2
image_size = 64
resume = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### data import/loarder
dataset = datasets.ImageFolder(  root=datasource,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(number_of_color_channels)], [0.5 for _ in range(number_of_color_channels)])
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,pin_memory=True, persistent_workers=True,
                                         shuffle=True, num_workers=dataload_workers)

netG = Generator(channels_noise=latent_vector_size, channels_img=number_of_color_channels, features_g=feature_map_generator).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (number_of_gpus > 1):
    netG = nn.DataParallel(netG, list(range(number_of_gpus)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)
netD = Discriminator( channels_img=number_of_color_channels, features_d=feature_map_discriminator).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (number_of_gpus > 1):
    netD = nn.DataParallel(netD, list(range(number_of_gpus)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(image_size, latent_vector_size, 1, 1, device=device)

# Setup Adam optimizers for both G and D64, nz, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta_adam, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta_adam, 0.999))

###load checkpoint stuff....
checkpoint_epoch=0
if resume:
    latest = newest_file(model_dir)
    checkpoint = torch.load(latest,map_location="cpu")
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    netG.load_state_dict(checkpoint['netG_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    checkpoint_epoch = checkpoint['epoch']
    checkpoint_epoch = checkpoint_epoch + 1 ### incremtn by one for the next epoch
    del checkpoint
### start....
netD.train()
netG.train()
fixed_noise = torch.randn(image_size, latent_vector_size, 1, 1, device=device)


print("Starting Training Loop...")
# For each epoch
for epoch in range(checkpoint_epoch,number_epochs):
    # For each batch in the dataloader
    for batch_idx, (data, what) in enumerate(dataloader): ## get real data
        noise = torch.randn(batch_size, latent_vector_size, 1, 1).to(device) 
        real = data.to(device) + torch.normal(mean=0.0, std=0.1, size=data.size()).to(device) #### add noise to the real images 
        fake = netG(noise) # generate fake
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = netD(real).reshape(-1) # reshape data
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) ###calculate loss for the real data for the discriminator labels 1
        
        disc_fake = netD(fake.detach()).reshape(-1) # reshape data
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) ###calculate loss for the fake data for the discriminator labels 0
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        netD.zero_grad()
        loss_disc.backward()
        optimizerD.step()
        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = netD(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        netG.zero_grad()
        loss_gen.backward()
        optimizerG.step()
        
        if (batch_idx % 5) == 0 :
            print(
                f"Epoch [{epoch}/{number_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
    
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        img_grid_fake = torchvision.utils.make_grid(fake, padding=2, normalize=True).detach().cpu()
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        torchvision.utils.save_image(img_grid_fake, f"{training_dir}/training_image-{epoch}.png")
    
    if (epoch % 20)==0:
        print(f"Epoch {epoch}: Saving checkpoint")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        current_checkpoint = f"{model_dir}/checkpoint_{epoch}.tar"
        print(f"Saving: {current_checkpoint}")
        save_checkpoint(epoch=epoch,generator_model=netG,disriminator_model=netD,generator_optimizer=optimizerG,disriminator_optimizer=optimizerD,path=current_checkpoint)

final_checkpoint = f"{model_dir}/checkpoint_{epoch}.tar"
save_checkpoint(epoch=epoch,generator_model=netG,disriminator_model=netD,generator_optimizer=optimizerG,disriminator_optimizer=optimizerD,path=final_checkpoint)
make_animation_from_folder(input_folder=training_dir,output_movie=f'{output_dir}/training_after_{epoch}.mp4')