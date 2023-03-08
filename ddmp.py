import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data,save_images,newest_file

##### UNET Model stuff
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        ## encoder above - now bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        ## decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4,beta_end=0.02,img_size=64,device='cuda') -> None:
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(start= self.beta_start,end = self.beta_end,steps=self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    
    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)): ## mybe tqdm
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


##training loop
def train(dataset_path,output_dir, image_size=64, batch_size=64,number_of_epoch= 20 ,learning_rate = 3e-4,number_of_gpus=1, device="cuda",resume=False):
    device = device
    dataloader = get_data(dataset_path, image_size = image_size,batch_size=batch_size)
    
    model = UNet().to(device)
    if (device == 'cuda') and (number_of_gpus > 1):
        model = nn.DataParallel(model, list(range(number_of_gpus)))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    checkpoint_epoch=1
    if resume:
        model_dir = f"{output_dir}/model/"
        latest = newest_file(model_dir)
        checkpoint = torch.load(latest,map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        mse.load_state_dict(checkpoint['mse'])
        checkpoint_epoch = checkpoint['epoch']
        checkpoint_epoch += 1
        del checkpoint
        model = model.to(device)

    for epoch in range(checkpoint_epoch,number_of_epoch):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            model = model.to(device)
            predicted_noise = model(x_t, t) ### checkoint problem
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch}/{number_of_epoch}] Batch {i}/{len(dataloader)} \
                  Loss: { loss.item():.4f}")
        sampled_images = diffusion.sample(model, n=8)
        model_dir = f"{output_dir}/model/"
        picture_dir = f"{output_dir}/pictures/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(picture_dir):
            os.makedirs(picture_dir)
        save_images(images = sampled_images,path = f"{picture_dir}{epoch}.jpg")
        #torch.save(model.state_dict(),f"{model_dir}{epoch}.pt")
        torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mse': mse.state_dict()},f"{model_dir}{epoch}.pt")

if __name__ == '__main__':
    dataset_path = "/home/max/Schreibtisch/max_processed_rgba"
    output_dir = "/mnt/data/spiced/final_project/ddmp_ply_peach_64x64/" 
    image_size =64
    batch_size =8 
    learning_rate = 3e-4
    device ="cuda"
    number_of_epoch = 1100
    number_of_gpus=2
    training = True
    resume = True
    if training:
        train(dataset_path,output_dir=output_dir,
        image_size=image_size,batch_size=batch_size,
        number_of_epoch=number_of_epoch, learning_rate = learning_rate,number_of_gpus=number_of_gpus, device=device,resume=resume)
    else:
        import matplotlib.pyplot as plt
        model = UNet().to(device)
        if (device == 'cuda') and (number_of_gpus > 1):
            model = nn.DataParallel(model, list(range(number_of_gpus)))
        model_dir = f"{output_dir}/model/"
        latest = newest_file(model_dir)
        print(latest)
        ckpt = torch.load(latest)
        model.load_state_dict(ckpt['model'])
        diffusion = Diffusion(img_size=64, device=device)
        x = diffusion.sample(model, 8)
        plt.figure(figsize=(32, 32))
        plt.imshow(torch.cat([
                             torch.cat([i for i in x.cpu()], dim=-1),
                             ], dim=-2).permute(1, 2, 0).cpu())
        plt.show()