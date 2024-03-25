import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack


def l2_norm(x):
	return F.normalize(x, p=2, dim=-1)


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(3, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], dim, 3, 1, 1))
        self.num_patches = 16 * 16
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class Codebook(nn.Module):
    def __init__(self, codebook_size=1024, codebook_dim=256, beta=0.25):
        super(Codebook, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):

        # for computing the difference between z and embeddings
        z = rearrange(z, 'b d h w -> b h w d')
        z = l2_norm(z)
        z_flattened = rearrange(z, 'b h w d -> (b h w) d') 

        embedd_norm = l2_norm(self.embedding.weight)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
			torch.sum(embedd_norm**2, dim=1) - 2 * \
			torch.einsum('bd,nd->bn', z_flattened, embedd_norm)

        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices) # z_q shape is (n_indices, codebook_dim)

        b , h, w, d = z.shape
        z_q = rearrange(z_q, '(b h w) d -> b h w d', h=h, w=w)
        z_q = l2_norm(z_q)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        # for decoder 
        z_q = rearrange(z_q, 'b h w d -> b d h w')

        return z_q, min_encoding_indices, loss

    def indices_to_embeddings(self, indices):
        embeds = self.embedding(indices)
        h = w = embeds.shape[1] ** 0.5 
        embeds = rearrange(embeds, 'b (h w) d -> b d h w', h=int(h), w=int(w))
        return embeds

        


class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class VQGAN(nn.Module):
    def __init__(self, dim, codebook_size):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(dim)
        self.pre_quant = nn.Conv2d(dim, dim, 1)
        self.codebook = Codebook(codebook_size, dim)
        self.post_quant = nn.Conv2d(dim, dim, 1)
        self.decoder = Decoder(dim)

    def forward(self, imgs):
        enc_imgs = self.encoder(imgs)
        enc_imgs = self.pre_quant(enc_imgs)
        embeds, indices, loss = self.codebook(enc_imgs)
        embeds = self.post_quant(embeds)
        out = self.decoder(embeds)
        return out, loss
    
    def decode_indices(self, indices):
        embeds = self.codebook.indices_to_embeddings(indices)
        embeds = self.post_quant(embeds)
        imgs = self.decoder(embeds)
        return imgs
    
    def encode_imgs(self, imgs):
        b = imgs.shape[0]
        enc_imgs = self.encoder(imgs)
        enc_imgs = self.pre_quant(enc_imgs)
        _, indices, _ = self.codebook(enc_imgs)
        indices = rearrange(indices, '(b i) -> b i', b=b)
        return indices
    
    @property
    def num_patches(self):
        return self.encoder.num_patches

