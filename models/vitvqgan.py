import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack





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
        z_flattened = rearrange(z, 'b h w d -> (b h w) d') 

        # D - distance between z and embeddings :  z(b*h*w,d) - embeddings(n,D) 
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices) # z_q shape is (n_indices, codebook_dim)

        b , h, w, d = z.shape
        z_q = rearrange(z_q, '(b h w) d -> b h w d', h=h, w=w)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        # for decoder 
        z_q = rearrange(z_q, 'b h w d -> b d h w')

        return z_q, min_encoding_indices, loss

    def indices_to_embeddings(self, indices):
        embeds = self.embedding(indices)
        h = w = embeds.shape[0] ** 0.5 
        embeds = rearrange(embeds, '(b h w) d -> b d h w', h=int(h), w=int(w))
        return embeds

        


class ViTVQGAN(nn.Module):
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
        imgs = self.decoder(embeds)
        return imgs
    
    def encode_imgs(self, imgs):
        enc_imgs = self.encoder(imgs)
        enc_imgs = self.pre_quant(enc_imgs)
        _, indices, _ = self.codebook(enc_imgs)
        return indices


if __name__ == '__main__':

    codebook_dim = 32
    codebook_size = 8192

    vqgan = VQGAN(codebook_dim, codebook_size)

    img = torch.randn(2, 3, 256, 256)
    out, loss = vqgan(img)
    print(loss)

    img = torch.randn(1, 3, 256, 256)
    indices = vqgan.encode_imgs(img)
    img = vqgan.decode_indices(indices)
    print(img.shape)

