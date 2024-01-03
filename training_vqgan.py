import os
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
import matplotlib.pyplot as plt

from torch import autocast
from torch.cuda.amp import GradScaler


class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.vqgan.load_state_dict(
            torch.load(os.path.join("/media/userdisk1/code/VQGAN-pytorch/checkpoints", "vqgan_epoch_308.pt")))
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.load_state_dict(
            torch.load(os.path.join("/media/userdisk1/code/VQGAN-pytorch/checkpoints", "discriminator_epoch_308.pt")))

        # self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        # self.usage_codebook = np.zeros(shape=(args.num_codebook_vectors))

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(309, args.epochs):
            usage_codebook = np.zeros(shape=(args.num_codebook_vectors))
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.discriminator.zero_grad()
                    self.opt_disc.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)

                        with torch.no_grad():
                            decoded_images, _, q_loss = self.vqgan(imgs)
                        disc_real = self.discriminator(imgs)
                        disc_fake = self.discriminator(decoded_images.detach())
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i,
                                                              threshold=args.disc_start)
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    scaler.scale(gan_loss).backward()
                    scaler.step(self.opt_disc)
                    scaler.update()

                    # gan_loss.backward()
                    # print(self.vqgan.decoder.conv_out.weight.grad.max(), self.discriminator.model[-1].weight.grad.max())

                    # self.opt_disc.step()

                    self.vqgan.zero_grad()
                    self.opt_vq.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # imgs = imgs.to(device=args.device)
                        decoded_images, min_encoding_indices, q_loss = self.vqgan(imgs)
                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        rec_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        disc_fake = self.discriminator(decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        vq_loss = perceptual_rec_loss + q_loss + disc_factor * lbd * g_loss
                    min_encoding_indices_int = min_encoding_indices.cpu()
                    for j in range(len(min_encoding_indices_int)):
                        usage_codebook[min_encoding_indices_int[j]] += 1

                    scaler.scale(vq_loss).backward()
                    scaler.step(self.opt_vq)
                    scaler.update()

                    # self.opt_disc.zero_grad()
                    # gan_loss.backward()

                    # self.opt_disc.step()

                    if i % 1000 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4],
                                                          torch.clamp(decoded_images.add(1).mul(0.5), min=0.0, max=1)[
                                                          :4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                        lbd=lbd.data.cpu().numpy()
                    )
                    pbar.update(0)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))
                torch.save(self.discriminator.state_dict(),
                           os.path.join("checkpoints", f"discriminator_epoch_{epoch}.pt"))
                plt.imshow(usage_codebook.reshape(32, 32))
                plt.savefig(os.path.join("checkpoints", f"codebook_epoch_{epoch}.png"))
                np.save(os.path.join("checkpoints", f"codebook_epoch_{epoch}.npy"), usage_codebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                        help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0,
                        help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = r"/media/userdisk1/code/jpg/"

    train_vqgan = TrainVQGAN(args)
