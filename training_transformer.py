import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
from torch import autocast
from torch.cuda.amp import GradScaler

class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model.load_state_dict(
            torch.load(os.path.join("/media/userdisk1/code/VQGAN-pytorch/checkpoints", "transformer_167.pt")))
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        train_dataset = load_data(args)
        scaler = GradScaler()
        all_loss = 0
        best_loss = 0
        start_epoch = 167
        for epoch in range(start_epoch,args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)
                        logits, targets = self.model(imgs)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    #
                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()
                    # loss.backward()
                    # self.optim.step()
                    if i == 0:
                        all_loss = loss.cpu().detach().numpy().item()
                    else:
                        all_loss = all_loss * i / (i + 1) + loss.cpu().detach().numpy().item() / (i + 1)
                    pbar.set_postfix(Transformer_Loss=np.round(all_loss, 4))
                    pbar.update(0)

            with autocast(device_type='cuda', dtype=torch.float16):
                log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs.add(1).mul(0.5), os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
            plot_images(log)
            if epoch ==start_epoch:
                best_loss =all_loss
                torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}_{all_loss}.pt"))
            elif all_loss < best_loss:
                best_loss = all_loss
                torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}_{all_loss}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"/media/userdisk1/code/jpg/"
    args.checkpoint_path = r"/media/userdisk1/code/VQGAN-pytorch/checkpoints/vqgan_epoch_307.pt"

    train_transformer = TrainTransformer(args)


