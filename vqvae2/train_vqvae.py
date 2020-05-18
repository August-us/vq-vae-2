import argparse

import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from data_gen import get_dataset
from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

from tensorboardX import SummaryWriter
writer = SummaryWriter('./allTxlog/txlog_nembed32')
def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img,file) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )
        writer.add_scalar('mse',recon_loss.item(),epoch)
        writer.add_scalar('latent',latent_loss.item(),epoch)
        writer.add_scalar('avg_mse',mse_sum/mse_n,epoch)
        writer.add_scalar('loss',loss,epoch)
        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f'allSample/sample512/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--batchsize',type=int,default=64)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    loader = get_dataset(args.path,batch_size=args.batchsize)
    model = nn.DataParallel(VQVAE(embed_dim=32)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'allCheckpoint/checkpoint32/vqvae_{str(i + 1).zfill(3)}.pt'
        )
    writer.close()
