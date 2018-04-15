"""main.py"""

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


def main(args):
    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Factor-VAE')

    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=int, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--L', default=64, type=int, help='')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the noise z')
    parser.add_argument('-gamma', default=6.4, type=float, help='')
    parser.add_argument('-lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('-beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('-beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('-lr_D', default=1e-5, type=float, help='learning rate for training the discriminator')
    parser.add_argument('-beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('-beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load last checkpoint')

    args = parser.parse_args()

    main(args)