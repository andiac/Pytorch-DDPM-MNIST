import os
import numpy as np
import torch
import torchvision
import argparse
from tqdm import tqdm

from model import Model
from utils import get_beta_schedule, load_mnist

parser = argparse.ArgumentParser(description="yo")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--z-dim", type=int, default=3072, help="Dimension of latent space")
parser.add_argument("--temb-dim", type=int, default=3072, help="Dimension of time embedding")
parser.add_argument("--num-resmlp-blocks", type=int, default=4, help="Number of resmlp blocks")
parser.add_argument("--data-path", type=str, default="../data", help="Data path")
parser.add_argument("--plot-path", type=str, default="./plot", help="Plot path")
parser.add_argument("--plot-every", type=int, default=100, help="Plot every n timesteps")
parser.add_argument("--train-batch-size", type=int, default=512, help="Train batch size")
parser.add_argument("--test-batch-size", type=int, default=512, help="Test batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--num-sample", type=int, default=36, help="Number of samples")
args = parser.parse_args()

config = {"model": {"type": "simple",
                    "in_channels": 1,
                    "out_ch": 1,
                    "ch": 128,
                    "ch_mult": [1, 2, 2],
                    "num_res_blocks": 2,
                    "attn_resolutions": [14, ],
                    "dropout": 0.1,
                    "resamp_with_conv": True,
                   }, 
          "data":  {"dataset": "MNIST",
                    "image_size": 28,
                    "channels": 3,
                   },
          "diffusion": {
                        "beta_schedule": "linear",
                        "beta_start": 0.0001,
                        "beta_end": 0.02,
                        "num_diffusion_timesteps": 1000,
                       },
         }

config = argparse.Namespace(**config)
config.model = argparse.Namespace(**config.model)
config.data = argparse.Namespace(**config.data)
config.diffusion = argparse.Namespace(**config.diffusion)

train_loader, test_loader = load_mnist(args)
model = Model(config).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
betas = get_beta_schedule(config.diffusion.beta_schedule, beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end, num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps)
betas = torch.from_numpy(betas).float().to(args.device)
if args.device == "mps":
    alpha_bars = torch.from_numpy(np.cumprod(1 - betas.cpu().numpy())).float().to(args.device)
else:
    alpha_bars = (1 - betas).cumprod(dim=0)

num_timesteps = config.diffusion.num_diffusion_timesteps

for e in range(args.epochs):
    # train
    for x, _ in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.to(args.device)
        bs = x.shape[0]
        # antithetic sampling
        t = torch.randint(
            low=0, high=num_timesteps, size=(bs // 2 + 1,)
        ).to(args.device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:bs]
        abar = alpha_bars.index_select(0, t).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        x = x * abar.sqrt() + eps * (1.0 - abar).sqrt()
        output = model(x, t.float())
        loss = torch.mean((eps - output).view(-1, 28*28).square().sum(dim=1), dim=0)
        loss.backward()
        optimizer.step()

    # test
    print("Testing...")
    with torch.no_grad():
        eval_loss = 0.
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(args.device)
            bs = x.shape[0]
            # antithetic sampling
            t = torch.randint(
                low=0, high=num_timesteps, size=(bs // 2 + 1, )
            ).to(args.device)
            t = torch.cat([t, num_timesteps - t - 1], dim=0)[:bs]
            abar = alpha_bars.index_select(0, t).view(-1, 1, 1, 1)
            eps = torch.randn_like(x)
            x = x * abar.sqrt() + eps * (1.0 - abar).sqrt()
            output = model(x, t)
            eval_loss += torch.sum((eps - output).view(-1, 28*28).square().sum(dim=1), dim=0).item()
        print("Epoch: {}, Test loss: {}".format(e, eval_loss / len(test_loader)))

    # sample
    print("Sampling...")
    plot_path = os.path.join(args.plot_path, str(e))
    os.makedirs(plot_path, exist_ok=True)
    with torch.no_grad():
        x = torch.randn(args.num_sample, 1, 28, 28).to(args.device)
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.tensor([t], dtype=torch.float).to(args.device)
            x = (x - betas[t] * model(x, t_tensor) / torch.sqrt(1 - alpha_bars[t])) / torch.sqrt(1 - betas[t])
            if t != 0:
                x += torch.exp(0.5 * torch.log(betas[t])) * torch.randn_like(x)
            if t % args.plot_every == 0:
                torchvision.utils.save_image(x, os.path.join(plot_path, f"sample_{t}.png"), padding=1, nrow=int(args.num_sample ** 0.5))


