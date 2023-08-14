import numpy as np
import torch
from torchvision import datasets, transforms

# load mnist data
def load_mnist(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        ),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

