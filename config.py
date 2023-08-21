import argparse

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

