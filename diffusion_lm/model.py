import torch as T

class Diffusion:
    def __init__(self, noise_steps=1200, beta_start=1e-4, beta_end=0.02) -> None:
        pass

    def prepare_noise_schedule(self):
        pass

    def noise_text(self):
        pass

    def sample_timestep(self):
        pass

    def sample(self):
        pass


class UNet(T.nn.Module):
    def __init__(self) -> None:
        pass
