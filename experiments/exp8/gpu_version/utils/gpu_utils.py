import torch

class GPUConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

gpu_config = GPUConfig()
