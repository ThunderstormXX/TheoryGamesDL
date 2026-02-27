"""GPU Utilities for efficient tensor operations"""
import torch
import numpy as np

class GPUConfig:
    """Manage GPU device configuration"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()
    
    def to_device(self, data):
        """Convert numpy arrays or tensors to device"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data
    
    def to_cpu(self, tensor):
        """Convert tensor back to CPU numpy"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def print_info(self):
        """Print GPU information"""
        if self.use_gpu:
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("GPU not available, using CPU")

# Global GPU config
gpu_config = GPUConfig()
