from typing import Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMLP(nn.Module):
    """
    Classifier to compute gradients for the hvectors

    Attributes:
        linears (nn.ModuleList): A list of linear layers, one for each timestep. 
        Each maps the input from 1*1280*8*8 (hvector dimension) to 2 (num_classes).
    """

    def __init__(self, 
        input_size: int = 1*1280*8*8, 
        output_size: int = 2, 
        num_timesteps: int = 50
    ):
        super(CustomMLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_timesteps)])

    @torch.autocast(device_type="cuda")
    def forward(self, 
        input_tensor: torch.Tensor, 
        timestep_index: int
    ):
        # Reshape the input tensor to a 2D tensor with shape (batch_size, -1)
        batch_size = input_tensor.shape[0]
        reshaped_input = input_tensor.reshape(batch_size, -1)

        output_tensor = self.linears[timestep_index](reshaped_input)
        return output_tensor


def make_model(
    path: os.PathLike, 
    device: torch.device
) -> CustomMLP:
    model = CustomMLP().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


@torch.enable_grad()
def compute_sample_gradients(
    sample: torch.Tensor, 
    timestep: int, 
    class_index: int, 
    checkpoint_path: Union[str, os.PathLike], 
) -> torch.Tensor:
    
    device = sample.device
    sample = sample.detach().requires_grad_()

    model = make_model(checkpoint_path, device)

    logits = model(sample, timestep)
    loss = -logits[:, class_index].sum()
    gradients = torch.autograd.grad(loss, sample)
    
    return gradients[0]


@torch.enable_grad()
def compute_distribution_gradients(
    sample: torch.Tensor, 
    timestep: int, 
    checkpoint_path: Union[str, os.PathLike], 
    loss_strength: float,
    temperature: Union[float, int] = 8,
) -> torch.Tensor:
    
    device = sample.device
    sample = sample.detach().requires_grad_()
    
    model = make_model(checkpoint_path, device)
   
    logits = model(sample, timestep) / temperature
    logits = F.softmax(logits, dim=1)
    loss = torch.mean(logits, dim = 0)
    
    # Assume two classes and 1:1 distribution
    loss = (loss[0]*torch.log(loss[0] / 0.5)) + (loss[1]*torch.log(loss[1] / 0.5))
    gradients = torch.autograd.grad(loss*loss_strength, sample)[0]
    
    return gradients

