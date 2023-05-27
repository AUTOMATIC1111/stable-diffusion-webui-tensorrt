import os

from modules import sd_hijack, sd_unet
from modules import shared, devices
import torch


def export_current_unet_to_onnx(filename, opset_version=17):
    x = torch.randn(1, 4, 16, 16).to(devices.device, devices.dtype)
    timesteps = torch.zeros((1,)).to(devices.device, devices.dtype) + 500
    context = torch.randn(1, 77, 768).to(devices.device, devices.dtype)

    def disable_checkpoint(self):
        if getattr(self, 'use_checkpoint', False) == True:
            self.use_checkpoint = False
        if getattr(self, 'checkpoint', False) == True:
            self.checkpoint = False

    shared.sd_model.model.diffusion_model.apply(disable_checkpoint)

    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.apply_optimizations('None')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with devices.autocast():
        torch.onnx.export(
            shared.sd_model.model.diffusion_model,
            (x, timesteps, context),
            filename,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['x', 'timesteps', 'context'],
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'batch_size', 2: 'height', 3: 'width'},
                'timesteps': {0: 'batch_size'},
                'context': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'},
            },
        )

    sd_hijack.model_hijack.apply_optimizations()
    sd_unet.apply_unet()
