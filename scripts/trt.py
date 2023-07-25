import os
import numpy as np

import torch

from modules import script_callbacks, sd_unet, devices, shared, paths_internal

import trt_paths
import ui_trt

import pycuda.driver as cuda
from tensorrt import Dims


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, filename, name):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.filename = filename

    def create_unet(self):
        return TrtUnet(self.filename)


np_to_torch = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int32: torch.int32,
}


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.engine = None
        self.trtcontext = None
        self.buffers = None
        self.buffers_shape = ()
        self.nptype = None

    def allocate_buffers(self, feed_dict):

        buffers_shape = sum([x.shape for x in feed_dict.values()], ())
        if self.buffers_shape == buffers_shape:
            return

        self.buffers_shape = buffers_shape
        self.buffers = {}

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = self.nptype(self.engine.get_binding_dtype(binding))

            if binding in feed_dict:
                shape = Dims(feed_dict[binding].shape)
            else:
                shape = self.trtcontext.get_binding_shape(binding_idx)

            if self.engine.binding_is_input(binding):
                if not self.trtcontext.set_binding_shape(binding_idx, shape):
                    raise Exception(f'bad shape for TensorRT input {binding}: {tuple(shape)}')

            tensor = torch.empty(tuple(shape), dtype=np_to_torch[dtype], device=devices.device)
            self.buffers[binding] = tensor

    def infer(self, feed_dict):
        self.allocate_buffers(feed_dict)

        for name, tensor in feed_dict.items():
            self.buffers[name].copy_(tensor)

        for name, tensor in self.buffers.items():
            self.trtcontext.set_tensor_address(name, tensor.data_ptr())

        ctx = cuda.Context.attach()
        stream = cuda.Stream()

        self.trtcontext.execute_async_v3(stream.handle)

        stream.synchronize()
        ctx.detach()

    def forward(self, x, timesteps, context, *args, **kwargs):
        self.infer({"x": x, "timesteps": timesteps, "context": context})

        return self.buffers["output"].to(dtype=x.dtype, device=devices.device)

    def activate(self):
        import tensorrt as trt  # we import this late because it breaks torch onnx export

        TRT_LOGGER = trt.Logger()
        trt.init_libnvinfer_plugins(None, "")
        self.nptype = trt.nptype

        with open(self.filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.trtcontext = self.engine.create_execution_context()

    def deactivate(self):
        self.engine = None
        self.trtcontext = None
        self.buffers = None
        self.buffers_shape = ()
        devices.torch_gc()


def list_unets(l):

    trt_dir = os.path.join(paths_internal.models_path, 'Unet-trt')
    candidates = list(shared.walk_files(trt_dir, allowed_extensions=[".trt"]))
    for filename in sorted(candidates, key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]

        opt = TrtUnetOption(filename, name)
        l.append(opt)


script_callbacks.on_list_unets(list_unets)

script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
