# TensorRT support for webui

Adds the ability to convert loaded model's Unet module into TensortRT. Requires version least after commit [339b5315](htts://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/339b5315700a469f4a9f0d5afc08ca2aca60c579) (currently, it's the `dev` branch after 2023-05-27). Only tested to work on Windows.

Loras are baked in into the converted model. Hypernetwork support is not tested. Controlnet is not supported. Textual inversion works normally.

NVIDIA is also working on releaseing their version of TensorRT for webui, which might be more performant, but they can't release it yet.

There seems to be support for quickly replacing weight of a TensorRT engine without rebuilding it, and this extension does not offer this option yet.

## How to install

Apart from installing the extension normally, you also need to download zip with TensorRT from [NVIDIA](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

You need to choose the same version of CUDA as python's torch library is using. For torch 2.0.1 it is CUDA 11.8.

Extract the zip into extension directory, so that `TensorRT-8.6.1.6` (or similarly named dir) exists in the same place as `scripts` directory and `trt_path.py` file. Restart webui afterwards.

You don't need to install CUDA separately.

## How to use

1. Select the model you want to optimize and make a picture with it, including needed loras and hypernetworks.
2. Go to a `TensorRT` tab that appears if the extension loads properly.
3. In `Convert to ONNX` tab, press `Convert Unet to ONNX`.
   * This takes a short while.
   * It uses up around 6GB of VRAM.
   * After the conversion has finished, you will find an `.onnx` file with model in `models/Unet-onnx` directory.
4. In `Convert ONNX to TensorRT` tab, configure the necessary parameters (including writing full path to onnx model) and press `Convert ONNX to TensorRT`.
   * This takes very long - from 15 minues to an hour.
   * This takes up a lot of VRAM, around 4GB: you might want to press "Show command for conversion" and run the command yourself after shutting down webui.
   * After the conversion has finished, you will find a `.trt` file with model in `models/Unet-trt` directory.
5. In settings, in `Stable Diffusion` page, use `SD Unet` option to select newly generated TensorRT model.
6. Generate pictures.

## Stable Diffusion 2.0 support
Stable diffusion 2.0 conversion should fail for both ONNX and TensorRT because of incompatible shapes, but you may be able to rememdy this by chaning instances of 768 to 1024 in the code.
