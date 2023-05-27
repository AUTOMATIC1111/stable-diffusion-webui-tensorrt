import os

import launch

try:
    import trt_paths
except Exception as e:
    print("Could not find TensorRT directory; skipping install", e)


def install():
    if not launch.is_installed("tensorrt"):
        trt_whl_path = os.path.join(trt_paths.trt_path, "python")
        matching_files = [os.path.join(trt_whl_path, x) for x in os.listdir(trt_whl_path)]
        matching_files = [x for x in matching_files if "tensorrt-" in x and "cp310" in x]
        if len(matching_files) == 0:
            print(f"Could not find TensorRT .whl installer; looked in {trt_whl_path}")

        whl = matching_files[0]
        launch.run_pip(f'install "{whl}"', "TensorRT wheel")

    if not launch.is_installed("pycuda"):
        launch.run_pip(f'install pycuda', "pycuda")

    if not launch.is_installed("onnx"):
        launch.run_pip(f'install onnx', "onnx")


if trt_paths:
    install()
