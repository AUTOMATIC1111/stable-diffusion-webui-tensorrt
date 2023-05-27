import os
import torch

script_path = os.path.dirname(os.path.realpath(__file__))
trt_path = None
cuda_path = None


def set_paths():
    global trt_path, cuda_path

    cuda_path = os.path.dirname(torch.__file__)
    cuda_lib_path = os.path.join(cuda_path, "lib")

    assert os.path.exists(cuda_lib_path), "CUDA lib directory not found: " + cuda_lib_path

    looked_in = []
    trt_path = None
    for dirname in os.listdir(script_path):
        path = os.path.join(script_path, dirname)
        if not os.path.isdir(path):
            continue

        if os.path.exists(os.path.join(path, 'lib')) and (os.path.exists(os.path.join(path, 'bin', 'trtexec.exe')) or os.path.exists(os.path.join(path, 'bin', 'trtexec'))):
            trt_path = path
            break

        looked_in.append(path)

    assert trt_path is not None, "Was not able to find TensorRT directory. Looked in: " + ", ".join(looked_in)

    trt_lib_path = os.path.join(trt_path, "lib")
    if trt_lib_path not in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + trt_lib_path

    if cuda_lib_path not in os.environ['PATH']:
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + cuda_lib_path

    os.environ['CUDA_PATH'] = cuda_path  # use same cuda as torch is using


set_paths()
