import os.path


def get_trt_command(trt_filename, onnx_filename, min_bs, max_bs, min_token_count, max_token_count, min_width, max_width, min_height, max_height, use_fp16, trt_extra_args):
    
    for val, name in zip([min_width, max_width, min_height, max_height], ["min_width", "max_width", "min_height", "max_height"]):
        assert val % 64 == 0, name + ' must be a multiple of 64'

    for val, name in zip([min_token_count, max_token_count], ["min_token_count", "max_token_count"]):
        assert val % 75 == 0, name + ' must be a multiple of 75'

    assert os.path.isfile(onnx_filename), 'onnx model not found: ' + onnx_filename

    import trt_paths
    trt_exec_candidates = [
        os.path.join(trt_paths.trt_path, "bin", "trtexec"),
        os.path.join(trt_paths.trt_path, "bin", "trtexec.exe"),
    ]

    trt_exec = next(iter([x for x in trt_exec_candidates if os.path.isfile(x)]), None)
    assert trt_exec, f"could not find trtexec; searched in: {', '.join(trt_exec_candidates)}"

    cond_dim = 768  # XXX should be detected for SD2.0
    x_min = f"{min_bs * 2}x4x{min_height // 8}x{min_width // 8}"
    x_max = f"{max_bs * 2}x4x{max_height // 8}x{max_width // 8}"
    context_min = f"{min_bs * 2}x{min_token_count // 75 * 77}x{cond_dim}"
    context_max = f"{max_bs * 2}x{max_token_count // 75 * 77}x{cond_dim}"
    timestamps_min = f"{min_bs * 2}"
    timestamps_max = f"{max_bs * 2}"

    os.makedirs(os.path.dirname(trt_filename), exist_ok=True)

    return f""""{trt_exec}" --onnx="{onnx_filename}" --saveEngine="{trt_filename}" --minShapes=x:{x_min},context:{context_min},timesteps:{timestamps_min} --maxShapes=x:{x_max},context:{context_max},timesteps:{timestamps_max}{' --fp16' if use_fp16 else ''} {trt_extra_args}"""
