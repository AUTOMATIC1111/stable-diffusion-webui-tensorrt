import asyncio
import html
import os

import launch
import trt_paths
from modules import script_callbacks, paths_internal, shared
import gradio as gr

import export_onnx
import export_trt
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui_components import FormRow


def export_unet_to_onnx(filename, opset, batch_run, batch_directory):
    print(f'Starting Conversion to .onnx')
    
    # Check if 'Unet-onnx' directory exists and create it if not
    unet_onnx_path = os.path.join(paths_internal.models_path, "Unet-onnx")
    os.makedirs(unet_onnx_path, exist_ok=True)
    
    # Use default folder if batch_directory is empty
    if not batch_directory:
        batch_directory = os.path.join(paths_internal.models_path, "Stable-diffusion")
    # Batch mode
    if batch_run:
        print(f"--Batch Models mode--")
        
        onnx_files = os.listdir(os.path.join(paths_internal.models_path, "Unet-onnx"))
        onnx_files_to_process = list()
        # Take all files if destination folder is empty
        if not onnx_files:
            print(f"Unet-onnx is empty, adding all .safetensors and .ckpt files from {batch_directory}\n")  # Debug line
            onnx_files_to_process = [file for file in os.listdir(batch_directory) if file.endswith(".safetensors") or file.endswith(".ckpt")]
        else:
            for batch_file in os.listdir(batch_directory):
                add_flag = True
                for onnx_file in onnx_files:
                    if batch_file.split('.')[0] == onnx_file.split('.')[0]:
                        add_flag = False
                if add_flag and (batch_file.endswith(".safetensors") or batch_file.endswith(".ckpt")):
                    onnx_files_to_process.append(batch_file)
        print(f"Files to process:\n{onnx_files_to_process}\n")
        # Exit if no files to process
        if not onnx_files_to_process:
            print("No files to convert...\nPlease uncheck the 'Run Batch' checkbox or use a folder containing models.")
            return
        # Process files
        for i, file in enumerate(onnx_files_to_process):
            print(f"Converting model file: {file}")  # Debug line
            modelname = os.path.splitext(file)[0] + ".onnx"
            onnx_filename = os.path.join(unet_onnx_path, modelname)
            print(f"Target ONNX filename: {onnx_filename}\n")  # Debug line
            
            export_onnx.export_current_unet_to_onnx(onnx_filename, opset)
            
        # Ending message
        print(f'Batch conversion completed for files in {batch_directory}')
        return
    # Single mode
    else:
        print(f"--Single Model mode--")
        if not filename:
            modelname = shared.sd_model.sd_checkpoint_info.model_name + ".onnx"
            onnx_filename = os.path.join(unet_onnx_path, modelname)
        print(f"Target ONNX filename: {onnx_filename}\n")  # Debug line
        
        export_onnx.export_current_unet_to_onnx(onnx_filename, opset)

        # Ending message
        print(f'Done! Model saved as {onnx_filename}')
        return


def convert_onnx_to_trt(filename, onnx_filename, batch_run, batch_directory, *args):
    assert not cmd_opts.disable_extension_access, "Won't run the command to create TensorRT file because extension access is disabled (use --enable-insecure-extension-access)"
    print(f'Starting Conversion to .trt')
    
    # Check if 'Unet-trt' directory exists and create it if not
    unet_onnx_path = os.path.join(paths_internal.models_path, "Unet-trt")
    os.makedirs(unet_onnx_path, exist_ok=True)
    
    # Use default folder if batch_directory is empty
    if not batch_directory:
        batch_directory = os.path.join(paths_internal.models_path, "Unet-onnx")
    # Batch mode
    if batch_run:
        print(f"--Batch Models mode--")
        
        trt_files = os.listdir(os.path.join(paths_internal.models_path, "Unet-trt"))
        trt_files_to_process = list()
        # Take all files if destination folder is empty
        if not trt_files:
            print(f"Unet-trt is empty, adding all .onnx files from {batch_directory}\n")  # Debug line
            trt_files_to_process = [file for file in os.listdir(batch_directory) if file.endswith(".onnx")]
        else:
            for batch_file in os.listdir(batch_directory):
                add_flag = True
                for trt_file in trt_files:
                    if batch_file.split('.')[0] == trt_file.split('.')[0]:
                        add_flag = False
                if add_flag and batch_file.endswith(".onnx"):
                    trt_files_to_process.append(batch_file)
        print(f"Files to process:\n{trt_files_to_process}\n")
        # Exit if no files to process
        if not trt_files_to_process:
            print("No files to convert...\nPlease uncheck the 'Run Batch' checkbox or use a folder containing models.")
            return
        # Process files
        for file in trt_files_to_process:
            onnx_file = os.path.join(batch_directory, file)
            print(f"Converting ONNX file: {onnx_file}")  # Debug line
            modelname = os.path.splitext(file)[0] + ".trt"
            filename = os.path.join(paths_internal.models_path, "Unet-trt", modelname)

            trt_filename = get_trt_filename(filename, onnx_file, batch_run)
            print(f"Target TRT filename: {trt_filename}\n")  # Debug line
            command = export_trt.get_trt_command(trt_filename, onnx_file, *args)
            launch.run(command, live=True)
        # Ending message
        print(f'Batch conversion completed for files in {batch_directory}')
        return
    # Single mode
    else:
        print(f"--Single Model mode--")
        filename = get_trt_filename(filename, onnx_filename)
        print(f"Target TRT filename: {filename}\n")  # Debug line
        command = export_trt.get_trt_command(filename, onnx_filename, *args)
        launch.run(command, live=True)

        # Ending message
        print(f'Done! Model saved as {filename}')
        return


def get_trt_filename(filename, onnx_filename, batch_run=False, *args):
    if batch_run:
        modelname = os.path.splitext(os.path.basename(onnx_filename))[0] + ".trt"
        return os.path.join(paths_internal.models_path, "Unet-trt", modelname)
    else:
        return filename


def get_trt_command(filename, onnx_filename, *args):
    filename = get_trt_filename(filename, onnx_filename)
    command = export_trt.get_trt_command(filename, onnx_filename, *args)

    env_command = f"""
    set PATH=%PATH%;{trt_paths.cuda_path}\\lib
    set PATH=%PATH%;{trt_paths.trt_path}\\lib
    """

    run_command = f"""
    {command}
    """

    return "Command generated", f"""
<p>
Environment variables: <br>
<pre style="white-space: pre-line;">
{html.escape(env_command)}
</pre>
</p>
<p>
Command: <br>
<pre style="white-space: pre-line;">
{html.escape(run_command)}
</pre>
</p>
"""


async def calculate_and_check_constraints(max_width, max_height, max_batch_size):
    B = max_batch_size * 2
    unknown = 4
    H = max_height / 8
    W = max_width / 8

    calculated_value = int(B * unknown * H * W)

    return f"{calculated_value} / 92160", ""


def on_button_clicked(max_width, max_height, max_bs):
    loop = asyncio.get_event_loop()
    calculated_value, _ = loop.run_until_complete(calculate_and_check_constraints(max_width, max_height, max_bs))
    return calculated_value, ""


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as trt_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="trt_tabs"):
                    with gr.Tab(label="Convert to ONNX"):
                        gr.HTML(value="<p style='margin-bottom: 0.7em'>Convert currently loaded checkpoint into ONNX. The conversion will fail catastrophically if TensorRT was used at any point prior to conversion, so you might have to restart webui before doing the conversion.</p>")

                        onnx_filename = gr.Textbox(label='Filename', value="", elem_id="onnx_filename", info="Leave empty to use the same name as model and put results into models/Unet-onnx directory")
                        onnx_opset = gr.Number(label='ONNX opset version', precision=0, value=17, info="Leave this alone unless you know what you are doing")

                        batch_run_onnx = gr.Checkbox(label='Run Batch', value=False)
                        batch_directory_onnx = gr.Textbox(label='Directory', value="", info="Input directory containing models. Leave empty to use the default 'models/Stable-diffusion' folder as source.")

                        button_export_unet = gr.Button(value="Convert Unet to ONNX", variant='primary', elem_id="onnx_export_unet")

                    with gr.Tab(label="Convert ONNX to TensorRT"):
                        trt_source_filename = gr.Textbox(label='Onnx model filename', value="", elem_id="trt_source_filename")
                        trt_filename = gr.Textbox(label='Output filename', value="", elem_id="trt_filename", info="Leave empty to use the same name as onnx and put results into models/Unet-trt directory")

                        with gr.Column(elem_id="trt_width"):
                            min_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Minimum width", value=512, elem_id="trt_min_width")
                            max_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Maximum width", value=512, elem_id="trt_max_width")

                        with gr.Column(elem_id="trt_height"):
                            min_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Minimum height", value=512, elem_id="trt_min_height")
                            max_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Maximum height", value=512, elem_id="trt_max_height")

                        with gr.Column(elem_id="trt_batch_size"):
                            min_bs = gr.Slider(minimum=1, maximum=16, step=1, label="Minimum batch size", value=1, elem_id="trt_min_bs")
                            max_bs = gr.Slider(minimum=1, maximum=16, step=1, label="Maximum batch size", value=1, elem_id="trt_max_bs")

                        with gr.Column(elem_id="trt_token_count"):
                            min_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Minimum prompt token count", value=75, elem_id="trt_min_token_count")
                            max_token_count = gr.Slider(minimum=75, maximum=750, step=75, label="Maximum prompt token count", value=75, elem_id="trt_max_token_count")

                        button_check_constraints = gr.Button(value="Calculate values limit for conversion", variant='secondary', elem_id="calculate")
                                                
                        with gr.Column(elem_id="trt_calculated_value"):
                            calculated_value_label = gr.Label(elem_id="calculated_value", value="32768 / 92160", label="Current Value / Limit", show_label=True)

                        trt_extra_args = gr.Textbox(label='Extra arguments', value="", elem_id="trt_extra_args", info="Extra arguments for trtexec command in plain text form")

                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            use_fp16 = gr.Checkbox(label='Use half floats', value=True, elem_id="trt_fp16")

                        batch_run_trt = gr.Checkbox(label='Run Batch', value=False)
                        batch_directory_trt = gr.Textbox(label='Directory', value="", info="Input directory containing models. Leave empty to use the default 'models/Unet-onnx' folder as source.")

                        button_export_trt = gr.Button(value="Convert ONNX to TensorRT", variant='primary', elem_id="trt_convert_from_onnx")
                        button_show_trt_command = gr.Button(value="Show command for conversion", variant='secondary', elem_id="trt_convert_from_onnx")

            with gr.Column(variant='panel'):
                trt_result = gr.Label(elem_id="trt_result", value="", show_label=False)
                trt_info = gr.HTML(elem_id="trt_info", value="")
                button_check_constraints
                calculated_value_label

        button_export_unet.click(
            wrap_gradio_gpu_call(export_unet_to_onnx, extra_outputs=["Conversion failed"]),
            inputs=[onnx_filename, onnx_opset, batch_run_onnx, batch_directory_onnx],
            outputs=[trt_result, trt_info],
        )

        button_check_constraints.click(
            on_button_clicked,
            inputs=[max_width, max_height, max_bs],
            outputs=[calculated_value_label, trt_info],
        )

        button_export_trt.click(
            wrap_gradio_gpu_call(convert_onnx_to_trt, extra_outputs=[""]),
            inputs=[trt_filename, trt_source_filename, batch_run_trt, batch_directory_trt, min_bs, max_bs, min_token_count, max_token_count, min_width, max_width, min_height, max_height, use_fp16, trt_extra_args],
            outputs=[trt_result, trt_info],
        )

        button_show_trt_command.click(
            wrap_gradio_gpu_call(get_trt_command, extra_outputs=[""]),
            inputs=[trt_filename, trt_source_filename, min_bs, max_bs, min_token_count, max_token_count, min_width, max_width, min_height, max_height, use_fp16, trt_extra_args],
            outputs=[trt_result, trt_info],
        )

    return [(trt_interface, "TensorRT", "tensorrt")]
    
