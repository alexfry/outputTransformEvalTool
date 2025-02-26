import os
import subprocess
import glob
import argparse
import re
from outputTransformEvalTool import compare_images  # Import from outputTransformEvalTool.py
import matplotlib.pyplot as plt
import numpy as np
from outputTransformEvalTool import read_exr_image, compute_image_statistics  # For recomputing stats if needed

def generate_ocioconvert_commands(input_images, output_dir, start_frame=0, end_frame=9999):
    OCIO_ORIGINAL_PATH = "/Users/afry/GitHub/OpenColorIO/build/src/apps/ocioconvert/ocioconvert"
    OCIO_MODIFIED_PATH = "/Users/afry/GitHub/OpenColorIO_aces2_optimization/build/src/apps/ocioconvert/ocioconvert"
    
    for path in [OCIO_ORIGINAL_PATH, OCIO_MODIFIED_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Executable not found: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Executable permission denied: {path}. Run 'chmod +x {path}' to fix.")
    
    ViewPairs = [
        ["Rec.2100-PQ - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 500 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 2000 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (P3 D65)"],
    ]
    
    glob_pattern = input_images.replace("#", "[0-9]*")
    input_files = glob.glob(glob_pattern)
    
    os.makedirs(output_dir, exist_ok=True)
    
    commands = []
    frame_pattern = r'^(.*?)\.(\d+)\.exr$'
    
    for input_file in input_files:
        match = re.search(frame_pattern, os.path.basename(input_file))
        if not match:
            continue
            
        base_name, frame_num = match.groups()
        frame_num = int(frame_num)
        
        if frame_num < start_frame or frame_num > end_frame:
            continue
        
        frame_str = f"{frame_num:04d}"
        
        for view, display in ViewPairs:
            view_str = view.replace(" ", "_").replace(".", "").replace("-", "_")
            display_str = display.replace(" ", "_").replace(".", "").replace("-", "_").replace("(", "").replace(")", "")
            
            output_241 = os.path.join(
                output_dir,
                f"{base_name}_{view_str}_{display_str}_OCIO241.{frame_str}.exr"
            )
            output_a2o = os.path.join(
                output_dir,
                f"{base_name}_{view_str}_{display_str}_OCIOa2o.{frame_str}.exr"
            )
            
            VIEW_PARAMS = ["--view", None, "aces", None, view, display]
            
            cmd_241 = [OCIO_ORIGINAL_PATH] + VIEW_PARAMS[:]
            cmd_241[2] = input_file
            cmd_241[4] = output_241
            
            cmd_a2o = [OCIO_MODIFIED_PATH] + VIEW_PARAMS[:]
            cmd_a2o[2] = input_file
            cmd_a2o[4] = output_a2o
            
            commands.append(("OCIO241", cmd_241, output_241))
            commands.append(("OCIOa2o", cmd_a2o, output_a2o))
    
    return commands

def run_commands(commands):
    env = os.environ.copy()
    env["OCIO"] = "/Users/afry/Downloads/studio-config-aces-v1-and-v2.ocio"
    
    for version, cmd, output_path in commands:
        print(f"Running {version}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {version} command: {e}")
            return False
    return True

def plot_delta_e_stats(output_dir):
    # Find all Delta E EXR files in the output directory
    delta_e_files = glob.glob(os.path.join(output_dir, "*_DeltaITP.*.exr"))
    if not delta_e_files:
        print("No Delta E images found in the output directory.")
        return
    
    # Group files by view pair (excluding frame number and '_DeltaITP')
    stats_by_view = {}
    frame_pattern = r'^(.*?_.*?)_DeltaITP\.(\d+)\.exr$'
    
    for file_path in delta_e_files:
        match = re.search(frame_pattern, os.path.basename(file_path))
        if not match:
            continue
        base_name, frame_num = match.groups()
        frame_num = int(frame_num)
        
        # Extract view pair name (e.g., "Rec2100_PQ_Display_ACES2_0_SDR_100nits_Rec709")
        view_pair = '_'.join(base_name.split('_')[1:])  # Skip the initial base name (e.g., ACES_OT_VWG_SampleFrames)
        
        if view_pair not in stats_by_view:
            stats_by_view[view_pair] = {}
        delta_e_data = read_exr_image(file_path)
        stats = compute_image_statistics(delta_e_data)  # Recompute stats from the image data
        stats_by_view[view_pair][frame_num] = stats
    
    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#373737')
    ax.set_facecolor('#373737')
    
    metrics = ['mean', 'median', 'max', 'p95']  # Stats to plot
    colors = ['cyan', 'magenta', 'yellow', 'green']  # One color per metric
    
    for view_pair, frame_stats in stats_by_view.items():
        frames = sorted(frame_stats.keys())
        for metric, color in zip(metrics, colors):
            values = [frame_stats[frame][metric] for frame in frames]
            ax.plot(frames, values, label=f"{view_pair} {metric}", color=color, marker='o')
    
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Delta E Value")
    ax.set_title("Delta E Statistics Across Frames")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare two OCIO pipelines and analyze output differences")
    parser.add_argument(
        "--input-images",
        required=False,
        help="Input file pattern with # for frame number (e.g., '/path/to/files.#.exr')"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed files or where Delta E images are located"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame number (optional)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=9999,
        help="Ending frame number (optional)"
    )
    parser.add_argument(
        "--mode",
        choices=['SDR', 'HDR'],
        default='HDR',
        help="Mode for image comparison: 'SDR' or 'HDR' (default: HDR)"
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=0.01,
        help="Scaling factor for SDR luminance mapping to HDR (default: 0.01)"
    )
    parser.add_argument(
        "--plot-stats",
        action='store_true',
        help="Plot Delta E statistics across all frames after processing, or standalone if no input images provided."
    )
    parser.add_argument(
        "--only-plot",
        action='store_true',
        help="Only plot Delta E statistics without generating images. Assumes Delta E EXR files are already present in the output-dir"
    )
    
    args = parser.parse_args()
    
    if args.only_plot:
        plot_delta_e_stats(args.output_dir)
        return

    if args.input_images is None:
      if args.plot_stats:
        print("Warning: No input images specified, but --plot-stats is enabled. Only plotting will be performed.")
        plot_delta_e_stats(args.output_dir)
        return
      else:
        print("Error: No input images specified and not --plot-stats or --only-plot. Aborting.")
        return
    
    commands = generate_ocioconvert_commands(
        args.input_images,
        args.output_dir,
        args.start_frame,
        args.end_frame
    )
    
    if run_commands(commands):
        frame_outputs = {}
        for version, _, output_path in commands:
            frame_num = os.path.basename(output_path).split('.')[-2]
            parts = os.path.basename(output_path).split('_')
            view_display = '_'.join(parts[1:-2])
            key = (frame_num, view_display)
            if key not in frame_outputs:
                frame_outputs[key] = {}
            frame_outputs[key][version] = output_path
        
        for (frame_num, view_display), outputs in frame_outputs.items():
            if 'OCIO241' in outputs and 'OCIOa2o' in outputs:
                print(f"\nComparing outputs for frame {frame_num} with view {view_display}")
                compare_images(
                    outputs['OCIO241'],
                    outputs['OCIOa2o'],
                    args.output_dir,
                    mode=args.mode,
                    scaling_factor=args.scaling_factor
                )
        
        if args.plot_stats:
            plot_delta_e_stats(args.output_dir)

if __name__ == "__main__":
    main()
