import os
import subprocess
import glob
import argparse
import re
from outputTransformEvalTool import compare_images
import matplotlib.pyplot as plt
import numpy as np
from outputTransformEvalTool import read_exr_image, compute_image_statistics
from matplotlib.widgets import Button

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
    
    display_names = [pair[1] for pair in ViewPairs]
    display_str_map = {display.replace("2.0", "20").replace(" ", "_").replace(".", "").replace("-", "_").replace("(", "").replace(")", "").replace("P3 D65", "P3D65").replace("__", "_"): display for display in display_names}
    print(f"Display map: {display_str_map}")  # Debug the map
    
    glob_pattern = input_images.replace("#", "[0-9]*")
    input_files = glob.glob(glob_pattern)
    
    os.makedirs(output_dir, exist_ok=True)
    
    commands = []
    frame_pattern = r'^(.*?)\.(\d{4})\.exr$'  # Match 4-digit frame numbers
    
    for input_file in input_files:
        match = re.search(frame_pattern, os.path.basename(input_file))
        if not match:
            print(f"Skipping unmatched input file: {input_file}")
            continue
        base_name, frame_num = match.groups()
        frame_num = int(frame_num)
        if frame_num < start_frame or frame_num > end_frame:
            continue
        frame_str = f"{frame_num:04d}"
        
        for view, display in ViewPairs:
            view_str = view.replace(" ", "_").replace(".", "").replace("-", "_")
            display_str = display.replace("2.0", "20").replace(" ", "_").replace(".", "").replace("-", "_").replace("(", "").replace(")", "").replace("P3 D65", "P3D65").replace("__", "_")
            output_241 = os.path.join(output_dir, f"{base_name}_{view_str}_{display_str}_OCIO241_DeltaITP.{frame_str}.exr")
            output_a2o = os.path.join(output_dir, f"{base_name}_{view_str}_{display_str}_OCIOa2o_DeltaITP.{frame_str}.exr")
            VIEW_PARAMS = ["--view", input_file, "aces", output_241, view, display]
            cmd_241 = [OCIO_ORIGINAL_PATH] + VIEW_PARAMS + ["--string-attribute", f"OCIO_Binary={OCIO_ORIGINAL_PATH}"]
            VIEW_PARAMS = ["--view", input_file, "aces", output_a2o, view, display]
            cmd_a2o = [OCIO_MODIFIED_PATH] + VIEW_PARAMS + ["--string-attribute", f"OCIO_Binary={OCIO_MODIFIED_PATH}"]
            commands.append(("OCIO241", cmd_241, output_241, OCIO_ORIGINAL_PATH))
            commands.append(("OCIOa2o", cmd_a2o, output_a2o, OCIO_MODIFIED_PATH))
    
    return commands, display_names

def run_commands(commands):
    env = os.environ.copy()
    env["OCIO"] = "/Users/afry/Downloads/studio-config-aces-v1-and-v2.ocio"
    for version, cmd, output_path, binary_path in commands:
        print(f"Running {version}: {' '.join(cmd)} (Binary: {binary_path})")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {version} command: {e}")
            return False
    return True

def plot_delta_e_stats(output_dir, display_names, ViewPairs, start_frame=0, end_frame=9999):
    delta_e_files = glob.glob(os.path.join(output_dir, "*_DeltaITP.*.exr"))
    if not delta_e_files:
        print("No Delta E images found in the output directory.")
        return
    
    stats_by_display = {}
    frame_pattern = r'^(.*?_.*?)_DeltaITP\.(\d{4})\.exr$'  # Match 4-digit frame numbers
    
    # Use the same display_str_map as generated
    display_str_map = {display.replace("2.0", "20").replace(" ", "_").replace(".", "").replace("-", "_").replace("(", "").replace(")", "").replace("P3 D65", "P3D65").replace("__", "_"): display for display in display_names}
    
    for file_path in delta_e_files:
        match = re.search(frame_pattern, os.path.basename(file_path))
        if not match:
            print(f"Skipping file with unmatched pattern: {file_path}")
            continue
        base_name, frame_num_str = match.groups()
        frame_num = int(frame_num_str)
        print(f"Detected frame: {frame_num} from {file_path}")  # Debug frame number
        if frame_num < start_frame or frame_num > end_frame:
            print(f"Skipping frame {frame_num} outside range [{start_frame}, {end_frame}]")
            continue
        
        parts = base_name.split('_')
        print(f"Base name parts: {parts}")  # Debug filename parts
        display_name = "Unknown"
        # Look for the display_str part after "Display"
        display_idx = -1
        for i, part in enumerate(parts):
            if part == "Display" or part == "display":
                display_idx = i + 1
                break
        if display_idx >= 0 and display_idx < len(parts):
            display_str = "_".join(parts[display_idx:]).split("_OCIO")[0].strip("_")
            cleaned_display_str = display_str.replace("P3_D65", "P3D65").replace("2.0", "20").replace(".", "").replace("-", "_").replace("__", "_")
            print(f"Extracted display_str: {display_str}, cleaned: {cleaned_display_str}, map keys: {list(display_str_map.keys())}")
            for key, orig_display in display_str_map.items():
                if key == cleaned_display_str:
                    display_name = orig_display
                    print(f"Matched display: {display_name} for display_str {display_str} (cleaned: {cleaned_display_str})")
                    break
                else:
                    print(f"Key {key} does not match cleaned {cleaned_display_str}")
            # Fallback: Partial match on display type if exact match fails
            if display_name == "Unknown":
                for key, orig_display in display_str_map.items():
                    if any(part in key for part in ["SDR_100_nits", "HDR_500_nits", "HDR_1000_nits", "HDR_2000_nits", "HDR_4000_nits"] if part in cleaned_display_str):
                        display_name = orig_display
                        print(f"Fallback matched display: {display_name} for display_str {display_str} (cleaned: {cleaned_display_str})")
                        break
        
        if display_name not in stats_by_display:
            stats_by_display[display_name] = {}
        delta_e_data = read_exr_image(file_path)
        stats = compute_image_statistics(delta_e_data)
        stats_by_display[display_name][frame_num] = stats
    
    # Reorder displays to match ViewPairs order, move "Unknown" to end
    ordered_displays = [d for d in [pair[1] for pair in ViewPairs] if d in stats_by_display] + [d for d in stats_by_display if d not in [pair[1] for pair in ViewPairs]]
    stats_by_display = {d: stats_by_display[d] for d in ordered_displays}
    
    # Use a single large plot if only one display, otherwise adjust layout
    n_displays = len(stats_by_display)
    print(f"Detected displays: {list(stats_by_display.keys())}")  # Debug all detected displays
    if n_displays == 1:
        fig, ax = plt.subplots(figsize=(20, 9))  # Reduced height by 10% (from 10 to 9)
    else:
        fig, ax = plt.subplots(figsize=(20, 9))  # Reduced height by 10% (from 10 to 9)
    fig.patch.set_facecolor('#373737')
    plt.style.use('dark_background')
    
    metrics = ['mean', 'median', 'max', 'p95']
    colors = ['cyan', 'magenta', 'yellow', 'green']
    line_styles = ['-', '--', '-.', ':']
    
    max_value = 0
    for frame_stats in stats_by_display.values():
        for stats in frame_stats.values():
            max_value = max(max_value, stats['max'])
    
    current_display_idx = 0
    def update_plot(display_idx):
        nonlocal current_display_idx, ax
        current_display_idx = display_idx % n_displays
        ax.clear()
        display_name = list(stats_by_display.keys())[current_display_idx]
        frames = sorted([f for f in stats_by_display[display_name].keys() if start_frame <= f <= end_frame])
        if not frames:
            ax.text(0.5, 0.5, "No data in range", transform=ax.transAxes, ha='center')
        else:
            print(f"Plotting frames for {display_name}: {frames} with values {[stats_by_display[display_name][f]['mean'] for f in frames]}")
            for metric, color, style in zip(metrics, colors, line_styles):
                values = [stats_by_display[display_name][frame][metric] for frame in frames]
                ax.plot(frames, values, label=metric, color=color, linestyle=style, marker='o')
            ax.set_title(f"Display: {display_name}", pad=15)  # Add padding to title
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Delta E Value")
            ax.set_xlim(frames[0] - 0.5, frames[-1] + 0.5)
            ax.set_ylim(0, max_value * 1.1 if max_value > 0 else 1)
            ax.set_yticks(np.arange(0, max_value * 1.1 + 1, 1.0))  # Set Y-axis ticks every 1.0
            ax.set_xticks(frames)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#444444')  # Dark grey background
            ax.tick_params(colors='white')  # White axis markers
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)
        fig.canvas.draw()

    # Initial plot
    update_plot(0)
    
    # Add navigation buttons if multiple displays, adjust position to avoid overlap
    if n_displays > 1:
        ax_prev = plt.axes([0.7, 0.00, 0.1, 0.075])  # Moved to midpoint (0.00) between -0.05 and 0.05
        ax_next = plt.axes([0.81, 0.00, 0.1, 0.075])  # Moved to midpoint (0.00) between -0.05 and 0.05
        b_prev = Button(ax_prev, 'Previous', color='grey', hovercolor='darkgrey')
        b_next = Button(ax_next, 'Next', color='grey', hovercolor='darkgrey')
        b_prev.label.set_color('white')  # Set text color to white
        b_next.label.set_color('white')  # Set text color to white
        
        def next_display(event):
            update_plot(current_display_idx + 1)
        def prev_display(event):
            update_plot(current_display_idx - 1)
        
        b_next.on_clicked(next_display)
        b_prev.on_clicked(prev_display)
    
    plt.tight_layout(pad=5.0)  # Increased padding to ensure space for buttons
    plt.suptitle("Delta E Statistics Across Frames by Display Type", fontsize=16, color='white', y=1.05)  # Shift supertitle up
    plt.gcf().set_size_inches(20, 9, forward=True)  # Enforce reduced size to fit screen
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare two OCIO pipelines and analyze output differences")
    parser.add_argument(
        "--input-images",
        required=True,
        help="Input file pattern with # for frame number (e.g., '/path/to/files.#.exr')"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed files"
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
        help="Plot Delta E statistics across all frames after processing"
    )
    parser.add_argument(
        "--only-plot",
        action='store_true',
        help="Only plot Delta E statistics from existing files without reprocessing"
    )
    
    args = parser.parse_args()
    
    ViewPairs = [
        ["Rec.2100-PQ - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 500 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 2000 nits (P3 D65)"],
        ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (P3 D65)"],
    ]
    
    if args.only_plot:
        plot_delta_e_stats(args.output_dir, [pair[1] for pair in ViewPairs], ViewPairs, args.start_frame, args.end_frame)
        return
    
    commands, display_names = generate_ocioconvert_commands(
        args.input_images,
        args.output_dir,
        args.start_frame,
        args.end_frame
    )
    
    if run_commands(commands):
        frame_outputs = {}
        for version, _, output_path, binary_path in commands:
            frame_num = os.path.basename(output_path).split('.')[-2]
            parts = os.path.basename(output_path).split('_')
            view_display = '_'.join(parts[1:-2])
            key = (frame_num, view_display)
            if key not in frame_outputs:
                frame_outputs[key] = {}
            frame_outputs[key][version] = (output_path, binary_path)
        
        for (frame_num, view_display), outputs in frame_outputs.items():
            if 'OCIO241' in outputs and 'OCIOa2o' in outputs:
                print(f"\nComparing outputs for frame {frame_num} with view {view_display}")
                compare_images(
                    outputs['OCIO241'][0],
                    outputs['OCIOa2o'][0],
                    args.output_dir,
                    mode=args.mode,
                    scaling_factor=args.scaling_factor,
                    ocio_241_path=outputs['OCIO241'][1],
                    ocio_a2o_path=outputs['OCIOa2o'][1]
                )
        
        if args.plot_stats:
            plot_delta_e_stats(args.output_dir, display_names, ViewPairs, args.start_frame, args.end_frame)

if __name__ == "__main__":
    main()