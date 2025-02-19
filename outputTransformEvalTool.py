import sys
sys.path.append('/opt/homebrew/Cellar/openimageio/3.0.3.1/lib/python3.13/site-packages')

import OpenImageIO as oiio
import numpy as np
import colour
import argparse
from PIL import Image, ImageDraw, ImageFont
import statistics
import matplotlib.pyplot as plt
import os

# Function to read EXR image using OpenImageIO
def read_exr_image(file_path):
    img_input = oiio.ImageInput.open(file_path)
    if not img_input:
        raise IOError(f"Cannot open image: {file_path}")
    spec = img_input.spec()
    width = spec.width
    height = spec.height
    
    channels = spec.nchannels
    # Read the image data as a flat array
    data = img_input.read_image(format=oiio.FLOAT)
    img_input.close()
    if data is None:
        raise IOError(f"Failed to read image data from: {file_path}")
    # Convert data to a NumPy array and reshape
    image = np.array(data)
    image = image.reshape((height, width, channels))
    return image

# Function to write EXR image using OpenImageIO
def write_exr_image(file_path, image_data, spec=None):
    if image_data.ndim == 2:
        height, width = image_data.shape
        channels = 1
    elif image_data.ndim == 3:
        height, width, channels = image_data.shape
    else:
        raise ValueError("Invalid image data dimensions.")
    
    if spec is None:
        spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    
    out = oiio.ImageOutput.create(file_path)
    if not out:
        raise IOError(f"Could not create output file: {file_path}")
    out.open(file_path, spec)
    image_data_flat = image_data.flatten()
    out.write_image(image_data_flat)
    out.close()

# Function to convert linear RGB to ICtCp and return PQ values
def rgbInput_to_ictcp(rgb, mode='HDR', scaling_factor=1):
    if mode.upper() == 'HDR':
        # Ensure the RGB values are within [0, 1]
        pq_rgb = np.clip(rgb, 0.0, 1.0)
        # pq_rgb = pq
        # Invert the Perceptual Quantizer (PQ) OETF
        # linear_rgb = colour.models.eotf_BT2100_PQ(pq_rgb)
        # Convert RGB to ICtCp using Rec.2100 PQ method
        ictcp = colour.RGB_to_ICtCp(pq_rgb, method='ITU-R BT.2100-2 PQ')
    elif mode.upper() == 'SDR':
        # Undo gamma 2.4 encoding to get linear RGB
        linear_rgb = colour.models.eotf_BT1886(rgb)
        # Define source and target colourspaces
        rec709 = colour.models.RGB_COLOURSPACES['ITU-R BT.709']
        rec2020 = colour.models.RGB_COLOURSPACES['ITU-R BT.2020']
        # Convert linear RGB from Rec.709 to Rec.2020
        linear_rec2020_rgb = colour.RGB_to_RGB(
            linear_rgb,
            input_colourspace=rec709,
            output_colourspace=rec2020,
            chromatic_adaptation_transform='CAT02',
            apply_cctf_decoding=False,
            apply_cctf_encoding=False)
        # Apply scaling factor to map SDR luminance to HDR luminance
        linear_rec2020_rgb_scaled = linear_rec2020_rgb * (1/scaling_factor)
        # Clip values to [0, 1] after scaling
        # linear_rec2020_rgb_clipped = np.clip(linear_rec2020_rgb_scaled, 0.0, 1.0)
        # Apply PQ OETF to simulate HDR encoding
        pq_rgb = colour.models.eotf_inverse_BT2100_PQ(linear_rec2020_rgb_scaled)
        # Convert to ICtCp
        ictcp = colour.RGB_to_ICtCp(pq_rgb, method='ITU-R BT.2100-2 PQ')
    else:
        raise ValueError("Mode must be 'SDR' or 'HDR'")

    return ictcp, pq_rgb  # Return both ICtCp and PQ values

# Function to compute Delta E using colour.delta_E
def compute_delta_e(a, b, method='ITP', **kwargs):
    # Compute Delta E using the specified method
    delta_e = colour.delta_E(a, b, method=method, **kwargs)
    return delta_e

def compute_image_statistics(delta_e):
    """Compute statistical metrics for the Delta E image."""
    stats = {
        'mean': float(np.mean(delta_e)),
        'median': float(np.median(delta_e)),
        'std_dev': float(np.std(delta_e)),
        'min': float(np.min(delta_e)),
        'max': float(np.max(delta_e)),
        'p95': float(np.percentile(delta_e, 95)),  # 95th percentile
    }
    return stats

def add_stats_to_metadata(output_path, stats):
    """Add statistics to EXR metadata."""
    inp = oiio.ImageInput.open(output_path)
    spec = inp.spec()
    inp.close()
    
    # Add stats to metadata
    for key, value in stats.items():
        spec.attribute(f'DeltaE_{key}', value)
    
    # Read existing image
    img = read_exr_image(output_path)
    
    # Write back with new metadata
    write_exr_image(output_path, img, spec)

def overlay_stats_text(image_data, stats):
    """Overlay statistics on the image."""
    # Convert to 8-bit for PIL
    img_8bit = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
    
    # Create PIL image
    if len(img_8bit.shape) == 2:
        img_pil = Image.fromarray(img_8bit, 'L')
    else:
        img_pil = Image.fromarray(img_8bit[:,:,0], 'L')
    
    # Create drawing context
    draw = ImageDraw.Draw(img_pil)
    
    # Prepare text
    stats_text = [
        f"Mean ΔE: {stats['mean']:.2f}",
        f"Median ΔE: {stats['median']:.2f}",
        f"Std Dev: {stats['std_dev']:.2f}",
        f"95th percentile: {stats['p95']:.2f}",
        f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]"
    ]
    
    # Position text in top-left corner
    y_position = 10
    for text in stats_text:
        draw.text((10, y_position), text, fill=255)
        y_position += 20
    
    # Convert back to float32
    return np.array(img_pil, dtype=np.float32) / 255.0

def display_delta_e(delta_e, stats, title="Delta E Heatmap", image1_path="", image2_path=""):
    """Display the Delta E heatmap with a colorbar and statistics."""
    # Set the background color
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#373737')
    ax = plt.gca()
    ax.set_facecolor('#373737')
    
    # Create heatmap
    im = plt.imshow(delta_e, cmap='viridis')
    plt.colorbar(im, label='Delta E')
    
    # Add title and input filenames
    plt.title(title)
    if image1_path and image2_path:
        file1 = os.path.basename(image1_path)
        file2 = os.path.basename(image2_path)
        plt.figtext(0.02, 0.02, f"Input 1: {file1}\nInput 2: {file2}", 
                   color='white', fontsize=8, va='bottom')
    
    # Add statistics text
    stats_text = (
        f"Mean ΔE: {stats['mean']:.2f}\n"
        f"Median ΔE: {stats['median']:.2f}\n"
        f"Std Dev: {stats['std_dev']:.2f}\n"
        f"95th percentile: {stats['p95']:.2f}\n"
        f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]"
    )
    plt.text(1.2, 0.5, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='#373737', alpha=0.8), color='white')
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute Delta E heatmap between two EXR images using ICtCp color space.')
    parser.add_argument('image1', help='Path to the first EXR image.')
    parser.add_argument('image2', help='Path to the second EXR image.')
    parser.add_argument('-o', '--output', default='', help='Output EXR file for the Delta E heatmap.')
    parser.add_argument('-m', '--mode', choices=['SDR', 'HDR'], default='HDR', help="Mode for processing images: 'SDR' or 'HDR'.")
    parser.add_argument('-s', '--scaling_factor', type=float, default=.01, help='Scaling factor for SDR luminance mapping to HDR (default: 0.01).')
    parser.add_argument('--normalize', action='store_true', help='Normalize Delta E values to [0, 1] for visualization.')
    parser.add_argument('--export_pq', help='Optional output EXR file to export PQ values before ICtCp conversion.')
    parser.add_argument('--display', action='store_true', help='Display the Delta E heatmap instead of saving to file.')

    args = parser.parse_args()

    image1_path = args.image1
    image2_path = args.image2
    heatmap_output_path = args.output
    mode = args.mode
    scaling_factor = args.scaling_factor
    normalize = args.normalize
    pq_export_path = args.export_pq

    print(f"Processing images in {mode} mode.")
    if mode.upper() == 'SDR':
        print(f"Using scaling factor: {scaling_factor}")

    # Read the images
    image1 = read_exr_image(image1_path)
    image2 = read_exr_image(image2_path)

    # Check if images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images have different dimensions.")

    # Ensure the images have at least 3 channels (RGB)
    if image1.shape[2] < 3 or image2.shape[2] < 3:
        raise ValueError("Images must have at least 3 channels (RGB).")

    # Extract the first three channels (assuming RGB)
    image1_rgb = image1[:, :, :3]
    image2_rgb = image2[:, :, :3]

    # Convert images to ICtCp color space and get PQ values
    ictcp1, pq_rgb1 = rgbInput_to_ictcp(image1_rgb, mode=mode, scaling_factor=scaling_factor)
    ictcp2, pq_rgb2 = rgbInput_to_ictcp(image2_rgb, mode=mode, scaling_factor=scaling_factor)

    # Optional: Export PQ values before ICtCp conversion
    if pq_export_path:
        # Export PQ values for the first image
        write_exr_image(pq_export_path, pq_rgb1)
        print(f"PQ values exported to {pq_export_path}")

    # Compute Delta E per pixel
    delta_e = compute_delta_e(ictcp1, ictcp2)

    if normalize:
        # Normalize Delta E values to [0, 1] for visualization
        delta_e_min = delta_e.min()
        delta_e_max = delta_e.max()
        delta_e_normalized = (delta_e - delta_e_min) / (delta_e_max - delta_e_min + 1e-8)
        delta_e_to_write = delta_e_normalized
        print(f"Delta E values normalized to range [0, 1].")
    else:
        delta_e_to_write = delta_e

    # Compute statistics
    stats = compute_image_statistics(delta_e)
    
    # Print statistics to console
    print("\nDelta E Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Handle display or save based on user choice
    if args.display:
        display_delta_e(delta_e_to_write, stats, 
                       image1_path=image1_path, 
                       image2_path=image2_path)
    
    if not args.display or args.output:  # Save if display is not requested or output is explicitly specified
        # Add overlay text to the image
        delta_e_with_text = overlay_stats_text(delta_e_to_write, stats)
        
        # if heatmap_output_path is blank, append _DeltaITP to the first image path
        if heatmap_output_path == '':
            parts = image1_path.rsplit('.', 2)
            if len(parts) == 3:
                base_path, frame_num, extension = parts
                heatmap_output_path = f"{base_path}_DeltaITP.{frame_num}.{extension}"
            else:
                base_path = image1_path.rsplit('.', 1)[0]
                extension = image1_path.split('.')[-1]
                heatmap_output_path = f"{base_path}_DeltaITP.{extension}"

        # Write the Delta E heatmap with overlay
        write_exr_image(heatmap_output_path, delta_e_with_text)
        
        # Add statistics to metadata
        add_stats_to_metadata(heatmap_output_path, stats)
        
        print(f"Delta E heatmap saved to {heatmap_output_path}")

if __name__ == '__main__':
    main()