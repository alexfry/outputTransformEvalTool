import sys
sys.path.append('/opt/homebrew/Cellar/openimageio/3.0.3.1/lib/python3.13/site-packages')
import OpenImageIO as oiio
import numpy as np
import colour
import argparse
from PIL import Image, ImageDraw, ImageFont
import statistics
import os

def read_exr_image(file_path):
    img_input = oiio.ImageInput.open(file_path)
    if not img_input:
        raise IOError(f"Cannot open image: {file_path}")
    spec = img_input.spec()
    data = img_input.read_image(format=oiio.FLOAT)
    img_input.close()
    if data is None:
        raise IOError(f"Failed to read image data from: {file_path}")
    image = np.array(data).reshape((spec.height, spec.width, spec.nchannels))
    return image

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
    out.write_image(image_data.flatten())
    out.close()

def rgbInput_to_ictcp(rgb, mode='HDR', scaling_factor=1):
    if mode.upper() == 'HDR':
        pq_rgb = np.clip(rgb, 0.0, 1.0)
        ictcp = colour.RGB_to_ICtCp(pq_rgb, method='ITU-R BT.2100-2 PQ')
    elif mode.upper() == 'SDR':
        linear_rgb = colour.models.eotf_BT1886(rgb)
        rec709 = colour.models.RGB_COLOURSPACES['ITU-R BT.709']
        rec2020 = colour.models.RGB_COLOURSPACES['ITU-R BT.2020']
        linear_rec2020_rgb = colour.RGB_to_RGB(
            linear_rgb, rec709, rec2020, chromatic_adaptation_transform='CAT02',
            apply_cctf_decoding=False, apply_cctf_encoding=False)
        linear_rec2020_rgb_scaled = linear_rec2020_rgb * (1/scaling_factor)
        pq_rgb = colour.models.eotf_inverse_BT2100_PQ(linear_rec2020_rgb_scaled)
        ictcp = colour.RGB_to_ICtCp(pq_rgb, method='ITU-R BT.2100-2 PQ')
    else:
        raise ValueError("Mode must be 'SDR' or 'HDR'")
    return ictcp, pq_rgb

def compute_delta_e(a, b, method='ITP'):
    return colour.delta_E(a, b, method=method)

def compute_image_statistics(delta_e):
    return {
        'mean': float(np.mean(delta_e)),
        'median': float(np.median(delta_e)),
        'std_dev': float(np.std(delta_e)),
        'min': float(np.min(delta_e)),
        'max': float(np.max(delta_e)),
        'p95': float(np.percentile(delta_e, 95)),
    }

def add_stats_to_metadata(output_path, stats):
    inp = oiio.ImageInput.open(output_path)
    spec = inp.spec()
    inp.close()
    for key, value in stats.items():
        spec.attribute(f'DeltaE_{key}', value)
    img = read_exr_image(output_path)
    write_exr_image(output_path, img, spec)

def overlay_stats_text(image_data, stats, image1_name, image2_name):
    # Create a blank float mask for text overlay (same shape as image_data)
    height, width = image_data.shape
    text_mask = np.zeros((height, width), dtype=np.float32)
    
    # Create a blank 8-bit image for PIL text rendering
    img_pil = Image.fromarray(np.zeros((height, width), dtype=np.uint8), 'L')
    draw = ImageDraw.Draw(img_pil)
    
    # Define stats text, including input filenames
    stats_text = [
        f"Input 1: {image1_name}",
        f"Input 2: {image2_name}",
        f"Mean ΔE: {stats['mean']:.2f}",
        f"Median ΔE: {stats['median']:.2f}",
        f"Std Dev: {stats['std_dev']:.2f}",
        f"95th percentile: {stats['p95']:.2f}",
        f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]"
    ]
    
    # Draw text on the blank image
    try:
        font = ImageFont.truetype("Arial.ttf", 16)  # Adjust size or font as needed
    except:
        font = ImageFont.load_default()  # Fallback to default font
    
    y_position = 10
    for text in stats_text:
        draw.text((10, y_position), text, fill=255, font=font)
        y_position += 25
    
    # Convert to float mask (0 or 1 where text is)
    text_mask = (np.array(img_pil, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)  # Binary mask
    
    # Use a fixed value for text brightness
    text_value = 1.0  # Fixed value for text, as per your change
    
    # Composite: keep original delta_e where text_mask is 0, use text_value where 1
    output_data = np.where(text_mask == 1, text_value, image_data)
    
    return output_data

def compare_images(image1_path, image2_path, output_dir, mode='HDR', scaling_factor=0.01):
    image1 = read_exr_image(image1_path)
    image2 = read_exr_image(image2_path)
    
    if image1.shape != image2.shape:
        print(f"Skipping comparison: Images {image1_path} and {image2_path} have different dimensions.")
        return
    
    if image1.shape[2] < 3 or image2.shape[2] < 3:
        print(f"Skipping comparison: Images must have at least 3 channels (RGB).")
        return
    
    image1_rgb = image1[:, :, :3]
    image2_rgb = image2[:, :, :3]
    
    ictcp1, _ = rgbInput_to_ictcp(image1_rgb, mode=mode, scaling_factor=scaling_factor)
    ictcp2, _ = rgbInput_to_ictcp(image2_rgb, mode=mode, scaling_factor=scaling_factor)
    
    delta_e = compute_delta_e(ictcp1, ictcp2)
    stats = compute_image_statistics(delta_e)
    
    print(f"\nDelta E Statistics for {os.path.basename(image1_path)} vs {os.path.basename(image2_path)}:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    base_name = os.path.basename(image1_path).rsplit('_OCIO241', 1)[0]
    frame_num = os.path.basename(image1_path).split('.')[-2]
    heatmap_output_path = os.path.join(output_dir, f"{base_name}_DeltaITP.{frame_num}.exr")
    
    # Overlay text with filenames
    delta_e_with_text = overlay_stats_text(delta_e, stats, os.path.basename(image1_path), os.path.basename(image2_path))
    write_exr_image(heatmap_output_path, delta_e_with_text)
    add_stats_to_metadata(heatmap_output_path, stats)
    print(f"Delta E heatmap with text saved to {heatmap_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compute Delta E heatmap between two EXR images using ICtCp color space.')
    parser.add_argument('image1', help='Path to the first EXR image.')
    parser.add_argument('image2', help='Path to the second EXR image.')
    parser.add_argument('-o', '--output', default='', help='Output directory for the Delta E heatmap.')
    parser.add_argument('-m', '--mode', choices=['SDR', 'HDR'], default='HDR', help="Mode: 'SDR' or 'HDR'.")
    parser.add_argument('-s', '--scaling-factor', type=float, default=0.01, help='Scaling factor for SDR to HDR.')
    
    args = parser.parse_args()
    
    output_dir = args.output if args.output else os.path.dirname(args.image1)
    compare_images(args.image1, args.image2, output_dir, args.mode, args.scaling_factor)

if __name__ == '__main__':
    main()