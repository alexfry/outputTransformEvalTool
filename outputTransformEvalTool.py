import sys
sys.path.append('/opt/homebrew/Cellar/openimageio/2.5.16.0_2/lib/python3.12/site-packages')

import OpenImageIO as oiio
import numpy as np
import colour
import argparse

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
def write_exr_image(file_path, image_data):
    if image_data.ndim == 2:
        height, width = image_data.shape
        channels = 1
    elif image_data.ndim == 3:
        height, width, channels = image_data.shape
    else:
        raise ValueError("Invalid image data dimensions.")
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    out = oiio.ImageOutput.create(file_path)
    if not out:
        raise IOError(f"Could not create output file: {file_path}")
    out.open(file_path, spec)
    # Flatten the image data for writing
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
        linear_rgb = colour.models.eotf_BT2100_PQ(pq_rgb)
        # Convert RGB to ICtCp using Rec.2100 PQ method
        ictcp = colour.RGB_to_ICtCp(linear_rgb, method='ITU-R BT.2100-2 PQ')
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
        linear_rec2020_rgb_scaled = linear_rec2020_rgb * 100.0
        # Clip values to [0, 1] after scaling
        # linear_rec2020_rgb_clipped = np.clip(linear_rec2020_rgb_scaled, 0.0, 1.0)
        # Apply PQ OETF to simulate HDR encoding
        pq_rgb = colour.models.eotf_inverse_BT2100_PQ(linear_rec2020_rgb_scaled)
        # Convert to ICtCp
        ictcp = colour.RGB_to_ICtCp(linear_rec2020_rgb_scaled, method='ITU-R BT.2100-2 PQ')
    else:
        raise ValueError("Mode must be 'SDR' or 'HDR'")

    return ictcp, pq_rgb  # Return both ICtCp and PQ values

# Function to compute Delta E using colour.delta_E
def compute_delta_e(a, b, method='ITP', **kwargs):
    # Compute Delta E using the specified method
    delta_e = colour.delta_E(a, b, method=method, **kwargs)
    return delta_e

# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute Delta E heatmap between two EXR images using ICtCp color space.')
    parser.add_argument('image1', help='Path to the first EXR image.')
    parser.add_argument('image2', help='Path to the second EXR image.')
    parser.add_argument('-o', '--output', default='', help='Output EXR file for the Delta E heatmap.')
    parser.add_argument('-m', '--mode', choices=['SDR', 'HDR'], default='HDR', help="Mode for processing images: 'SDR' or 'HDR'.")
    parser.add_argument('-s', '--scaling_factor', type=float, default=.01, help='Scaling factor for SDR luminance mapping to HDR (default: 10).')
    parser.add_argument('--normalize', action='store_true', help='Normalize Delta E values to [0, 1] for visualization.')
    parser.add_argument('--export_pq', help='Optional output EXR file to export PQ values before ICtCp conversion.')

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

    # if heatmap_output_path is blank, append _DeltaITP to the first image path, keeping the orginal frame number if present, befpore the frame number
    if heatmap_output_path == '':
        heatmap_output_path = image1_path.split('.')
        heatmap_output_path[-3] = heatmap_output_path[-3] + '_DeltaITP'
        heatmap_output_path = '.'.join(heatmap_output_path)

    # Write the Delta E heatmap to an EXR file
    write_exr_image(heatmap_output_path, delta_e_to_write)

    print(f"Delta E heatmap saved to {heatmap_output_path}")

if __name__ == '__main__':
    main()