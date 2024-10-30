Output Transform Evaluation Tool

Overview

The Output Transform Evaluation Tool (outputTransformEvalTool.py) is a Python script designed to compute a Delta E (ΔE) heatmap between two EXR images using the ICtCp color space. This tool is particularly useful for evaluating perceptual differences between two images, especially in workflows involving High Dynamic Range (HDR) and Standard Dynamic Range (SDR) content.

Features

	•	Supports HDR and SDR Images: Process images in either HDR (Rec.2020 color space) or SDR (Rec.709 color space) modes.
	•	Delta E Calculation: Computes the color difference using the colour.delta_E function with the ‘ITP’ method, which operates in the ICtCp color space.
	•	PQ Values Export: Optionally export the Perceptual Quantizer (PQ) encoded RGB values before the ICtCp conversion for verification and analysis.
	•	Luminance Scaling: Adjustable scaling factor for SDR images to map luminance levels to HDR luminance levels.
	•	Normalization: Option to normalize the Delta E values to the [0, 1] range for visualization purposes.
	•	Automatic Output Naming: If no output file is specified, the script automatically generates an output filename based on the first input image.

Dependencies

	•	Python 3.x
	•	NumPy
	•	Colour Science Library
	•	OpenImageIO
	•	argparse (part of the Python Standard Library)

Installation

1. Install Python 3.x

Ensure that Python 3.x is installed on your system. You can download it from the official website.

2. Install Required Python Packages

Install the required Python packages using pip:

pip install numpy colour-science OpenImageIO

Note: OpenImageIO might require additional steps to install, as it depends on system-specific libraries.

For macOS using Homebrew:

brew install openimageio

Then, you might need to adjust your PYTHONPATH to include the OpenImageIO Python bindings. In the script, the following line is included to adjust the path:

import sys
sys.path.append('/opt/homebrew/Cellar/openimageio/2.5.16.0_2/lib/python3.12/site-packages')

Adjust the path according to your system.

3. Clone or Download the Script

Download the outputTransformEvalTool.py script to your local machine.

Usage

python outputTransformEvalTool.py image1.exr image2.exr [options]

Positional Arguments

	•	image1: Path to the first EXR image.
	•	image2: Path to the second EXR image.

Optional Arguments

	•	-o, --output: Output EXR file for the Delta E heatmap. If not specified, the script appends _DeltaITP to the first image’s filename.
	•	-m, --mode: Mode for processing images: 'SDR' or 'HDR' (default: 'HDR').
	•	-s, --scaling_factor: Scaling factor for SDR luminance mapping to HDR (default: 0.01).
	•	--normalize: Normalize Delta E values to [0, 1] for visualization.
	•	--export_pq: Optional output EXR file to export PQ values before ICtCp conversion.

Examples

1. Basic Usage

Compute the Delta E heatmap between two HDR images:

python outputTransformEvalTool.py hdr_image1.exr hdr_image2.exr

2. Specify Output File

python outputTransformEvalTool.py image1.exr image2.exr -o output_heatmap.exr

3. Processing SDR Images

Process two SDR images with default scaling factor:

python outputTransformEvalTool.py sdr_image1.exr sdr_image2.exr -m SDR

4. Adjust Scaling Factor for SDR Images

Adjust the scaling factor to map SDR peak luminance to a different HDR peak luminance:

python outputTransformEvalTool.py sdr_image1.exr sdr_image2.exr -m SDR -s 0.02

5. Normalize Delta E Values

Normalize the Delta E values for visualization purposes:

python outputTransformEvalTool.py image1.exr image2.exr --normalize

6. Export PQ Values

Export the PQ-encoded RGB values before ICtCp conversion for verification:

python outputTransformEvalTool.py image1.exr image2.exr --export_pq pq_values.exr

Understanding the Options

Mode (-m, --mode)

	•	'HDR': Assumes that the input images are linear RGB images in the Rec.2020 color space.
	•	'SDR': Assumes that the input images are gamma-encoded RGB images in the Rec.709 color space.

Scaling Factor (-s, --scaling_factor)

	•	Used in SDR mode to map SDR luminance levels to HDR luminance levels.
	•	Default: 0.01.
	•	Note: In the script, the scaling factor is not directly used in the calculations; instead, a hardcoded value of 100.0 is used during the scaling step in SDR mode. You might need to adjust this value in the script or specify a different scaling factor when running the script.

Normalization (--normalize)

	•	When enabled, the Delta E values are normalized to the [0, 1] range.
	•	Useful for visualization when viewing the heatmap in image viewers.

Export PQ Values (--export_pq)

	•	When provided, the script will export the PQ-encoded RGB values of the first image before converting to ICtCp.
	•	Helps in verifying the intermediate PQ values for correctness.

Output File (-o, --output)

	•	If not specified, the script automatically generates an output filename by appending _DeltaITP to the base name of the first input image, preserving any existing frame numbers.

Output

	•	Delta E Heatmap: An EXR image where each pixel’s value represents the color difference between the corresponding pixels in the two input images.
	•	PQ Values (optional): An EXR image containing the PQ-encoded RGB values of the first input image.

Viewing the EXR Files

To view the output EXR files, you can use software that supports the OpenEXR format, such as:

	•	OpenImageIO’s iv tool
	•	DJV Imaging
	•	Nuke
	•	DaVinci Resolve
	•	Blender

Note: When viewing the Delta E heatmap, you might need to adjust the exposure or use false color mapping to interpret the values correctly.

Troubleshooting

	•	Import Errors: Ensure all dependencies are installed and that your PYTHONPATH includes the paths to the installed packages.
	•	Version Compatibility: The script is compatible with colour-science library version 0.4.3 and later.
	•	Adjusting Paths: If you have installed libraries in non-standard locations, you might need to adjust sys.path in the script.

Additional Notes

	•	Scaling Factor in SDR Mode: The default scaling factor is 0.01, but in the code, the scaling factor used during the luminance scaling step in SDR mode is hardcoded as 100.0. You may need to adjust this value in the script directly or ensure consistency between the --scaling_factor argument and the code.

# In the script
linear_rec2020_rgb_scaled = linear_rec2020_rgb * 100.0


	•	HDR Mode Processing: In HDR mode, the script assumes the input images are in linear RGB in the Rec.2020 color space and uses them directly without applying PQ encoding before converting to ICtCp. Ensure your HDR images are formatted accordingly.
	•	SDR to HDR Conversion: In SDR mode, the script:
	1.	Applies the inverse BT.1886 EOTF to obtain linear RGB.
	2.	Converts linear RGB from Rec.709 to Rec.2020 color space.
	3.	Scales the luminance (currently by 100.0).
	4.	Applies the inverse BT.2100 PQ EOTF to simulate HDR encoding.
	5.	Converts to ICtCp color space.
	•	Output Filename Handling: If the output filename is not specified (-o option), the script generates an output filename by appending _DeltaITP to the base name of the first input image, attempting to preserve any frame numbers.
	•	Modifying the Script: For customization, such as changing the scaling factor in the SDR processing pipeline or adjusting the HDR assumptions, you may need to modify the script directly.

Contributing

If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

License

This project is licensed under the MIT License.

Acknowledgments

	•	Colour Science Library: For providing comprehensive tools for colorimetry and color science.
	•	OpenImageIO: For image input/output support.

Let me know if you need any further assistance or modifications!