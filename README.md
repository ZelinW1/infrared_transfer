# Infrared Camera Style Transfer Tool

This is a simple tool built with Python and OpenCV to extract the unique sensor "fingerprint" (including vignetting and fixed-pattern noise) from a series of images taken by a specific infrared camera. It then applies this fingerprint to other "clean" images, performing a style transfer that makes them appear as if they were captured by that same camera.

The project is based on a straightforward and effective statistical method: averaging a large number of images to cancel out scene-specific content, thereby isolating the inherent, unchanging patterns of the sensor.

**Read this in other languages:**
- [English](README.md)
- [简体中文](README.zh-CN.md)

## ✨ Features

- **Fingerprint Extraction**: Automatically analyzes all images in a specified directory to extract the camera's vignetting map (low-frequency) and noise map (high-frequency).
- **Style Application**: Applies the extracted camera fingerprint to any clean image.
- **Configuration-Driven**: All paths and algorithm parameters are managed through a `config.yaml` file, requiring no code modification for tuning.
- **Modular Design**: The codebase is clearly structured into two main modules—analysis and application—making it easy to understand and extend.
- **Command-Line Interface**: A simple entry point, `main.py`, controls the entire workflow through command-line arguments.

## 🔧 Project Structure

```
infrared-style-transfer/
├── data/
│   ├── raw_images/       # Source images from the target IR camera for analysis
│   └── clean_images/     # Clean images to apply the style to
├── output/
│   ├── camera_fingerprint/ # Stores the extracted camera fingerprint
│   └── stylized_images/    # Stores the final stylized images
├── src/
│   ├── camera_analyzer.py    # Core Module 1: Fingerprint Extraction
│   └── style_applicator.py   # Core Module 2: Style Application
├── main.py                 # Main execution script for the project
├── config.yaml             # Configuration file
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
```

## 🚀 Quick Start

### 1. Setup

First, clone or download this project and ensure you have Python 3.x installed.

Then, from the project's root directory, install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

- **Raw Images**: Place a large number of diverse images (50+ recommended) taken by the target infrared camera into the `data/raw_images/` directory.
- **Clean Images**: Place the "clean" images you want to apply the camera style to into the `data/clean_images/` directory.

### 3. (Optional) Configure Parameters

Open the `config.yaml` file. Here, you can adjust parameters such as the processing resolution, Gaussian blur kernel size, and noise intensity. The default settings are a good starting point for most use cases.

### 4. Run the Tool

Open a terminal in the project's root directory and execute the following commands:

- **Step 1: Extract the Camera Fingerprint**
  ```bash
  python main.py extract
  ```
  This command will analyze the images in `data/raw_images/` and save the resulting `vignetting_map.exr` and `noise_map.exr` to the `output/camera_fingerprint/` directory.

- **Step 2: Apply the Style**
  ```bash
  python main.py apply
  ```
  This command will load the previously extracted fingerprint, process all images in `data/clean_images/`, and save the results to `output/stylized_images/`.

- **All-in-One: Run the Full Pipeline**
  ```bash
  python main.py all
  ```
  This command executes both the extraction and application steps sequentially.

## 🛠️ How It Works

The tool is based on a core assumption: `Image = Scene Content + Fixed Sensor Pattern`.

By averaging a large set of images with varying scenes, the random "Scene Content" cancels out, converging towards a smooth mean. The "Fixed Sensor Pattern," which includes artifacts like Photo-Response Non-Uniformity (PRNU) and lens vignetting, remains constant across all images and is therefore reinforced through averaging. Using a Gaussian blur and image subtraction, we can further separate this pattern into its low-frequency (vignetting) and high-frequency (noise) components, effectively creating the camera's "fingerprint."