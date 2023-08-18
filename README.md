# Image Super-Resolution

## Overview
Single-image Super-resolution entails the fascinating task of enhancing low-resolution images into higher-resolution versions, all without sacrificing intricate details or introducing unwanted artifacts that conventional methods (e.g., bicubic or bilinear interpolation) might induce. This process of augmenting image quality from low-resolution (LR) to high-resolution (HR) is paramount in numerous applications, including security cameras and surveillance. If you're curious, you can experience a live web demo of the cutting-edge super-resolution model [here](https://replicate.com/xpixelgroup/hat).

## Dataset
The first step in this journey is obtaining the necessary dataset. The model's training relies on the DIVerse 2K resolution dataset, a collection of high-quality images that includes both high and low-resolution versions. You can download these images directly from the [source](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or utilize the provided shell script `get_data.sh` located in the "script" folder to expedite the process.

To run a `.sh` file (shell script) on Windows, you can make use of the Windows Subsystem for Linux (WSL): `./script/get_data.sh`

Additionally, image evaluation is performed using the PSNR (Peak Signal-to-Noise Ratio) metric, a measure of image fidelity. More details about the PSNR metric and its code can be found [here](https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py).

## Additional Resources

For an in-depth exploration of the project, complete with illuminating visuals and informative images, refer to the attached PowerPoint presentation:
- [Image_Super_Resolution](Image_Super_Resolution.pptx)
