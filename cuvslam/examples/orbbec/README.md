# Tutorial: Running PyCuVSLAM Visual Odometry on Orbbec Stereo Camera

This tutorial demonstrates how to perform live PyCuVSLAM tracking using stereo images and depth data from a Orbbec stereo
camera

> **Notes:**
> * These scripts have been developed and validated on the Orbbec 335L stereo camera.
> * Some Orbbec camera models, such as the Orbbec 336, use a rolling shutter RGB sensor. Rolling shutters may lead to poor visual feature tracking in PyCuVSLAM. For best results, choose a camera with a global shutter sensor.

## Setting Up the cuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for instructions on installing and configuring all required dependencies

## Setting Up OrbbecSDK

Install the [OrbbecSDK V2 Python wrapper](https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html#online-installation) following the official documentation.

> **Note:** Ensure that all system dependencies and udev rules are configured according to the [Orbbec official guide](https://github.com/orbbec/OrbbecSDK#environment-setup).

## Running Stereo Visual Odometry

To start stereo visual odometry, run:

```bash
python3 run_stereo.py
```

After starting, you should see a visualization similar to the following:
![Visualization Example](../assets/tutorial_orbbec_stereo.gif)

## Running Monocular-Depth Visual Odometry

Monocular-Depth Visual Odometry requires pixel-to-pixel correspondence between camera and depth images. For Orbbec cameras, please ensure depth images are aligned with the RGB camera.

We recommend enabling the Orbbec infrared emitter for improved depth image quality. Note that when the emitter is enabled, infrared images from stereo cameras will contain artificial features from the emitter pattern. Therefore, avoid using infrared images for visual tracking in this mode. Instead, use the RGB camera as your visual data source.

To run monocular-depth visual odometry, execute:

```bash
python3 run_rgbd.py
```

You should see the following rerun visualization, displaying the camera trajectory along with the input RGB and depth images:

![Visualization Example](../assets/tutorial_orbbec_rgbd.gif)

> **Note**: Mono-Depth mode in PyCuVSLAM requires significantly more computational resources due to dense feature processing in the pipeline. For more details on recommended operational modes and resolutions for Jetson devices, refer to the [Hardware Scalability section](https://arxiv.org/html/2506.04359v3#A1.F13) of the PyCuVSLAM technical report.
