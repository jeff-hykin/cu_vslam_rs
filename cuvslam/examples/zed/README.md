# PyCuVSLAM with ZED Stereo Camera

In this folder, you will find two tutorials:

1. [live](live/README.md): Shows how to perform live PyCuVSLAM tracking using stereo images and depth data from a ZED
   stereo camera.
2. [recording](recording/README.md): Shows how to perform ZED recording and later offline stereo tracking with rerun
   visualisation.

> **USB Camera Troubleshooting:**  occasionally, the USB camera may fail to start. If this occurs, briefly disconnect
> and reconnect it to the USB port

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed environment setup instructions

## ZED SDK Installation

ZED SDK `5.4.0` is recommended for the Python examples in this repository. The shared
[`examples/requirements.txt`](../requirements.txt) pins `numpy==2.2.4`; the `pyzed` wheel shipped with ZED SDK `5.4`
supports NumPy 2.x. Some older `pyzed` wheels, such as the one shipped with ZED SDK `4.1`, require `numpy<2.0`, which
conflicts with the shared example requirements and can break SciPy/Rerun imports.

### ZED C++ SDK and tools

Follow the official installation documentation for
[Linux x86](https://www.stereolabs.com/docs/development/zed-sdk/linux#download-and-install-the-zed-sdk)
or [Jetpack](https://www.stereolabs.com/docs/development/zed-sdk/jetson).

During setup, it's recommended to skip the ZED Python API installation.

When prompted:

```
Do you want to install the Python API (recommended) [Y/n] ?
```

press `n`.

You can then manually install `pyzed` in your virtual environment.
This avoids having potentially incompatible versions of `cython` and `numpy` installed globally for your user.

### ZED Python SDK

> **Note:**  The official StereoLabs documentation
> [Installing the Python API](https://www.stereolabs.com/docs/development/python/install#installing-the-python-api)

Run `get_python_api.py` from the `/usr/local/zed` folder inside your
[Virtual environment](../README.md#using-venv). It will download pyzed*.whl and
install it with `pip` for your active environment only.

```
pip install requests==2.32.5
python3 /usr/local/zed/get_python_api.py
pip install -r ../requirements.txt
pip check
```

Reinstalling `../requirements.txt` after `pyzed` keeps the shared example versions pinned, and `pip check` verifies that
the installed ZED Python API is compatible with those versions.
