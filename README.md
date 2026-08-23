# cu_vslam_rs

Rust FFI for [NVIDIA cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) visual odometry: raw bindings in `cu_vslam_rs::ffi` plus a safe `Tracker` wrapper. The repository also vendors a cuVSLAM v17.0.0 fork (`cuvslam/`) and a nix flake that packages a cuVSLAM SDK per platform — including **macOS on Apple silicon via [CuMetal](https://github.com/jeff-hykin/cuda-metal)**.

## Layout

| Path | What it is | License |
|---|---|---|
| `src/`, `shim/`, `build.rs` | The `cu_vslam_rs` crate: extern-C shim over `cuvslam2.h`, raw `ffi` bindings, safe `Tracker` | Apache-2.0 |
| `cuvslam/` | Fork of nvidia-isaac/cuVSLAM v17.0.0 with odometry-state serialization, macOS/CuMetal build support, and a deterministic VO RANSAC seed | NVIDIA Community License (see `cuvslam/LICENSE`) |
| `flake.nix` | Per-platform SDK packages + the crate | — |

Only the crate (`src/`, `shim/`, `build.rs`) is published to crates.io; the vendored fork and flake are repo-only.

## Using the crate

```toml
[dependencies]
cu_vslam_rs = "0.1"
```

Building needs `CUVSLAM_SDK_DIR` pointing at an SDK layout:

```text
$CUVSLAM_SDK_DIR/
  include/cuvslam/cuvslam2.h
  lib/libcuvslam.so        # or .dylib on macOS
```

`build.rs` compiles the shim against `include/`, links `libcuvslam`, and embeds `lib/` as an rpath.

```rust
use cu_vslam_rs::{CameraParams, ImageRef, Tracker, ffi};

let mut tracker = Tracker::new(&cameras, imu_calibration.as_ref(), &config)?;
let estimate = tracker.track(&images, &depths)?;
if let Some((world_from_rig, covariance)) = estimate.world_from_rig {
    // pose is valid; None while tracking is lost
}
```

## Getting an SDK via the flake

```sh
# The default variant for your machine (metal on Apple silicon, cuda12 on x86_64-linux):
nix build github:jeff-hykin/cu_vslam_rs
export CUVSLAM_SDK_DIR=$PWD/result

# Or a specific variant:
nix build github:jeff-hykin/cu_vslam_rs#sdk-x86_64-cuda12   # built from cuvslam/, sm_89 + sm_120
nix build github:jeff-hykin/cu_vslam_rs#sdk-x86_64-cuda13   # NVIDIA prebuilt tarball
nix build github:jeff-hykin/cu_vslam_rs#sdk-orin            # built from cuvslam/, sm_87
nix build github:jeff-hykin/cu_vslam_rs#sdk-thor            # NVIDIA prebuilt tarball
nix build github:jeff-hykin/cu_vslam_rs#sdk-metal           # macOS Apple silicon, CuMetal

# Dev shell with CUVSLAM_SDK_DIR already set:
nix develop github:jeff-hykin/cu_vslam_rs
cargo build
```

Flakes that consume this one can take it as an input and use `cu-vslam-rs.packages.${system}."sdk-<variant>"` as `CUVSLAM_SDK_DIR` for their own Rust builds.

`x86_64-cuda12` and `orin` are compiled from `cuvslam/` in a sandboxed nix build (`ENFORCE_GPU=OFF`, so CPU/GPU is a runtime switch). `thor` uses NVIDIA's prebuilt tarball. `metal` uses a CuMetal build of `cuvslam/` hosted as a release artifact (`ENFORCE_GPU=ON`; the Metal path is the only backend).

## macOS via CuMetal

NVIDIA ships no macOS build. The `metal` SDK is `cuvslam/` compiled for Apple silicon with [CuMetal](https://github.com/jeff-hykin/cuda-metal), which maps the CUDA runtime and kernels onto Metal. The archive carries `libcumetal.dylib` and a prewarmed `share/cumetal-cache` of compiled metallibs; set `CUMETAL_PREBUILT_CACHE_DIR` to that directory so CuMetal can use the read-only store path as its lookup cache. Build scripts live at `cuvslam/cmake/CuMetal.cmake` and `cuvslam/scripts/package_cpp_dist_macos.sh`.

## Determinism

Upstream cuVSLAM seeds its visual-odometry RANSAC from `std::random_device`, so identical input produces slightly different trajectories run-to-run (~0.2 m rmse spread observed). This fork seeds it with a fixed constant (`cuvslam/libs/math/ransac.h`), matching the `seed(0)` that upstream's own SLAM `reproduce_mode` uses, making VO deterministic on the CPU and CuMetal paths.

## License

The crate (everything outside `cuvslam/`) is Apache-2.0. The vendored `cuvslam/` fork remains under the NVIDIA Community License; binaries built from it must ship with `cuvslam/LICENSE` (the flake's SDK packages install it to `share/cuvslam/`).
