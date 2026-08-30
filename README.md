# cu_vslam_rs

Rust FFI for [NVIDIA cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) visual odometry: raw bindings in `cu_vslam_rs::ffi` plus a safe `Tracker` wrapper. The repository also vendors a cuVSLAM v17.0.0 fork (`cuvslam/`) and a nix flake that packages a cuVSLAM SDK per platform**.

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

let mut tracker = Tracker::new(&cameras, imu_calibration.as_ref(), &depth_camera_ids, &config)?;
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

`x86_64-cuda12` and `orin` are compiled from `cuvslam/` in a sandboxed nix build. `thor` uses NVIDIA's prebuilt tarball. `metal` uses a CuMetal build of `cuvslam/` hosted as a release artifact, because prewarming its metallib cache needs Xcode's Metal compiler and so cannot run in the nix sandbox.

Every variant we compile ourselves — `x86_64-cuda12`, `orin`, `metal` — is `ENFORCE_GPU=OFF`, so `use_gpu` selects the backend at runtime rather than at build time. On macOS that means `use_gpu: false` runs on the CPU and `true` runs through CuMetal, out of one library. One caveat applies on every platform: **RGBD needs the GPU.** cuVSLAM v17 lifts depth into landmarks only in a CUDA kernel (`lift_kernel`), with no CPU counterpart, so an RGBD tracker with `use_gpu: false` reports success while returning the identity pose. Stereo (`CUV_ODOMETRY_MULTICAMERA`) runs on either backend.

## macOS via CuMetal

NVIDIA ships no macOS build. The `metal` SDK is `cuvslam/` compiled for Apple silicon with [CuMetal](https://github.com/jeff-hykin/cuda-metal), which maps the CUDA runtime and kernels onto Metal. The archive carries `libcumetal.dylib` and a prewarmed `share/cumetal-cache` of compiled metallibs, which CuMetal finds by looking beside the `libcumetal.dylib` it was loaded from; `CUMETAL_PREBUILT_CACHE_DIR` overrides that search but is not needed. A cache is only valid for the exact `libcumetal.dylib` that produced it, so replacing that library means regenerating it with `cumetal_prewarm`. Build scripts live at `cuvslam/cmake/CuMetal.cmake` and `cuvslam/scripts/package_cpp_dist_macos.sh`.

CuMetal always creates its writable cache directory on first use — `~/Library/Caches/io.cumetal/registration-jit/`, or `$CUMETAL_CACHE_DIR` if set — even when the prebuilt cache serves every kernel. An empty `registration-jit/` after a run is expected, not a sign the bundled cache was ignored.

## Determinism

Upstream cuVSLAM seeds its visual-odometry RANSAC from `std::random_device`, so identical input produces slightly different trajectories run-to-run (~0.2 m rmse spread observed). This fork seeds it with a fixed constant (`cuvslam/libs/math/ransac.h`), matching the `seed(0)` that upstream's own SLAM `reproduce_mode` uses.

`cargo test --test determinism` drives a synthetic scene twice and asserts the two trajectories match bit for bit and cover the commanded distance, once per backend the SDK carries. Both assertions pass against `sdk-x86_64-cuda12` and against `sdk-metal` from `cuvslam-v17.0.0-metal.3` on. The RGBD cases only run on the GPU, for the reason above.

Verify a Metal SDK the way another machine would see it: point `CUMETAL_CACHE_DIR` at an empty directory *and* unset `DEVELOPER_DIR`. That makes JIT impossible, so a passing run proves the bundled metallib cache was actually read rather than silently regenerated from a local Xcode.

`cargo test --test metal_smoke` is the short version: one GPU-only stereo run with no CPU fallback. It asserts motion rather than status codes, because every CuMetal defect found so far reported success while returning the identity pose.

## License

The crate (everything outside `cuvslam/`) is Apache-2.0. The vendored `cuvslam/` fork remains under the NVIDIA Community License; binaries built from it must ship with `cuvslam/LICENSE` (the flake's SDK packages install it to `share/cuvslam/`).
