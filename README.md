# cu_vslam_rs

Rust FFI for [NVIDIA cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) visual odometry: raw bindings in `cu_vslam_rs::ffi` plus a safe `Tracker` wrapper. The repository also vendors a cuVSLAM v17.0.0 fork (`cuvslam/`) and a nix flake that packages a cuVSLAM SDK per platform.

## Layout

| Path | What it is | License |
|---|---|---|
| `src/`, `shim/`, `build.rs` | The `cu_vslam_rs` crate: extern-C shim over `cuvslam2.h`, raw `ffi` bindings, safe `Tracker` | NVIDIA Community License |
| `cuvslam/` | Fork of nvidia-isaac/cuVSLAM v17.0.0 with odometry-state serialization and a deterministic VO RANSAC seed | NVIDIA Community License (see `cuvslam/LICENSE`) |
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
  lib/libcuvslam.so
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
# The default variant for your machine (orin on aarch64-linux, cuda12 on x86_64-linux):
nix build github:jeff-hykin/cu_vslam_rs
export CUVSLAM_SDK_DIR=$PWD/result

# Or a specific variant:
nix build github:jeff-hykin/cu_vslam_rs#sdk-x86_64-cuda12   # built from cuvslam/, sm_89 + sm_120
nix build github:jeff-hykin/cu_vslam_rs#sdk-x86_64-cuda13   # NVIDIA prebuilt tarball
nix build github:jeff-hykin/cu_vslam_rs#sdk-orin            # built from cuvslam/, sm_87
nix build github:jeff-hykin/cu_vslam_rs#sdk-thor            # NVIDIA prebuilt tarball

# Dev shell with CUVSLAM_SDK_DIR already set:
nix develop github:jeff-hykin/cu_vslam_rs
cargo build
```

Flakes that consume this one can take it as an input and use `cu-vslam-rs.packages.${system}."sdk-<variant>"` as `CUVSLAM_SDK_DIR` for their own Rust builds.

`x86_64-cuda12` and `orin` are compiled from `cuvslam/` in a sandboxed nix build (`ENFORCE_GPU=OFF`, so CPU/GPU is a runtime switch). `thor` uses NVIDIA's prebuilt tarball.

## Determinism

Upstream cuVSLAM seeds its visual-odometry RANSAC from `std::random_device`, so identical input produces slightly different trajectories run-to-run (~0.2 m rmse spread observed). This fork seeds it with a fixed constant (`cuvslam/libs/math/ransac.h`), matching the `seed(0)` that upstream's own SLAM `reproduce_mode` uses.

`cargo test --test determinism` drives a synthetic scene twice and asserts the two trajectories match bit for bit and cover the commanded distance. Both assertions pass against `sdk-x86_64-cuda12`.

## License

Everything here is under the NVIDIA Community License (`LICENSE`), including the vendored `cuvslam/` fork. Binaries built from it must ship the license text; the flake's SDK packages install it to `share/cuvslam/`.
