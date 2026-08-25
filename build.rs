use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shim/cuvslam_shim.h");
    println!("cargo:rerun-if-changed=shim/cuvslam_shim.cpp");
    println!("cargo:rerun-if-env-changed=CUVSLAM_SDK_DIR");
    println!("cargo:rustc-check-cfg=cfg(cuvslam_stub)");

    let Ok(sdk_dir) = env::var("CUVSLAM_SDK_DIR").map(PathBuf::from) else {
        // No SDK: compile the stub, whose Tracker::new always errors, so plain
        // `cargo check`/`clippy`/`test` work on machines without cuVSLAM.
        println!("cargo:rustc-cfg=cuvslam_stub");
        return;
    };

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("shim/cuvslam_shim.cpp")
        .include(sdk_dir.join("include"))
        .compile("cuvslam_shim");

    let lib_dir = sdk_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cuvslam");
    // libcuvslam's install name is @rpath/libcuvslam.dylib on macOS; embed the SDK path.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    // rustc-link-arg applies only to this package, so a dependent binary needs its own
    // rpath. `links = "cuvslam"` turns this into DEP_CUVSLAM_LIB_DIR in their build script.
    println!("cargo:lib_dir={}", lib_dir.display());
}
