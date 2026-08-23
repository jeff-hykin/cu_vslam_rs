use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shim/cuvslam_shim.h");
    println!("cargo:rerun-if-changed=shim/cuvslam_shim.cpp");
    println!("cargo:rerun-if-env-changed=CUVSLAM_SDK_DIR");

    let sdk_dir = PathBuf::from(
        env::var("CUVSLAM_SDK_DIR").expect("set CUVSLAM_SDK_DIR to a cuVSLAM SDK (include/cuvslam/cuvslam2.h + lib/)"),
    );

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
}
