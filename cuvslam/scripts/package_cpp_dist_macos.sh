#!/bin/bash
# macOS counterpart to package_cpp_dist.sh.
#
# The archive layout is deliberately identical to the Linux one so a consumer's
# build system only has to switch the library extension. The one addition is
# the CuMetal runtime: libcuvslam.dylib links it, and macOS has no distro
# package to resolve it from, so it travels inside the archive.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <build-dir> <archive-path>" >&2
  echo "  e.g. $0 build_metal dist/cuvslam-cpp-17.0.0-arm64-metal-macos.tar.gz" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR=$(cd "$1" && pwd)
ARCHIVE_PATH="$2"
CUMETAL_PREFIX="${CUMETAL_PREFIX:-$HOME/.local/cumetal}"

declare -a sources=(
  "$BUILD_DIR/bin/libcuvslam.dylib"
  "$BUILD_DIR/bin/cuvslam_api_launcher"
  "$CUMETAL_PREFIX/lib/libcumetal.dylib"
  "$REPO_ROOT/libs/cuvslam/cuvslam2.h"
  "$REPO_ROOT/libs/cuvslam/cuvslam_gpu.h"
  "$REPO_ROOT/libs/cuvslam/ground_constraint2.h"
  "$REPO_ROOT/LICENSE"
)

for source in "${sources[@]}"; do
  if [ ! -f "$source" ]; then
    echo "Required macOS distribution file is missing: $source" >&2
    exit 1
  fi
done

staging_dir=$(mktemp -d -t cuvslam-macos-dist.XXXXXX)
trap 'rm -rf "$staging_dir"' EXIT

mkdir -p "$staging_dir/bin" "$staging_dir/include/cuvslam"
install -m 0755 "${sources[0]}" "$staging_dir/bin/libcuvslam.dylib"
install -m 0755 "${sources[1]}" "$staging_dir/bin/cuvslam_api_launcher"
install -m 0755 "${sources[2]}" "$staging_dir/bin/libcumetal.dylib"
install -m 0644 "${sources[3]}" "$staging_dir/include/cuvslam/cuvslam2.h"
install -m 0644 "${sources[4]}" "$staging_dir/include/cuvslam/cuvslam_gpu.h"
install -m 0644 "${sources[5]}" "$staging_dir/include/cuvslam/ground_constraint2.h"
install -m 0644 "${sources[6]}" "$staging_dir/LICENSE"

# Third-party dylibs that CuMetal links from Homebrew. A machine that never
# installed Homebrew has no /opt/homebrew, so each one is copied in beside the
# runtime and its reference rewritten to @rpath.
vendor_homebrew_dependency() {
    local dependent="$1"
    local absolute_path="$2"
    local file_name
    file_name=$(basename "$absolute_path")
    if [ ! -f "$staging_dir/bin/$file_name" ]; then
        install -m 0755 "$absolute_path" "$staging_dir/bin/$file_name"
        install_name_tool -id "@rpath/$file_name" "$staging_dir/bin/$file_name"
    fi
    install_name_tool -change "$absolute_path" "@rpath/$file_name" "$dependent"
}

while read -r homebrew_path; do
    vendor_homebrew_dependency "$staging_dir/bin/libcumetal.dylib" "$homebrew_path"
done < <(otool -L "$staging_dir/bin/libcumetal.dylib" | awk '/^\t\/opt\/homebrew\//{print $1}')

# Every consumer unpacks the archive somewhere different, so absolute rpaths
# baked in at build time are stripped and replaced with a loader-relative one.
relocate() {
    local target="$1"
    while read -r absolute_rpath; do
        install_name_tool -delete_rpath "$absolute_rpath" "$target" 2>/dev/null || true
    done < <(otool -l "$target" | awk '/LC_RPATH/{found=1} found && /^ *path /{print $2; found=0}' | grep '^/' || true)
    install_name_tool -add_rpath "@loader_path" "$target"
    codesign --force --sign - "$target" >/dev/null 2>&1 || true
}

for binary in "$staging_dir"/bin/*; do
    relocate "$binary"
done

# CuMetal compiles each kernel to a metallib on first launch, which needs Xcode's metal
# compiler. Shipping the compiled kernels means a consumer only needs the Command Line Tools.
# The cache key covers libcumetal's LC_UUID, so this has to run against the staged copy after
# relocation and re-signing rather than the build tree's. DYLD_LIBRARY_PATH is what keeps the
# tool and libcuvslam.dylib on that one copy instead of loading two libcumetal images.
cache_dir="$staging_dir/share/cumetal-cache"
prewarm_bin="$CUMETAL_PREFIX/bin/cumetal_prewarm"
if [ ! -x "$prewarm_bin" ]; then
  echo "Required macOS distribution file is missing: $prewarm_bin" >&2
  exit 1
fi
prewarm_cache=$(mktemp -d -t cuvslam-prewarm.XXXXXX)
trap 'rm -rf "$staging_dir" "$prewarm_cache"' EXIT
mkdir -p "$cache_dir"
DYLD_LIBRARY_PATH="$staging_dir/bin" \
  CUMETAL_CACHE_DIR="$prewarm_cache" \
  CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
  "$prewarm_bin" "$staging_dir/bin/libcuvslam.dylib"
cp "$prewarm_cache"/registration-jit/* "$cache_dir/"

mkdir -p "$(dirname "$ARCHIVE_PATH")"
tar -czf "$ARCHIVE_PATH" -C "$staging_dir" .

echo "Created $ARCHIVE_PATH"
tar -tzf "$ARCHIVE_PATH" | sed 's#^\./##' | awk 'NF && !/\/$/' | LC_ALL=C sort
