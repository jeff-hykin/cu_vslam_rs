{
  description = "cu_vslam_rs: Rust FFI for NVIDIA cuVSLAM, with per-platform SDK packages (including macOS via CuMetal) built from the vendored fork in cuvslam/";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        isDarwin = nixpkgs.lib.hasSuffix "-darwin" system;
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; cudaSupport = !isDarwin; };
        };
        # Orin's iGPU needs nixpkgs' Jetson CUDA libraries; the default SBSA ones fail at cusolverDnCreate.
        pkgsOrin = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; cudaSupport = true; cudaCapabilities = [ "8.7" ]; };
        };
        cudaPkgsFor = name: (if name == "orin" then pkgsOrin else pkgs);

        # Every C++ build NVIDIA ships for this release. The ubuntu flavor does not
        # matter under autoPatchelf. `cuda` is the matching nixpkgs set: a runtime
        # newer than the SDK's fails inside cuSOLVER.
        sdks = {
          x86_64-cuda12 = {
            url = "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/cuvslam-cpp-17.0.0-x86_64-cuda12.6.3-ubuntu24.04.tar.gz";
            hash = "sha256-X2iCVMzKTlOuFcLyZJZU3vgQOdoWAU4LuXU0WpdyE9Q=";
            system = "x86_64-linux";
            cuda = "cudaPackages_12_6";
          };
          x86_64-cuda13 = {
            url = "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/cuvslam-cpp-17.0.0-x86_64-cuda13.2.0-ubuntu24.04.tar.gz";
            hash = "sha256-fEG94wknx6JBrDml3r0Kuy/yjS0HfxgkKUT26UuGbkg=";
            system = "x86_64-linux";
            cuda = "cudaPackages_13_2";
          };
          # Jetson Orin, sm_87, JetPack 6.
          orin = {
            url = "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/cuvslam-cpp-17.0.0-orin-cuda12.6.3-ubuntu22.04.tar.gz";
            hash = "sha256-V6e4zKsSZJG0rCqaPkHyw7wSPVCyeN/6Ma/tiY9GDw0=";
            system = "aarch64-linux";
            cuda = "cudaPackages_12_6";
          };
          # Jetson Thor, sm_110, JetPack 7.
          thor = {
            url = "https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.0/cuvslam-cpp-17.0.0-thor-cuda13.0.1-ubuntu24.04.tar.gz";
            hash = "sha256-w5b476aY+oS8XVQn9EodgwXf8nrhnD9aioykLSoZTT8=";
            system = "aarch64-linux";
            cuda = "cudaPackages_13_0";
          };
          # Ours, since NVIDIA ships no macOS build: cuvslam/ compiled for Apple
          # silicon against CuMetal, targeting macOS 13 on apple-m1. It additionally
          # carries libcumetal.dylib and share/cumetal-cache. ENFORCE_GPU=OFF like
          # the fork builds below, so use_gpu is a runtime switch here too.
          metal = {
            url = "https://github.com/jeff-hykin/cu_vslam_rs/releases/download/cuvslam-v17.0.0-metal.5/cuvslam-cpp-17.0.0-arm64-metal-macos.tar.gz";
            hash = "sha256-ugma/HyDi7caSXwVCMCtZB3MAi93BWbMcbvSrLYL1Xw=";
            system = "aarch64-darwin";
          };
          # Non-Jetson ARM. NVIDIA ships no generic-arm tarball, so this variant only
          # exists as a fork build; everything but `system` comes from forkBuilds.
          aarch64 = {
            system = "aarch64-linux";
          };
          # ARM with no NVIDIA GPU (Raspberry Pi).
          aarch64-cpu = {
            system = "aarch64-linux";
          };
        };

        # Variants built from the fork instead of NVIDIA's prebuilt tarball. Only
        # metal stays on a tarball: there is no CUDA toolchain to fork-build with on
        # macOS. cudssCuda picks which cuDSS 0.8.0.10 archive cuNLS links against;
        # it must match the nvcc major.
        # cuda12_8 on x86: sm_120 (Blackwell) needs nvcc >= 12.8.
        forkBuilds = {
          x86_64-cuda12 = {
            cuda = "cudaPackages_12_8";
            archs = "89;120";
            cudssPlatform = "linux-x86_64";
            cudssCuda = "cuda12";
            cudssSha256 = "01s7xssfjadz1zfjprwp66j82h04snfpmjxg149m6a2bqq2nlw99";
          };
          x86_64-cuda13 = {
            cuda = "cudaPackages_13_3";
            archs = "89;120";
            cudssPlatform = "linux-x86_64";
            cudssCuda = "cuda13";
            cudssSha256 = "0ds9g8jv95pjnmvx9axm5d24in3161dsrjjqs42ykfywh3yza65s";
          };
          orin = {
            cuda = "cudaPackages_12_6";
            archs = "87";
            # cuDSS >= 0.8 ships aarch64 as "linux-sbsa".
            cudssPlatform = "linux-sbsa";
            cudssCuda = "cuda12";
            cudssSha256 = "12xixcrfl9yv2gf7rc0nkn2fhh171m2mnhvpfvgrfs4qbh0jd54l";
          };
          # Jetson Thor, sm_110. cuda13_0 matches JetPack 7's driver.
          thor = {
            cuda = "cudaPackages_13_0";
            archs = "110";
            cudssPlatform = "linux-sbsa";
            cudssCuda = "cuda13";
            cudssSha256 = "02clxpqz0b60rfyrkz763yk0n15kk8bbn6wpqp1i0bkrjrbpxzn5";
          };
          # Every ARM target in one library, which is what the PyPI wheel ships: the
          # tag space has a single aarch64 slot, so it carries Hopper, Thor and
          # Blackwell cubins and falls back to CPU on the ARM boxes with no NVIDIA
          # GPU at all. Orin stays out of reach here however wide the arch list gets,
          # since JetPack 6 predates the CUDA 13 driver.
          aarch64 = {
            cuda = "cudaPackages_13_3";
            archs = "90;110;120";
            cudssPlatform = "linux-sbsa";
            cudssCuda = "cuda13";
            cudssSha256 = "02clxpqz0b60rfyrkz763yk0n15kk8bbn6wpqp1i0bkrjrbpxzn5";
          };
          # No `cuda`: built USE_CUDA=OFF.
          aarch64-cpu = { };
        };

        # The fork's FetchContent dependencies, pre-fetched (the sandbox is offline) and
        # handed to cmake as FETCHCONTENT_SOURCE_DIR_* overrides. URL hashes are copied
        # verbatim from the fork's cmake/ext/*.cmake pins.
        depTarball = name: url: sha256: pkgs.runCommand "cuvslam-dep-${name}" { } ''
          mkdir -p $out
          tar xf ${pkgs.fetchurl { inherit url sha256; }} --strip-components=1 -C $out
        '';
        depGithub = name: repo: rev: sha256: pkgs.fetchzip {
          name = "cuvslam-dep-${name}";
          url = "https://github.com/${repo}/archive/${rev}.tar.gz";
          inherit sha256;
        };
        forkDepsFor = fork: {
          eigen = depTarball "eigen"
            "https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz"
            "b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2";
          # cuNLS pins /usr/local/cuda and its own arch list with plain set()s ahead of
          # project(), stomping the parent configuration; drop them so the nix toolchain
          # and our CMAKE_CUDA_ARCHITECTURES flow through.
          cunls = pkgs.runCommand "cuvslam-dep-cunls" { } ''
            mkdir -p $out
            tar xf ${pkgs.fetchurl {
              url = "https://github.com/nvidia-isaac/cuNLS/archive/refs/tags/Release_07_13_2026.tar.gz";
              sha256 = "23b2917ae3903e6a688edb1652e40202d314527cd7fa9db68c762f0429375f77";
            }} --strip-components=1 -C $out
            sed -i -e '/set(CMAKE_CUDA_COMPILER/d' -e '/set(CMAKE_CUDA_ARCHITECTURES/d' \
              $out/CMakeLists.txt
            # thrust::make_tuple reached sparse_matrix.cu through zip_iterator.h until
            # CCCL 3 (CUDA 13) stopped including tuple.h from there.
            sed -i '/#include <thrust\/transform.h>/a #include <thrust/tuple.h>' \
              $out/cunls/minimizer/sparse_matrix.cu
          '';
          lmdb = depTarball "lmdb"
            "https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_0.9.31.tar.gz"
            "dd70a8c67807b3b8532b3e987b0a4e998962ecc28643e1af5ec77696b081c9b0";
          gflags = depTarball "gflags"
            "https://github.com/gflags/gflags/archive/v2.3.0.tar.gz"
            "f619a51371f41c0ad6837b2a98af9d4643b3371015d873887f7e8d3237320b2f";
          googletest = depTarball "googletest"
            "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz"
            "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c";
          jsoncpp = depTarball "jsoncpp"
            "https://github.com/open-source-parsers/jsoncpp/archive/1.9.6.tar.gz"
            "f93b6dd7ce796b13d02c108bc9f79812245a82e577581c4c9aabe57075c90ea2";
          libjpeg = depTarball "libjpeg"
            "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.3/libjpeg-turbo-3.1.3.tar.gz"
            "075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0";
          libpng = depTarball "libpng"
            "https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.55.tar.gz"
            "71a2c5b1218f60c4c6d2f1954c7eb20132156cae90bdb90b566c24db002782a6";
          spdlog = depTarball "spdlog"
            "https://github.com/gabime/spdlog/archive/v1.17.0.tar.gz"
            "d8862955c6d74e5846b3f580b1605d2428b11d97a410d86e2fb13e857cd3a744";
          "yaml-cpp" = depTarball "yaml-cpp"
            "https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz"
            "25cb043240f828a8c51beb830569634bc7ac603978e0f69d6b63558dadefd49a";
          zlib = depTarball "zlib"
            "https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz"
            "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c";
          cnpy = depGithub "cnpy" "rogersce/cnpy"
            "4e8810b1a8637695171ed346ce68f6984e585ef4"
            "1dgw86l47mwwbs11zqf8sas823qpjfgy0904hy0gmak8wfjw7hrl";
          circularbuffer = depGithub "circularbuffer" "vinitjames/circularbuffer"
            "cef66805cb5424e27300a966becc7c2678117c27"
            "0vi0y131x106v1fvpq3wr0dlyh3q4higkc7lqqrvvvmklvz1mnxn";
          dense_hash_map = depGithub "dense_hash_map" "Jiwan/dense_hash_map"
            "74277fc4813028ae4a9e8d9176788eb8001177a6"
            "0q4z9zvzas2pg566g889j4chy6w3m41bb82zrxs6ihl1arnral6q";
          # Downloaded by cuNLS's own cmake (AddCUDSS.cmake), also via FetchContent.
          cudss = depTarball "cudss"
            "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/${fork.cudssPlatform}/libcudss-${fork.cudssPlatform}-0.8.0.10_${fork.cudssCuda}-archive.tar.xz"
            fork.cudssSha256;
        };

        cudaLibs = name: sdk: pkgs.lib.optionals (sdk ? cuda) (
          let cudaSet = (cudaPkgsFor name).${sdk.cuda}; in
          with cudaSet; [ cuda_cudart libcublas libcusolver libcusparse ]
            ++ pkgs.lib.optionals (cudaSet ? libnvjitlink) [ libnvjitlink ]
        );

        sdkFor = name: sdk: pkgs.stdenv.mkDerivation {
          pname = "cuvslam-sdk-${name}";
          version = "17.0.0";
          src = pkgs.fetchurl { inherit (sdk) url hash; };
          sourceRoot = ".";
          # ELF-only, and none of the CUDA runtime has a darwin build.
          nativeBuildInputs = pkgs.lib.optionals (!isDarwin) [ pkgs.autoPatchelfHook ];
          buildInputs = pkgs.lib.optionals (!isDarwin) [ pkgs.stdenv.cc.cc.lib ]
            ++ cudaLibs name sdk;
          installPhase = ''
            runHook preInstall
            mkdir -p $out/lib $out/include $out/bin $out/share/cuvslam
            cp bin/libcuvslam.${if isDarwin then "dylib" else "so"} $out/lib/
            cp bin/cuvslam_api_launcher $out/bin/ || true
            cp -r include/cuvslam $out/include/
            # cuvslam2.h declares GetVersion(int32_t*) but includes only <cstddef>, so it
            # relied on <cstdint> arriving through another libstdc++ header. gcc 15 stopped
            # leaking it, and every error after the undeclared type is cascade from this.
            sed -i '/#include <cstddef>/a #include <cstdint>' $out/include/cuvslam/cuvslam2.h
            # The NVIDIA Community License requires this to travel with the binary.
            cp LICENSE $out/share/cuvslam/
            echo "Licensed by NVIDIA Corporation under the NVIDIA Community License." \
              > $out/share/cuvslam/NOTICE
            ${pkgs.lib.optionalString isDarwin ''
              cp bin/libcumetal.dylib $out/lib/
              cp -r share/cumetal-cache $out/share/
              # The archive keeps every dylib beside the launcher; splitting them into
              # lib/ and bin/ moves the launcher one directory away from them.
              ${pkgs.darwin.cctools}/bin/install_name_tool \
                -add_rpath "@loader_path/../lib" $out/bin/cuvslam_api_launcher
            ''}
            runHook postInstall
          '';
          meta.license = pkgs.lib.licenses.unfree;  # NVIDIA Community License
        };

        # Same output shape as sdkFor, compiled from cuvslam/. ENFORCE_GPU=OFF so one
        # library carries both backends and use_gpu becomes a runtime switch.
        forkSdkFor = name: fork: let
          hasCuda = fork ? cuda;
          cudaSet = (cudaPkgsFor name).${fork.cuda};
          # cuNLS and its cuDSS are CUDA-only.
          deps = if hasCuda then forkDepsFor fork else builtins.removeAttrs (forkDepsFor fork) [ "cunls" "cudss" ];
          stdenv = if hasCuda then cudaSet.backendStdenv else pkgs.stdenv;
        in stdenv.mkDerivation {
          pname = "cuvslam-fork-${name}";
          version = "17.0.0-odom-state";
          src = ./cuvslam;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config pkgs.removeReferencesTo ] ++ pkgs.lib.optionals hasCuda [ cudaSet.cuda_nvcc ];
          buildInputs = pkgs.lib.optionals hasCuda (
            [ cudaSet.cuda_cudart cudaSet.libcublas cudaSet.libcusolver cudaSet.libcusparse ]
            ++ pkgs.lib.optionals (cudaSet ? libnvjitlink) [ cudaSet.libnvjitlink ]
            ++ pkgs.lib.optionals (cudaSet ? cuda_cccl) [ cudaSet.cuda_cccl ]
          );
          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DENFORCE_GPU=OFF"
          ] ++ (if hasCuda then [
            "-DCMAKE_CUDA_ARCHITECTURES=${fork.archs}"
            # The fork caches /usr/local/cuda paths; point both at the nix toolkit.
            "-DCUDAToolkit_ROOT=${cudaSet.cudatoolkit}"
            "-DCMAKE_CUDA_COMPILER=${cudaSet.cudatoolkit}/bin/nvcc"
          ] else [ "-DUSE_CUDA=OFF" "-DUSE_CUNLS=OFF" ]);
          # Some dep builds write into their own source tree (zlib renames zconf.h), so
          # hand FetchContent writable copies rather than read-only store paths.
          preConfigure = pkgs.lib.concatStrings (pkgs.lib.mapAttrsToList
            (depName: depSource: ''
              mkdir -p dep-src
              cp -r --no-preserve=mode ${depSource} dep-src/${depName}
              cmakeFlagsArray+=("-DFETCHCONTENT_SOURCE_DIR_${pkgs.lib.toUpper depName}=$PWD/dep-src/${depName}")
            '')
            deps);
          buildFlags = [ "cuvslam" ];
          installPhase = ''
            runHook preInstall
            mkdir -p $out/lib $out/include/cuvslam $out/share/cuvslam
            cp bin/libcuvslam.so $out/lib/
            cp $src/libs/cuvslam/cuvslam2.h $out/include/cuvslam/
            cp $src/LICENSE $out/share/cuvslam/
            echo "Built from cuvslam/ in github.com/jeff-hykin/cu_vslam_rs with ENFORCE_GPU=OFF." \
              > $out/share/cuvslam/NOTICE
            runHook postInstall
          '';
          # nvcc embeds its link line, so the whole toolkit would otherwise ride along at runtime.
          postFixup = pkgs.lib.optionalString hasCuda ''
            remove-references-to -t ${cudaSet.cudatoolkit} $out/lib/libcuvslam.so
          '';
          disallowedReferences = pkgs.lib.optionals hasCuda [ cudaSet.cudatoolkit ];
          meta.license = pkgs.lib.licenses.unfree;  # NVIDIA Community License
        };

        forThisSystem = pkgs.lib.filterAttrs (_: sdk: sdk.system == system) sdks;

        # Fork-built variants override the tarball.
        sdkPackageFor = name: sdk:
          if forkBuilds ? ${name} then forkSdkFor name forkBuilds.${name} else sdkFor name sdk;

        # Nix can't see the SoC or driver, so the variant is picked at runtime.
        cuvslamVariant = pkgs.writeShellApplication {
          name = "cuvslam-variant";
          runtimeInputs = [ pkgs.coreutils pkgs.gnused ];
          text = ''
            if [ -n "''${CUVSLAM_VARIANT:-}" ]; then
              echo "$CUVSLAM_VARIANT"
              exit 0
            fi
            case "$(uname -s)-$(uname -m)" in
              Darwin-arm64) echo metal ;;
              Linux-aarch64)
                case "$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null || true)" in
                  *tegra264*) echo thor ;;
                  *tegra234*) echo orin ;;
                  *tegra194* | *tegra210*)
                    echo "cuvslam-variant: no cuVSLAM GPU build for JetPack 4/5; only use_gpu=false will work" >&2
                    echo aarch64-cpu ;;
                  *)
                    if [ -e /proc/driver/nvidia/version ]; then
                      echo aarch64
                    else
                      echo aarch64-cpu
                    fi ;;
                esac ;;
              Linux-x86_64)
                cuda_major=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -n 1 || true)
                case "$cuda_major" in
                  "")
                    echo "cuvslam-variant: no NVIDIA driver found; only use_gpu=false will work" >&2
                    echo x86_64 ;;
                  12) echo x86_64-cuda12 ;;
                  [0-9] | 1[01])
                    echo "cuvslam-variant: this driver supports CUDA $cuda_major and the GPU path needs 12+; only use_gpu=false will work" >&2
                    echo x86_64 ;;
                  *) echo x86_64-cuda13 ;;
                esac ;;
              *)
                echo "cuvslam-variant: no cuVSLAM build for $(uname -s)-$(uname -m)" >&2
                exit 1 ;;
            esac
          '';
        };

        allVariants = pkgs.lib.mapAttrs sdkPackageFor forThisSystem
          // pkgs.lib.optionalAttrs (forThisSystem ? x86_64-cuda12) {
               # The CPU-fallback name for x86 with no NVIDIA driver, mirroring
               # `aarch64`. Same derivation as x86_64-cuda12: that build is
               # ENFORCE_GPU=OFF so it runs CPU-only, and a cuda12 binary works
               # under whichever driver gets installed later.
               x86_64 = sdkPackageFor "x86_64-cuda12" forThisSystem.x86_64-cuda12;
             };

        # Excludes aarch64: a third CUDA stack on every Jetson.
        bundledVariants = builtins.removeAttrs allVariants [ "aarch64" ];

        sdkBundle = pkgs.runCommand "cuvslam-sdk-bundle" { } ''
          mkdir -p $out/bin
          ${pkgs.lib.concatStrings (pkgs.lib.mapAttrsToList (name: sdk: ''
            ln -s ${sdk} $out/${name}
          '') bundledVariants)}
          ln -s ${cuvslamVariant}/bin/cuvslam-variant $out/bin/
          cat > $out/bin/cuvslam-sdk-dir <<EOF
          #!${pkgs.runtimeShell}
          variant=\$(${cuvslamVariant}/bin/cuvslam-variant) || exit 1
          [ -e "$out/\$variant" ] || { echo "cuvslam-sdk-dir: the default has no \$variant SDK; build .#sdk-\$variant" >&2; exit 1; }
          echo "$out/\$variant"
          EOF
          chmod +x $out/bin/cuvslam-sdk-dir
        '';

        # The crate, linked against a given SDK.
        crateFor = sdkPackage: pkgs.rustPlatform.buildRustPackage {
          pname = "cu_vslam_rs";
          version = "0.1.0";
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./Cargo.toml ./Cargo.lock ./build.rs ./src ./shim
            ];
          };
          cargoLock.lockFile = ./Cargo.lock;
          env.CUVSLAM_SDK_DIR = sdkPackage;
          # Tests link libcuvslam, whose runtime wants a GPU the sandbox lacks.
          doCheck = false;
        };
      in {
        packages = pkgs.lib.mapAttrs' (name: sdk: { name = "sdk-${name}"; value = sdk; }) allVariants
          // { default = sdkBundle; cuvslam-variant = cuvslamVariant; };

        inherit bundledVariants;

        # A compile check against each SDK.
        checks = pkgs.lib.mapAttrs' (name: sdk: { name = "crate-${name}"; value = crateFor sdk; })
          (builtins.removeAttrs allVariants [ "x86_64" ]);

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
          shellHook = ''
            if [ -z "''${CUVSLAM_SDK_DIR:-}" ]; then
              case "$(${cuvslamVariant}/bin/cuvslam-variant)" in
${pkgs.lib.concatStrings (pkgs.lib.mapAttrsToList (name: sdk: ''
                ${name}) cuvslam_sdk_drv=${builtins.unsafeDiscardStringContext sdk.drvPath} ;;
'') allVariants)}                *) cuvslam_sdk_drv= ;;
              esac
              if [ -n "$cuvslam_sdk_drv" ] \
                && CUVSLAM_SDK_DIR=$(nix build --no-link --print-out-paths "$cuvslam_sdk_drv^out"); then
                export CUVSLAM_SDK_DIR
              fi
              unset cuvslam_sdk_drv
            fi
          '';
          # nix's glibc does not read ld.so.cache, so name the Jetson driver dirs.
          LD_LIBRARY_PATH = pkgs.lib.optionalString (system == "aarch64-linux") (
            pkgs.lib.concatStringsSep ":" [
              "/usr/lib/aarch64-linux-gnu/nvidia"
              "/usr/lib/aarch64-linux-gnu/tegra"
            ]
          );
        };
      });
}
