// build.rs — tell cargo where libtensorcore.{dylib,so} lives.
//
// Resolution order:
//   1. $TENSORCORE_LIB_DIR — explicit directory containing libtensorcore.*
//   2. $TENSORCORE_LIB     — full path to libtensorcore.* (we strip the
//                            filename and use the parent directory)
//   3. The build/ dir of an in-repo checkout: tensorcore/build/
//      (resolved from the crate's location: ../../../build)
//
// On macOS we also emit the rpath so the resulting binary can locate the
// dylib at run time without DYLD_LIBRARY_PATH gymnastics.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORCORE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TENSORCORE_LIB");
    println!("cargo:rerun-if-changed=build.rs");

    let lib_dir: Option<PathBuf> = env::var_os("TENSORCORE_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("TENSORCORE_LIB").and_then(|p| {
                PathBuf::from(&p).parent().map(|p| p.to_path_buf())
            })
        })
        .or_else(|| {
            // Crate sits at bindings/rust/tensorcore-rs; the repo's build/
            // dir is three levels up.
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            let candidate = manifest.parent()?.parent()?.parent()?.join("build");
            if candidate.join("libtensorcore.dylib").exists()
                || candidate.join("libtensorcore.so").exists()
            {
                Some(candidate)
            } else {
                None
            }
        });

    if let Some(dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }

    // Dynamic link to libtensorcore.
    println!("cargo:rustc-link-lib=dylib=tensorcore");
}
