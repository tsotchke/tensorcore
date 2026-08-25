// build.rs — tell cargo where the tensorcore shared/import library lives.
//
// Resolution order:
//   1. $TENSORCORE_LIB_DIR — explicit directory containing libtensorcore.*
//   2. $TENSORCORE_LIB     — full path to libtensorcore.* (we strip the
//                            filename and use the parent directory)
//   3. The build/ dir of an in-repo checkout: tensorcore/build/
//      (resolved from the crate's location: ../../../build)
//
// On Unix we also emit an rpath so the resulting binary can locate the shared
// library at run time. MSVC does not accept the Unix linker flag.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=TENSORCORE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TENSORCORE_LIB");
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let lib_dir: Option<PathBuf> = env::var_os("TENSORCORE_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("TENSORCORE_LIB").and_then(|p| {
                let library = PathBuf::from(&p);
                let parent = library.parent()?.to_path_buf();
                if target_os == "windows" {
                    let sibling_lib = parent.parent()?.join("lib");
                    if sibling_lib.join("tensorcore.lib").exists() {
                        return Some(sibling_lib);
                    }
                }
                Some(parent)
            })
        })
        .or_else(|| {
            // Crate sits at bindings/rust/tensorcore-rs; the repo's build/
            // dir is three levels up.
            let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            let candidate = manifest.parent()?.parent()?.parent()?.join("build");
            if candidate.join("libtensorcore.dylib").exists()
                || candidate.join("libtensorcore.so").exists()
                || candidate.join("tensorcore.lib").exists() {
                Some(candidate)
            } else if candidate.join("Release").join("tensorcore.lib").exists() {
                Some(candidate.join("Release"))
            } else {
                None
            }
        });

    if let Some(dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
        if matches!(target_os.as_str(), "macos" | "linux" | "freebsd") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }

    // Dynamic link to libtensorcore.
    println!("cargo:rustc-link-lib=dylib=tensorcore");
}
