//! Build script: compiles the C shim (`csrc/shim.c`) against llama.h and
//! links the static CPU build of llama.cpp that `make llama` produces
//! (`third_party/llama.cpp/build-cpu`, AVX2 + FMA kernels, OpenMP).
//!
//! `LLAMA_CPP_DIR` names another llama.cpp source tree and `LLAMA_BUILD_DIR`
//! another static build of it; `CC` another C compiler.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let src = env::var("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("third_party/llama.cpp"));
    let build = env::var("LLAMA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| src.join("build-cpu"));
    for var in ["LLAMA_CPP_DIR", "LLAMA_BUILD_DIR", "CC"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=csrc/shim.c");

    let libs = [
        ("llama", build.join("src")),
        ("ggml", build.join("ggml/src")),
        ("ggml-cpu", build.join("ggml/src")),
        ("ggml-base", build.join("ggml/src")),
    ];
    for (name, dir) in &libs {
        let a = dir.join(format!("lib{name}.a"));
        if !a.is_file() {
            panic!(
                "llama.cpp is not built: {} is missing. Run `make llama` first \
                 (or set LLAMA_BUILD_DIR to a static build).",
                a.display()
            );
        }
        println!("cargo:rerun-if-changed={}", a.display());
    }

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    compile_shim(&root, &src, &out);

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=jlshim");
    for (name, dir) in &libs {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
    }
    // llama.cpp is C++ and its CPU backend uses OpenMP threads.
    for lib in ["stdc++", "gomp", "m"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

/// `csrc/shim.c` -> `libjlshim.a`, warnings as errors.
fn compile_shim(root: &Path, src: &Path, out: &Path) {
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let obj = out.join("shim.o");
    let status = Command::new(&cc)
        .args([
            "-std=c11", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror", "-c",
        ])
        .arg(root.join("csrc/shim.c"))
        .arg("-I")
        .arg(src.join("include"))
        .arg("-I")
        .arg(src.join("ggml/include"))
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("could not run the C compiler `{cc}`: {e}"));
    assert!(status.success(), "csrc/shim.c did not compile with {cc}");
    let lib = out.join("libjlshim.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("could not run `ar`: {e}"));
    assert!(status.success(), "ar could not archive the shim");
}
