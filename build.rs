use std::env;
use std::path::PathBuf;
use std::process::Command;

const COMMANDS: &[&str] = &[
    "load_model",
    "unload_model",
    "generate",
    "is_loaded",
    "get_model_info",
    "list_models",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS).build();

    #[cfg(target_os = "macos")]
    compile_swift_macos();
}

#[cfg(target_os = "macos")]
fn should_clean_swift_build(swift_dir: &PathBuf, build_type: &str) -> bool {
    use std::fs;

    let lib_path = swift_dir.join(format!(".build/{}/libMLXInference.a", build_type));

    if !lib_path.exists() {
        return true;
    }

    let artifact_mtime = match fs::metadata(&lib_path) {
        Ok(m) => match m.modified() {
            Ok(t) => t,
            Err(_) => return true,
        },
        Err(_) => return true,
    };

    let sources = [
        swift_dir.join("Sources/MLXWrapper.swift"),
        swift_dir.join("Sources/Bridge.h"),
        swift_dir.join("Package.swift"),
    ];

    for source in &sources {
        if !source.exists() {
            continue;
        }
        match fs::metadata(source) {
            Ok(m) => match m.modified() {
                Ok(t) if t > artifact_mtime => return true,
                Err(_) => return true,
                _ => {}
            },
            Err(_) => return true,
        }
    }

    false
}

#[cfg(target_os = "macos")]
fn compile_swift_macos() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let swift_dir = manifest_dir.join("macos");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let build_type = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    println!(
        "cargo:warning=Building MLX Swift package in {} mode",
        build_type
    );

    if should_clean_swift_build(&swift_dir, build_type) {
        println!("cargo:warning=MLX Swift sources changed — running clean");
        let _ = Command::new("xcrun")
            .args(["swift", "package", "clean"])
            .current_dir(&swift_dir)
            .status();
    } else {
        println!("cargo:warning=MLX Swift sources unchanged — skipping clean");
    }

    let status = Command::new("xcrun")
        .args(["swift", "build", "-c", build_type, "--package-path"])
        .arg(&swift_dir)
        .status()
        .expect("Failed to execute swift build — is Xcode installed?");

    if !status.success() {
        println!("cargo:warning=MLX Swift build failed. Requires Xcode and macOS 26+.");
        panic!("Swift build failed");
    }

    let lib_path = swift_dir.join(format!(".build/{}/libMLXInference.a", build_type));
    if !lib_path.exists() {
        panic!("Swift library not found at: {:?}", lib_path);
    }

    std::fs::copy(&lib_path, out_dir.join("libMLXInference.a"))
        .expect("Failed to copy Swift library");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=MLXInference");

    // Force-load all ObjC classes from the static library.
    // MLXLMCommon discovers MLXLLM.TrampolineModelFactory via NSClassFromString
    // at runtime; without -ObjC the linker dead-strips it.
    println!("cargo:rustc-link-arg=-Wl,-ObjC");

    // macOS frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Swift runtime + compatibility library search paths
    // Note: macOS uses "macosx" (not "macos") for the platform directory
    let candidates = [
        "/usr/lib/swift",
        "/Library/Developer/CommandLineTools/usr/lib/swift/macosx",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
        "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
    ];
    for p in candidates {
        if std::path::Path::new(p).exists() {
            println!("cargo:rustc-link-search=native={}", p);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", p);
        }
    }

    println!("cargo:rerun-if-changed=macos/Sources/MLXWrapper.swift");
    println!("cargo:rerun-if-changed=macos/Sources/Bridge.h");
    println!("cargo:rerun-if-changed=macos/Package.swift");
}
