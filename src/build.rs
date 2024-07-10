extern crate cc;

fn main() {
    let libtorch = std::env::var("LIBTORCH").expect("LIBTORCH environment variable not set");
    
    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .include(format!("{}/include", libtorch))
        .include(format!("{}/include/torch/csrc/api/include", libtorch))
        .file("libtch/torch_api.cpp")
        .file("libtch/torch_api_generated.cpp")
        .compile("torch_api");
    
    println!("cargo:rustc-link-search=native={}/lib", libtorch);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
}
