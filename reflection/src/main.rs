use reflection::Shader;
use reflection::Strukt;
use std::{fs::File, io::Read, path::PathBuf};

fn main() {
    println!(
        "{}",
        std::env::args()
            .skip(1)
            .flat_map(|arg| {
                let path = PathBuf::from(arg);
                let mut buf = Vec::new();
                File::open(path).unwrap().read_to_end(&mut buf).unwrap();
                let shaders = Shader::from_spirv(&buf).unwrap();
                shaders.into_iter().map(|s| s.strukts)
            })
            .flatten()
            .fold(Vec::new(), |mut strukts: Vec<Strukt>, strukt: Strukt| {
                if strukts.iter().any(|s| s.name == strukt.name) {
                    return strukts;
                }
                strukts.push(strukt);
                strukts
            })
            .into_iter()
            .map(|s| { format!("{}\n", s) })
            .fold(String::new(), |a, b| a + &b)
    );
}
