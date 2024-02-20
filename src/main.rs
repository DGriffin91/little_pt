//cargo run --release -- -f assets/scenes/kitchen.ron -o kitchen.png

pub const EPSILON: f32 = 0.00001;

use std::{fs::File, path::PathBuf};

use clap::*;

use little_pt::Scene;
use ron::de::from_reader;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Render resolution
    #[clap(
        short,
        long,
        default_value = "512 512",
        value_delimiter = ' ',
        number_of_values = 2
    )]
    resolution: Vec<u32>,
    /// Render sample count
    #[arg(short, long, default_value_t = 24)]
    samples: u32,
    /// Trace recursion
    #[arg(short, long, default_value_t = 4)]
    trace_recursion: u32,
    /// Scene file
    #[arg(short, long)]
    file: PathBuf,
    /// Output image file
    #[arg(short, long)]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();
    let f = File::open(args.file).expect("Failed opening file");

    let mut scene: Scene = match from_reader(f) {
        Ok(x) => x,
        Err(e) => {
            println!("Failed to load config: {}", e);

            std::process::exit(1);
        }
    };
    scene.sun_direction = scene.sun_direction.normalize_or_zero();
    scene.camera.dir = scene.camera.dir.normalize_or_zero();

    scene
        .render(
            args.resolution[0],
            args.resolution[1],
            args.samples,
            args.trace_recursion,
        )
        .save(args.out)
        .unwrap();
}
