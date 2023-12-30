use std::{fs::File, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;
use web_rwkv_converter::{convert_safetensors, RENAME, TRANSPOSE};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let file = File::open(&cli.input)?;
    let map = unsafe { Mmap::map(&file)? };

    let output = cli.output.unwrap_or_else(|| {
        let path = cli
            .input
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let stem = cli.input.file_stem().expect("please name the file");
        let name: PathBuf = [&stem.to_string_lossy(), "st"].join(".").into();
        path.join(name)
    });
    convert_safetensors(cli.input, &map, output, RENAME, TRANSPOSE)?;

    Ok(())
}
