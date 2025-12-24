use std::{error::Error, fs::File, io};

use log::info;

use crate::Listing;

pub struct L2_1;

impl Listing for L2_1 {
    fn main(&self) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::new();

        let mut res = client.get(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        ).send()?.error_for_status()?;
        let mut file = File::create("/tmp/the-verdict.txt")?;

        info!(path = "tmp/the-verdict.txt"; "downloaded file");

        io::copy(&mut res, &mut file)?;
        Ok(())
    }
}
