use std::{
    error::Error,
    fs::{self, File},
    io,
};

use log::info;
use regex::Regex;

use crate::Listing;

pub struct L2_1;

fn tokenize(s: &str) -> Vec<&str> {
    let regex = Regex::new(r#"([,.:;?_!"()\"]|--|\s)"#).unwrap();
    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in regex.find_iter(s) {
        if mat.start() > last_end {
            result.push(&s[last_end..mat.start()]);
        }
        result.push(mat.as_str());
        last_end = mat.end();
    }

    if last_end < s.len() {
        result.push(&s[last_end..]);
    }

    result
        .into_iter()
        .filter(|s| !s.trim_end().is_empty())
        .collect()
}

impl Listing for L2_1 {
    fn main(&self) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::new();

        let path = "/tmp/the-verdict.txt";

        let mut res = client.get(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        ).send()?.error_for_status()?;
        let mut file = File::create(path)?;

        info!(path; "downloaded file");

        io::copy(&mut res, &mut file)?;

        let opened = fs::read_to_string(path)?;
        info!(count = opened.chars().count(), excerpt:? = opened[0..99]; "file details");

        let tokenized = tokenize("Hello, world. Is this-- a test?");
        info!(tokenized:? = tokenized; "tokenized");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io,
    };

    use crate::listings::ch02::tokenize;

    #[test]
    fn test_tokenize() {
        assert_eq!(
            tokenize("Hello, world. Is this-- a test?"),
            Vec::from([
                "Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"
            ])
        )
    }

    #[test]
    fn test_tokenize_short_story() {
        let client = reqwest::blocking::Client::new();

        let path = "/tmp/the-verdict.txt";

        let mut res = client.get(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
        ).send().unwrap().error_for_status().unwrap();
        let mut file = File::create(path).unwrap();

        io::copy(&mut res, &mut file).unwrap();

        let opened = fs::read_to_string(path).unwrap();

        assert_eq!(
            tokenize(&opened)[0..30],
            [
                "I", "HAD", "always", "thought", "Jack", "Gisburn", "rather", "a", "cheap",
                "genius", "--", "though", "a", "good", "fellow", "enough", "--", "so", "it", "was",
                "no", "great", "surprise", "to", "me", "to", "hear", "that", ",", "in"
            ]
        )
    }
}
