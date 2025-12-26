use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io,
};

use log::info;
use regex::Regex;
use tempfile::NamedTempFile;

use crate::Listing;

pub struct L2_1;

fn tokenize(s: &str) -> Vec<&str> {
    let regex = Regex::new(r#"([,.:;?_!"()\']|--|\s)"#).unwrap();
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

fn uniq(strings: Vec<&str>) -> Vec<String> {
    strings
        .iter()
        .cloned()
        .map(|str| str.to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn construct_vocab(corpus: &String) -> HashMap<String, usize> {
    let mut all_words = uniq(tokenize(&corpus));
    all_words.sort();

    HashMap::from_iter(
        all_words
            .iter()
            .enumerate()
            .map(|tuple| (tuple.1.to_string(), tuple.0)),
    )
}

fn construct_vocab_from_url(corpus_url: String) -> Result<HashMap<String, usize>, Box<dyn Error>> {
    let client = reqwest::blocking::Client::new();

    let mut res = client.get(corpus_url).send()?.error_for_status()?;

    let mut file = NamedTempFile::new()?;

    io::copy(&mut res, &mut file)?;

    let opened = fs::read_to_string(file.path())?;
    info!(count = opened.chars().count(), excerpt:? = opened[0..99]; "file details");

    Ok(construct_vocab(&opened))
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

        let vocab = construct_vocab(&opened);

        info!(vocab_size = vocab.len(); "collected vocabulary");

        Ok(())
    }
}

trait Tokenizer {
    fn encode(&self, text: String) -> Vec<usize>;
    fn decode(&self, ids: Vec<usize>) -> String;
}

struct SimpleTokenizerV1 {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

impl SimpleTokenizerV1 {
    fn new(vocab: HashMap<String, usize>) -> Self {
        SimpleTokenizerV1 {
            str_to_int: vocab.clone(),
            int_to_str: vocab.iter().map(|(k, v)| (*v, k.clone())).collect(),
        }
    }
}

impl Tokenizer for SimpleTokenizerV1 {
    fn encode(&self, text: String) -> Vec<usize> {
        tokenize(&text)
            .iter()
            .map(|token| self.str_to_int[*token])
            .collect()
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        let tokens: Vec<String> = ids.iter().map(|id| self.int_to_str[id].clone()).collect();
        let joined = tokens.join(" ");
        let regex = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();
        let replaced = regex.replace_all(&joined, "$1").to_string();

        replaced
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io,
    };

    use crate::listings::ch02::{SimpleTokenizerV1, Tokenizer, construct_vocab_from_url, tokenize};

    #[test]
    fn test_simple_tokenizer_tokenize() {
        let tokenizer = SimpleTokenizerV1::new(construct_vocab_from_url(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt".to_string(),
        ).unwrap());

        let input =
            "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride.";

        let ids = tokenizer.encode(input.to_string());

        assert_eq!(
            ids,
            [
                1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754,
                793, 7,
            ],
        );

        assert_eq!(
            tokenizer.decode(ids),
            "\" It' s the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
        );
    }

    #[test]
    fn test_tokenize() {
        assert_eq!(
            tokenize("Hello, world. Is this-- a test?"),
            Vec::from([
                "Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"
            ])
        );

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
        );
    }

    #[test]
    fn test_construct_vocab() {
        let vocab = construct_vocab_from_url(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt".to_string(),
        ).unwrap();

        assert_eq!(vocab.len(), 1130);

        assert_eq!(vocab["!"], 0);
        assert_eq!(vocab["\""], 1);
        assert_eq!(vocab["'"], 2);
        assert_eq!(vocab["Her"], 49);
        assert_eq!(vocab["Hermia"], 50);

        assert_eq!(vocab["younger"], 1127);
        assert_eq!(vocab["your"], 1128);
        assert_eq!(vocab["yourself"], 1129);
    }
}
