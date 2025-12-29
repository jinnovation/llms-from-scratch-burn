#![allow(dead_code)]
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io,
};

use burn::{
    Tensor,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    prelude::Backend,
    tensor::{Int, TensorData},
};
use log::info;
use regex::Regex;
use tempfile::NamedTempFile;

use crate::Listing;

pub struct L2_1;

pub struct E2_1;

impl Listing for E2_1 {
    fn main(&self) -> Result<(), Box<dyn Error>> {
        let input = "Akwirw ier";
        let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
        let allowed_special: HashSet<&str> = HashSet::new();
        let encoded = tokenizer.encode(input, &allowed_special).0;
        let decoded = tokenizer.decode(encoded)?;

        info!(decoded:?, input:?, equal = decoded==input; "round-tripped");

        Ok(())
    }
}

static THE_VERDICT_URL: &str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";

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

fn construct_vocab(corpus: &str, additional_tokens: Option<Vec<String>>) -> HashMap<String, usize> {
    let mut all_words = uniq(tokenize(corpus));
    all_words.sort();
    if let Some(tokens) = additional_tokens {
        all_words.extend(tokens.iter().map(|t| t.to_string()));
    }

    HashMap::from_iter(
        all_words
            .iter()
            .enumerate()
            .map(|tuple| (tuple.1.to_string(), tuple.0)),
    )
}

fn text_from_url(url: String) -> Result<String, Box<dyn Error>> {
    let client = reqwest::blocking::Client::new();

    let mut res = client.get(url).send()?.error_for_status()?;

    let mut file = NamedTempFile::new()?;

    io::copy(&mut res, &mut file)?;

    Ok(fs::read_to_string(file.path())?)
}

fn construct_vocab_from_url(
    corpus_url: String,
    additional_tokens: Option<Vec<String>>,
) -> Result<HashMap<String, usize>, Box<dyn Error>> {
    let opened = text_from_url(corpus_url)?;
    info!(count = opened.chars().count(), excerpt:? = opened[0..99]; "file details");

    Ok(construct_vocab(&opened, additional_tokens))
}

impl Listing for L2_1 {
    fn main(&self) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::new();

        let path = "/tmp/the-verdict.txt";

        let mut res = client.get(THE_VERDICT_URL).send()?.error_for_status()?;
        let mut file = File::create(path)?;

        info!(path; "downloaded file");

        io::copy(&mut res, &mut file)?;

        let opened = fs::read_to_string(path)?;
        info!(count = opened.chars().count(), excerpt:? = opened[0..99]; "file details");

        let vocab = construct_vocab(&opened, None);

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
        Self {
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

        regex.replace_all(&joined, "$1").to_string()
    }
}

struct SimpleTokenizerV2 {
    str_to_int: HashMap<String, usize>,
    int_to_str: HashMap<usize, String>,
}

enum Corpus {
    Raw(String),
    Url(String),
}

impl Corpus {
    fn to_vocab(&self, additional_tokens: Option<Vec<String>>) -> HashMap<String, usize> {
        match self {
            Corpus::Raw(text) => construct_vocab(text, additional_tokens),
            Corpus::Url(url) => {
                construct_vocab_from_url(url.to_string(), additional_tokens).unwrap()
            }
        }
    }
}

impl SimpleTokenizerV2 {
    fn new(corpus: Corpus) -> Self {
        let additional_tokens = vec!["<|endoftext|>".into(), "<|unk|>".into()];
        let vocab = corpus.to_vocab(additional_tokens.into());
        Self {
            str_to_int: vocab.clone(),
            int_to_str: vocab.iter().map(|(k, v)| (*v, k.clone())).collect(),
        }
    }
}

impl Tokenizer for SimpleTokenizerV2 {
    fn encode(&self, text: String) -> Vec<usize> {
        tokenize(&text)
            .iter()
            .map(|token| {
                *(self
                    .str_to_int
                    .get(*token)
                    .unwrap_or(&self.str_to_int["<|unk|>"]))
            })
            .collect()
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        let tokens: Vec<String> = ids.iter().map(|id| self.int_to_str[id].clone()).collect();
        let joined = tokens.join(" ");
        let regex = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();

        regex.replace_all(&joined, "$1").to_string()
    }
}

#[derive(Clone, Debug)]
struct GPTDatasetItem {
    input_ids: [usize; 4],
    target_ids: [usize; 4],
}

#[derive(Clone, Debug)]
struct GPTDatasetBatch<B: Backend> {
    input_ids: Tensor<B, 2, Int>,
    target_ids: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
struct GPTDatasetBatcher {}

impl<B: Backend> Batcher<B, GPTDatasetItem, GPTDatasetBatch<B>> for GPTDatasetBatcher {
    fn batch(
        &self,
        items: Vec<GPTDatasetItem>,
        device: &<B as Backend>::Device,
    ) -> GPTDatasetBatch<B> {
        let input_tensors: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| TensorData::from(item.input_ids).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 4]))
            .collect();

        let target_tensors: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| TensorData::from(item.target_ids).convert::<B::IntElem>())
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 4]))
            .collect();

        let input_ids = Tensor::cat(input_tensors, 0);
        let target_ids = Tensor::cat(target_tensors, 0);

        GPTDatasetBatch {
            input_ids,
            target_ids,
        }
    }
}

struct GPTDatasetV1 {
    dataset: InMemDataset<GPTDatasetItem>,
}

impl Dataset<GPTDatasetItem> for GPTDatasetV1 {
    fn get(&self, index: usize) -> Option<GPTDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl GPTDatasetV1 {
    fn new_from_text(
        txt: String,
        tokenizer: Box<dyn Tokenizer>,
        max_length: usize,
        stride: usize,
    ) -> Self {
        let token_ids = tokenizer.encode(txt);

        let mut input_ids: Vec<Vec<usize>> = Vec::new();
        let mut target_ids: Vec<Vec<usize>> = Vec::new();

        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = &token_ids[i..i + max_length];
            let target_chunk = &token_ids[i + 1..=(i + max_length)];

            input_ids.push(input_chunk.to_vec());
            target_ids.push(target_chunk.to_vec());
        }

        let zipped: Vec<_> = input_ids
            .into_iter()
            .zip(target_ids)
            .map(|(input_ids, target_ids)| GPTDatasetItem {
                input_ids: input_ids.try_into().unwrap(),
                target_ids: target_ids.try_into().unwrap(),
            })
            .collect();

        Self {
            dataset: InMemDataset::new(zipped),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs::{self, File},
        io,
    };

    use burn::data::dataset::Dataset;

    use crate::listings::ch02::{
        Corpus, GPTDatasetBatcher, GPTDatasetV1, SimpleTokenizerV1, SimpleTokenizerV2,
        THE_VERDICT_URL, Tokenizer, construct_vocab_from_url, text_from_url, tokenize,
    };

    use burn::data::dataloader::DataLoaderBuilder;

    #[test]
    fn test_simple_tokenizer_v2_special_tokens() {
        let tokenizer = SimpleTokenizerV2::new(Corpus::Url(THE_VERDICT_URL.to_string()));

        assert_eq!(tokenizer.str_to_int["younger"], 1127);
        assert_eq!(tokenizer.str_to_int["your"], 1128);
        assert_eq!(tokenizer.str_to_int["yourself"], 1129);
        assert_eq!(tokenizer.str_to_int["<|endoftext|>"], 1130);
        assert_eq!(tokenizer.str_to_int["<|unk|>"], 1131);
    }

    #[test]
    fn test_simple_tokenizer_v2_roundtrip() {
        let tokenizer = SimpleTokenizerV2::new(Corpus::Url(THE_VERDICT_URL.to_string()));

        let input = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.";

        let encoded = tokenizer.encode(input.to_string());

        assert_eq!(
            encoded,
            [
                1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7
            ],
        );

        let decoded = tokenizer.decode(encoded);

        assert_eq!(
            decoded,
            "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
        )
    }

    #[test]
    fn test_simple_tokenizer_v1_tokenize() {
        let tokenizer = SimpleTokenizerV1::new(
            construct_vocab_from_url(THE_VERDICT_URL.to_string(), None).unwrap(),
        );

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

        let mut res = client
            .get(THE_VERDICT_URL)
            .send()
            .unwrap()
            .error_for_status()
            .unwrap();
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
            THE_VERDICT_URL.to_string(),
            vec!["<|endoftext|>".to_string(), "<|unk|>".to_string()].into(),
        )
        .unwrap();

        assert_eq!(vocab.len(), 1132);

        assert_eq!(vocab["!"], 0);
        assert_eq!(vocab["\""], 1);
        assert_eq!(vocab["'"], 2);
        assert_eq!(vocab["Her"], 49);
        assert_eq!(vocab["Hermia"], 50);

        assert_eq!(vocab["younger"], 1127);
        assert_eq!(vocab["your"], 1128);
        assert_eq!(vocab["yourself"], 1129);

        assert_eq!(vocab["<|endoftext|>"], 1130);
        assert_eq!(vocab["<|unk|>"], 1131);
    }

    #[test]
    fn test_gpt2_tiktoken_encoder() {
        let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();

        let integers = tokenizer
            .encode(
                "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.",
                &HashSet::from(["<|endoftext|>"]),
            )
            .0;

        assert_eq!(
            integers,
            [
                15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114,
                286, 617, 34680, 27271, 13
            ],
        );

        assert_eq!(
            tokenizer.decode(integers).unwrap(),
            "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
        );

        let enc_text = tokenizer
            .encode(
                text_from_url(THE_VERDICT_URL.to_string()).unwrap().as_str(),
                &HashSet::new(),
            )
            .0;

        assert_eq!(enc_text.len(), 5145);

        let enc_sample = &enc_text[50..];

        let context_size = 4;
        assert_eq!(enc_sample[0..context_size], [290, 4920, 2241, 287]);
        assert_eq!(enc_sample[1..context_size + 1], [4920, 2241, 287, 257]);
    }

    #[test]
    fn test_gpt_v1_dataset() {
        // ref: https://github.com/tracel-ai/burn/blob/439a26c0ff35c8557e0105786e7ce0d2b74c2c4b/examples/custom-image-dataset/examples/custom-image-dataset.rs
        use burn::backend::NdArray;

        type Backend = NdArray;

        let dataset = GPTDatasetV1::new_from_text(
            text_from_url(THE_VERDICT_URL.to_string()).unwrap(),
            Box::new(SimpleTokenizerV2::new(Corpus::Url(
                THE_VERDICT_URL.to_string(),
            ))),
            4,
            1,
        );

        let item = dataset.get(0).unwrap();

        let enc_text = SimpleTokenizerV2::new(Corpus::Url(THE_VERDICT_URL.to_string()))
            .encode(text_from_url(THE_VERDICT_URL.to_string()).unwrap());

        assert_eq!(item.input_ids.to_vec(), enc_text[0..4]);

        let dataloader = DataLoaderBuilder::<Backend, _, _>::new(GPTDatasetBatcher {})
            .batch_size(4)
            // .shuffle(0)
            .num_workers(0)
            .build(GPTDatasetV1::new_from_text(
                text_from_url(THE_VERDICT_URL.to_string()).unwrap(),
                Box::new(SimpleTokenizerV2::new(Corpus::Url(
                    THE_VERDICT_URL.to_string(),
                ))),
                4,
                1,
            ));

        let batch = dataloader.iter().next().unwrap();

        println!("{:?}", batch);
        println!("{:?}", item.input_ids.to_vec());

        assert_eq!(batch.input_ids.shape().dims, [4, 4]);
        assert_eq!(batch.target_ids.shape().dims, [4, 4]);
    }
}
