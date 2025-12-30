use std::collections::HashSet;

use tiktoken_rs::CoreBPE;

use crate::listings::ch02::Tokenizer;

pub struct UnsafeBPETokenizer {
    tokenizer: CoreBPE,
}

impl UnsafeBPETokenizer {
    pub fn new(model_name: &str) -> Self {
        Self {
            tokenizer: tiktoken_rs::get_bpe_from_model(model_name).unwrap(),
        }
    }
}

impl Tokenizer for UnsafeBPETokenizer {
    fn encode(&self, text: String) -> Vec<usize> {
        self.tokenizer
            .encode(text.as_str(), &HashSet::new())
            .0
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.tokenizer
            .decode(ids.into_iter().map(|x| x as u32).collect())
            .unwrap()
    }
}
