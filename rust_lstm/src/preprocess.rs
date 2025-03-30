use std::collections::HashMap;
use std::fs;
use tch::{Tensor, Kind};

pub const PAD_TOKEN: &str = "<pad>";
pub const SOS_TOKEN: &str = "<sos>";
pub const EOS_TOKEN: &str = "<eos>";


pub struct Vocab {
    pub token_to_id: HashMap<String, i64>,
    pub id_to_token: Vec<String>,
}


impl Vocab {
    pub fn new() -> Self {
        Vocab {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }
    
    pub fn register_token(&mut self, token: &str) {
        if !self.token_to_id.contains_key(token) {
            let idx = self.id_to_token.len() as i64;
            self.token_to_id.insert(token.to_string(), idx);
            self.id_to_token.push(token.to_string());
        }
    }
}


pub struct TranslationData {
    pub src_tensors: Vec<Tensor>,
    pub trg_tensors: Vec<Tensor>,
    pub src_vocab: Vocab,
    pub trg_vocab: Vocab,
}


pub fn load_data(data_path: &str, max_len: usize) -> TranslationData {
    let content = fs::read_to_string(data_path).expect("Failed to read data file");
    let mut src_sentences: Vec<Vec<String>> = Vec::new();
    let mut trg_sentences: Vec<Vec<String>> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 2 {
            println!("Skipping malformed line: {}", line);
            continue;
        }
        
        let src: Vec<String> = parts[0]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let trg: Vec<String> = parts[1]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
            
        src_sentences.push(src);
        trg_sentences.push(trg);
    }

    let mut src_vocab = Vocab::new();
    let mut trg_vocab = Vocab::new();

    for token in &[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] {
        src_vocab.register_token(token);
        trg_vocab.register_token(token);
    }
    
    for sentence in &src_sentences {
        for token in sentence {
            src_vocab.register_token(token);
        }
    }
    
    for sentence in &trg_sentences {
        for token in sentence {
            trg_vocab.register_token(token);
        }
    }

    let mut src_tensors = Vec::new();
    let mut trg_tensors = Vec::new();

    for (src, trg) in src_sentences.iter().zip(trg_sentences.iter()) {
        let src_ids = encode_sentence(src, &src_vocab, max_len);
        let trg_ids = encode_sentence(trg, &trg_vocab, max_len);
        
        let src_tensor = Tensor::f_from_slice(&src_ids).unwrap().to_kind(Kind::Int64);
        let trg_tensor = Tensor::f_from_slice(&trg_ids).unwrap().to_kind(Kind::Int64);
        
        src_tensors.push(src_tensor);
        trg_tensors.push(trg_tensor);
    }

    TranslationData {
        src_tensors,
        trg_tensors,
        src_vocab,
        trg_vocab,
    }
}


fn encode_sentence(sentence: &Vec<String>, vocab: &Vocab, max_len: usize) -> Vec<i64> {
    let sos = *vocab.token_to_id.get(SOS_TOKEN).unwrap();
    let eos = *vocab.token_to_id.get(EOS_TOKEN).unwrap();
    let pad = *vocab.token_to_id.get(PAD_TOKEN).unwrap();
    
    let mut ids = Vec::new();
    ids.push(sos);
    
    for token in sentence {
        if let Some(&id) = vocab.token_to_id.get(token) {
            ids.push(id);
        }
    }
    
    ids.push(eos);
    
    if ids.len() > max_len {
        ids.truncate(max_len);
    } else {
        while ids.len() < max_len {
            ids.push(pad);
        }
    }
    
    ids
}
