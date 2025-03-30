mod model;
mod preprocess;

use model::{Encoder, Decoder, Seq2Seq};
use preprocess::{load_data, Vocab, TranslationData, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN};
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use std::error::Error;
use clap::Parser;
use std::ffi::CString;
use libc::dlopen;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    data_path: String,

    #[arg(long, default_value_t = 2)]
    batch_size: usize,

    #[arg(long, default_value_t = 5)]
    epochs: i32,

    #[arg(long, default_value_t = 12)]
    max_len: i64,

    #[arg(long, default_value_t = 32)]
    emb_dim: i64,

    #[arg(long, default_value_t = 64)]
    hid_dim: i64,

    #[arg(long, default_value_t = 1)]
    n_layers: i64,

    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = String::from("torch/lib/libtorch_cuda.so"))]
    cuda_location: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let path = CString::new(args.cuda_location).unwrap();
    unsafe {
        dlopen(path.into_raw(), 1);
    }

    println!("cuda: {}", tch::Cuda::is_available());
    println!("cudnn: {}", tch::Cuda::cudnn_is_available());

    let device = if tch::Cuda::is_available() {
        println!("Using GPU (CUDA).");
        Device::Cuda(0)
    } else {
        println!("Using CPU.");
        Device::Cpu
    };

    println!("Selected device: {:?}", device);

    let data_path = args.data_path;
    let max_len = args.max_len;
    let batch_size = args.batch_size;

    let data: TranslationData = load_data(&data_path, max_len as usize);
    
    println!("Data loaded. Samples: {}", data.src_tensors.len());
    println!(
        "Source vocab size: {}, Target vocab size: {}",
        data.src_vocab.token_to_id.len(),
        data.trg_vocab.token_to_id.len()
    );
    
    let total_samples = data.src_tensors.len();
    let train_size = (total_samples as f64 * 0.8).round() as usize;
    let train_src_tensors: Vec<_> = data.src_tensors[..train_size]
    .iter()
    .map(|t| t.shallow_clone())
    .collect();
    let train_trg_tensors: Vec<_> = data.trg_tensors[..train_size]
    .iter()
    .map(|t| t.shallow_clone())
    .collect();
    let test_src_tensors: Vec<_> = data.src_tensors[train_size..]
    .iter()
    .map(|t| t.shallow_clone())
    .collect();
    let test_trg_tensors: Vec<_> = data.trg_tensors[train_size..]
    .iter()
    .map(|t| t.shallow_clone())
    .collect();

    println!(
        "Data split: {} train samples and {} test samples",
        train_src_tensors.len(),
        test_src_tensors.len()
    );

    let input_dim = data.src_vocab.token_to_id.len() as i64;
    let output_dim = data.trg_vocab.token_to_id.len() as i64;
    let emb_dim = args.emb_dim;
    let hid_dim = args.hid_dim;
    let n_layers = args.n_layers;
    let learning_rate = args.learning_rate;
    let epochs = args.epochs;

    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let encoder = Encoder::new(root, input_dim, emb_dim, hid_dim, n_layers, true);
    let decoder = Decoder::new(root, output_dim, emb_dim, hid_dim, n_layers);
    let model = Seq2Seq::new(encoder, decoder, device);
    println!("Model initialized.");

    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    for epoch in 1..=epochs {
        println!("\nEpoch {}/{}", epoch, epochs);
        let num_samples = train_src_tensors.len();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut i = 0;

        while i < num_samples {
            let end = (i + batch_size).min(num_samples);
            let batch_src = Tensor::stack(&train_src_tensors[i..end], 0).to_device(device);
            let batch_trg = Tensor::stack(&train_trg_tensors[i..end], 0).to_device(device);
            opt.zero_grad();
            let logits = model.forward(&batch_src, &batch_trg, 0.5);
            let vocab_size = logits.size()[2];

            let logits_flat = logits
                .slice(1, 1, logits.size()[1], 1)
                .reshape(&[-1, vocab_size]);
            let trg_flat = batch_trg
                .slice(1, 1, batch_trg.size()[1], 1)
                .reshape(&[-1]);
            let loss = logits_flat.cross_entropy_for_logits(&trg_flat);
            opt.backward_step(&loss);
            epoch_loss += loss.double_value(&[]);
            batch_count += 1;
            i = end;
        }
        let avg_loss = epoch_loss / batch_count as f64;
        println!("Training Loss: {:.4}", avg_loss);
    }

    let mut total_loss = 0.0;
    let mut count = 0;

    let num_test_samples = test_src_tensors.len();
    let mut i = 0;
    
    while i < num_test_samples {
        let end = (i + batch_size).min(num_test_samples);
        let batch_src = Tensor::stack(&test_src_tensors[i..end], 0).to_device(device);
        let batch_trg = Tensor::stack(&test_trg_tensors[i..end], 0).to_device(device);
        let logits = model.forward(&batch_src, &batch_trg, 0.0);
        let vocab_size = logits.size()[2];
    
        let logits_flat = logits
            .slice(1, 1, logits.size()[1], 1)
            .reshape(&[-1, vocab_size]);
        let trg_flat = batch_trg
            .slice(1, 1, batch_trg.size()[1], 1)
            .reshape(&[-1]);
        let loss = logits_flat.cross_entropy_for_logits(&trg_flat);
        total_loss += loss.double_value(&[]);
        count += 1;
        i = end;
    }
    
    let avg_loss = total_loss / count as f64;
    let perplexity = avg_loss.exp();
    println!("Test Loss: {:.4}, Perplexity: {:.4}", avg_loss, perplexity);
    
    if !data.src_tensors.is_empty() {
        println!("\nSample translations:");
        let samples_to_show = 3.min(data.src_tensors.len());
        
        for i in 0..samples_to_show {
            let src = &data.src_tensors[i];
            let trg = &data.trg_tensors[i];
            
            let src_batch = src.unsqueeze(0).to_device(device);
            let trg_batch = trg.unsqueeze(0).to_device(device);
            
            let logits = model.forward(&src_batch, &trg_batch, 0.0);
            
            let pred_tokens = logits.select(0, 0).argmax(-1, false);
            
            let src_words = decode_tokens(src, &data.src_vocab);
            let trg_words = decode_tokens(trg, &data.trg_vocab);
            let pred_words = decode_tokens(&pred_tokens, &data.trg_vocab);
            
            println!("Source: {}", src_words.join(" "));
            println!("Target: {}", trg_words.join(" "));
            println!("Predicted: {}\n", pred_words.join(" "));
            
            let vocab_size = logits.size()[2];
            let logits_flat = logits.slice(1, 1, logits.size()[1], 1).reshape(&[-1, vocab_size]);
            let trg_flat = trg_batch.slice(1, 1, trg_batch.size()[1], 1).reshape(&[-1]);
            let loss = logits_flat.cross_entropy_for_logits(&trg_flat);
            total_loss += loss.double_value(&[]);
            count += 1;
        }
    }
    
    for (src, trg) in data.src_tensors.iter().zip(data.trg_tensors.iter()).skip(3) {
        let src_batch = src.unsqueeze(0).to_device(device);
        let trg_batch = trg.unsqueeze(0).to_device(device);
        let logits = model.forward(&src_batch, &trg_batch, 0.0);
        let vocab_size = logits.size()[2];
        let logits_flat = logits.slice(1, 1, logits.size()[1], 1).reshape(&[-1, vocab_size]);
        let trg_flat = trg_batch.slice(1, 1, trg_batch.size()[1], 1).reshape(&[-1]);
        let loss = logits_flat.cross_entropy_for_logits(&trg_flat);
        total_loss += loss.double_value(&[]);
        count += 1;
    }
    
    let avg_loss_eval = total_loss / count as f64;
    let perplexity = avg_loss_eval.exp();
    println!("Evaluation - Loss: {:.4}, Perplexity: {:.4}", avg_loss_eval, perplexity);

    Ok(())
}

fn decode_tokens(tensor: &Tensor, vocab: &Vocab) -> Vec<String> {
    let id_to_token = &vocab.id_to_token;
    let token_ids: Vec<i64> = tensor.iter::<i64>().unwrap().collect();
    let special_tokens = vec![
        *vocab.token_to_id.get(SOS_TOKEN).unwrap(),
        *vocab.token_to_id.get(EOS_TOKEN).unwrap(),
        *vocab.token_to_id.get(PAD_TOKEN).unwrap(),
    ];
    
    token_ids
        .into_iter()
        .filter(|id| !special_tokens.contains(id))
        .map(|id| id_to_token[id as usize].clone())
        .collect()
}
