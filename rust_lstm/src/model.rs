use tch::{nn, Tensor, Device, Kind};
use tch::nn::{Embedding, Linear, LSTM, RNNConfig, LSTMState, RNN};


pub struct Encoder {
    pub embedding: Embedding,
    pub lstm: LSTM,
    pub n_layers: i64,
    pub _hid_dim: i64,
    pub bidirectional: bool,
}


impl Encoder {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        emb_dim: i64,
        hid_dim: i64,
        n_layers: i64,
        bidirectional: bool,
    ) -> Self {
        let embed_cfg = nn::EmbeddingConfig {
            padding_idx: 0,
            ..Default::default()
        };
        let embedding = nn::embedding(vs, input_dim, emb_dim, embed_cfg);

        let lstm_cfg = RNNConfig {
            num_layers: n_layers,
            bidirectional,
            batch_first: true,
            ..Default::default()
        };
        let lstm = nn::lstm(vs, emb_dim, hid_dim, lstm_cfg);

        Encoder {
            embedding,
            lstm,
            n_layers,
            _hid_dim: hid_dim,
            bidirectional,
        }
    }


    pub fn forward(&self, src: &Tensor) -> (Tensor, Tensor) {
        let embedded = src.apply(&self.embedding);

        let batch_size = embedded.size()[0];
        let init_state = self.lstm.zero_state(batch_size);

        let (_outputs, LSTMState((h, c))) = self.lstm.seq_init(&embedded, &init_state);

        if self.bidirectional {
            let num_dirs = 2;
            let h_forward = h.slice(0, 0, self.n_layers, 1);
            let h_backward = h.slice(0, self.n_layers, self.n_layers * num_dirs, 1);
            let c_forward = c.slice(0, 0, self.n_layers, 1);
            let c_backward = c.slice(0, self.n_layers, self.n_layers * num_dirs, 1);

            let h_combined = h_forward + h_backward;
            let c_combined = c_forward + c_backward;
            (h_combined, c_combined)
        } else {
            (h, c)
        }
    }
}


pub struct Decoder {
    pub embedding: Embedding,
    pub lstm: LSTM,
    pub fc_out: Linear,
    pub _n_layers: i64,
    pub _hid_dim: i64,
}


impl Decoder {
    pub fn new(
        vs: &nn::Path,
        output_dim: i64,
        emb_dim: i64,
        hid_dim: i64,
        n_layers: i64,
    ) -> Self {
        let embed_cfg = nn::EmbeddingConfig {
            padding_idx: 0,
            ..Default::default()
        };
        let embedding = nn::embedding(vs, output_dim, emb_dim, embed_cfg);

        let lstm_cfg = RNNConfig {
            num_layers: n_layers,
            bidirectional: false,
            batch_first: true,
            ..Default::default()
        };
        let lstm = nn::lstm(vs, emb_dim, hid_dim, lstm_cfg);

        let fc_out = nn::linear(vs, hid_dim, output_dim, Default::default());

        Decoder {
            embedding,
            lstm,
            fc_out,
            _n_layers: n_layers,
            _hid_dim: hid_dim,
        }
    }


    pub fn forward_step(
        &self,
        input_step: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let input_step = input_step.unsqueeze(1);
        let embedded = input_step.apply(&self.embedding);

        let (out, LSTMState((h_new, c_new))) =
            self.lstm.seq_init(&embedded, &LSTMState((h.shallow_clone(), c.shallow_clone())));
        let logits = out.squeeze_dim(1).apply(&self.fc_out);

        (logits, h_new, c_new)
    }
}


pub struct Seq2Seq {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub device: Device,
}


impl Seq2Seq {
    pub fn new(encoder: Encoder, decoder: Decoder, device: Device) -> Self {
        Seq2Seq {
            encoder,
            decoder,
            device,
        }
    }


    pub fn forward(
        &self,
        src: &Tensor,
        trg: &Tensor,
        teacher_forcing_ratio: f64,
    ) -> Tensor {
        let (mut h, mut c) = self.encoder.forward(src);

        let batch_size = src.size()[0];
        let trg_len = trg.size()[1];
        let vocab_size = self.decoder.fc_out.ws.size()[0];
        let outputs = Tensor::zeros(&[batch_size, trg_len, vocab_size], (Kind::Float, self.device));

        let mut input_step = trg.slice(1, 0, 1, 1).squeeze_dim(1);

        for t in 1..trg_len {
            let (logits, h_new, c_new) = self.decoder.forward_step(&input_step, &h, &c);
            h = h_new;
            c = c_new;

            outputs.narrow(1, t, 1).copy_(&logits.unsqueeze(1));

            let use_teacher = rand::random::<f64>() < teacher_forcing_ratio;
            if use_teacher {
                input_step = trg.slice(1, t, t + 1, 1).squeeze_dim(1);
            } else {
                input_step = logits.argmax(-1, false);
            }
        }
        outputs
    }
}
