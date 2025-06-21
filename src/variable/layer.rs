use std::sync::Arc;

use parking_lot::{ RwLock };
use serde::{ Serialize, de::DeserializeOwned };

use crate::{ ops::*, scalar::Real, Variable, Tensor, Module };


pub trait Layer<I: Real> {
  fn retrained(&self, generator: impl Fn() -> Tensor<I>) -> Self;
  fn dense(&self, size: usize) -> Self;
  fn dense_biased(&self, size: usize, bias_init: I) -> Self;
  fn dense_shared(&self, size: usize, weights: Option<Self>) -> (Self, Self) where Self: Sized;
  fn dense_taped(&self, size: usize, tape_holder: &Self) -> Self;
  fn dense_full(&self, size: usize, weights: Option<Self>, tape_holder: Option<&Self>, bias_init: I) -> (Self, Self) where Self: Sized;
  fn conv2d(&self, out_channels: usize, kernel_shape: [usize; 2], pad: bool) -> Self;
  fn layernorm(&self, num_dims: usize, scale: bool) -> Self;
  fn dropout(&self, probability: I, train: bool) -> Self;
  fn embed(&self, vocab_len: usize, embed_dim: usize) -> Self;
  fn embed_shared(&self, vocab_len: usize, embed_dim: usize, downscale: bool) -> (Self, Self) where Self: Sized;
  fn multi_head_attention(&self, key_value: &Self, mask: &Self, num_heads: usize) -> Self;
  fn lstm(&self, num_layers: usize, hidden_size: usize, output_size: usize, full: bool) -> Self;
  fn lstm_block(&self, num_layers: usize, hidden_size: usize, output_size: usize, hidden: Arc<RwLock<Vec<(Self, Self)>>>) -> Self where Self: Sized;
  fn lstm_cell(&self, hidden_size: usize, hidden: &(Self, Self)) -> (Self, Self) where Self: Sized;
}

impl<I: Real + Serialize + DeserializeOwned> Layer<I> for Variable<I> {
  fn retrained(&self, generator: impl Fn() -> Tensor<I>) -> Self {
    let tape = self.node.traintape.as_ref()
      .expect("Called #retrained on a Variable that wasn't generated from Module inputs");
    Self::from_tape(true, tape, generator)
  }

  fn dense(&self, size: usize) -> Self {
    self.dense_shared(size, None).0
  }

  fn dense_biased(&self, size: usize, bias_init: I) -> Self {
    self.dense_full(size, None, None, bias_init).0
  }

  fn dense_shared(&self, size: usize, weights: Option<Self>) -> (Self, Self) {
    self.dense_full(size, weights, None, I::zero())
  }

  fn dense_taped(&self, size: usize, tape_holder: &Self) -> Self {
    self.dense_full(size, None, Some(tape_holder), I::zero()).0
  }

  fn dense_full(&self, size: usize, weights: Option<Self>, tape_holder: Option<&Self>, bias_init: I) -> (Self, Self) {
    let tape_holder = tape_holder.or_else(|| Some(self) ).unwrap();
    let weights = weights.or_else(|| Some(tape_holder.retrained(|| {
      let dims = [self.dim(-1), size];
      Tensor::glorot_uniform(&dims)
    }))).unwrap();
    let bias = tape_holder.retrained(|| Tensor::fill(&[size], bias_init) );
    (self.mm(&weights) + bias, weights)
  }

  fn conv2d(&self, out_channels: usize, kernel_shape: [usize; 2], pad: bool) -> Self {
    let kernels = self.retrained(|| {
      let size = kernel_shape[0] * kernel_shape[1] * (out_channels + self.dim(-3)) / 2;
      let gain = (1.0 / size as f64).sqrt();
      Tensor::randn(&[out_channels, self.dim(-3), kernel_shape[0], kernel_shape[1]]) * I::from(gain).unwrap()
    });
    let bias = self.retrained(|| Tensor::zeros(&[out_channels]) );
    self.convolve2d(&kernels, [1,1], Some(&bias), pad)
  }

  fn layernorm(&self, num_dims: usize, scale: bool) -> Self {
    let num_dims = num_dims as isize;
    let mean = self.mean(-num_dims).unsqueeze_n(num_dims as usize, -1);
    let variance = self.variance(-num_dims).unsqueeze_n(num_dims as usize, -1);
    let normalized = (self - mean) / (variance + I::from(1e-5).unwrap()).sqrt();
    let dims = &self.shape()[-num_dims..-1];
    (if scale {
      normalized * self.retrained(|| Tensor::ones(&dims) )
    } else {
      normalized
    }) + self.retrained(|| Tensor::zeros(&dims) )
  }

  fn dropout(&self, probability: I, train: bool) -> Self {
    if !train { return self.clone() }
    let probability = I::one() - probability;
    let bernoulli = Tensor::fill(&self.shape().dims, probability).bernoulli() / probability;
    self * bernoulli.tracked()
  }

  fn embed(&self, vocab_len: usize, embed_dim: usize) -> Self {
    self.embed_shared(vocab_len, embed_dim, false).0
  }

  fn embed_shared(&self, vocab_len: usize, embed_dim: usize, downscale: bool) -> (Self, Self) {
    let table = self.retrained(|| {
      let gain = I::from(if downscale { (1.0 / embed_dim as f32).sqrt() } else { 1.0 }).unwrap();
      Tensor::randn(&[vocab_len, embed_dim]) * gain
    });
    (table.look_up(self), table)
  }

  fn multi_head_attention(&self, key_value: &Self, mask: &Self, num_heads: usize) -> Self {
    let dim_model = self.dim(-1);
    let head_dim = dim_model / num_heads;

    let transpose = |seq: &Self| {
      seq
      .dense(seq.dim(-1))
      .reshape_keep(&[-1, 0, num_heads as isize, head_dim as isize])
      .transpose(1,2)
    };

    let query = transpose(self);
    let key   = transpose(key_value);
    let value = transpose(key_value);

    let scores = query.attention(&key, &value, head_dim, mask);
    let concat = scores
      .transpose(1,2)
      .reshape_keep(&[-1, 0, dim_model as isize]);

    concat.dense(dim_model)
  }

  fn lstm(&self, num_layers: usize, hidden_size: usize, output_size: usize, full: bool) -> Self {
    let hidden = Arc::new(RwLock::new((0..num_layers).map(|l| {
      let size = if l == num_layers - 1 { output_size } else { hidden_size };
      let dims = [&self.shape()[0..-3], &[size]].concat();
      let zeros = Tensor::zeros(&dims).tracked();
      (zeros.clone(), zeros.clone())
    }).collect()));

    let block = Module::continued(self, move |inputs| vec![
      inputs[0]
      .lstm_block(num_layers, hidden_size, output_size, hidden.clone())
    ]);

    let output: Vec<_> = (0..self.dim(1)).map(|t| {
      let input = self.select(-2, t as isize);
      block.run_raw(0, &[&input])
    }).collect();

    if full {
      Variable::rows(&output).transpose(0, -2)
    } else {
      output
        .last().unwrap()
        .squeeze_all()
    }
  }

  fn lstm_block(&self, num_layers: usize, hidden_size: usize, output_size: usize, hidden: Arc<RwLock<Vec<(Self, Self)>>>) -> Self {
    let mut hidden = hidden.write();
    for l in 0..num_layers {
      let prev = if l == 0 { self } else { &hidden[l - 1].0 };
      let size = if l == num_layers - 1 { output_size } else { hidden_size };
      hidden[l] = prev.lstm_cell(size, &hidden[l]);
    }
    hidden.last().unwrap().0.clone()
  }

  fn lstm_cell(&self, hidden_size: usize, state: &(Self, Self)) -> (Self, Self) {
    let (hx, cx) = state;

    let size = hidden_size * 4;
    let gates = self.dense(size) + hx.dense_taped(size, self);
    let [input_gate, forget_gate, cell_input, output_gate] = gates.chunks(4, -1).try_into().unwrap();

    let cy = cx * forget_gate.sigmoid() + input_gate.sigmoid() * cell_input.tanh();
    let hy = output_gate.sigmoid() * cy.tanh();

    (hy, cy)
  }
}
