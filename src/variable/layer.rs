use std::sync::Arc;

use parking_lot::{ RwLock };
use serde::{ Serialize, de::DeserializeOwned };

use crate::{ ops::*, scalar::Real, Variable, Tensor, Module };


pub trait Layer<I: Real> {
  fn retrained(&self, generator: impl Fn() -> Tensor<I>) -> Self;
  fn dense(&self, size: usize) -> Self;
  fn dense_shared(&self, size: usize, weights: Option<Self>) -> (Self, Self) where Self: Sized;
  fn dense_taped(&self, size: usize, tape_holder: &Self) -> Self;
  fn dense_shared_taped(&self, size: usize, weights: Option<Self>, tape_holder: Option<&Self>) -> (Self, Self) where Self: Sized;
  fn conv2d(&self, out_channels: usize, kernel_shape: [usize; 2], pad: bool) -> Self;
  fn lstm_cell(&self, hidden: &(Self, Self)) -> (Self, Self) where Self: Sized;
  fn lstm_block(&self, num_layers: usize, hidden: Arc<RwLock<Vec<(Self, Self)>>>) -> Self where Self: Sized;
  fn lstm(&self, num_layers: usize, hidden_size: usize, full: bool) -> Self;
  fn layernorm(&self, num_dims: usize, scale: bool) -> Self;
  fn dropout(&self, probability: I, train: bool) -> Self;
  fn embed(&self, vocab_len: usize, embed_dim: usize) -> Self;
  fn embed_shared(&self, vocab_len: usize, embed_dim: usize, downscale: bool) -> (Self, Self) where Self: Sized;
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

  fn dense_shared(&self, size: usize, weights: Option<Self>) -> (Self, Self) {
    self.dense_shared_taped(size, weights, None)
  }

  fn dense_taped(&self, size: usize, tape_holder: &Self) -> Self {
    self.dense_shared_taped(size, None, Some(tape_holder)).0
  }

  fn dense_shared_taped(&self, size: usize, weights: Option<Self>, tape_holder: Option<&Self>) -> (Self, Self) {
    let tape_holder = tape_holder.or_else(|| Some(self) ).unwrap();
    let weights = weights.or_else(|| Some(tape_holder.retrained(|| {
      let dims = [self.dim(-1), size];
      let gain = I::from((2.0 / (dims[0] + dims[1]) as f64).sqrt()).unwrap();
      Tensor::randn(&dims) * gain
    }))).unwrap();
    let bias = tape_holder.retrained(|| Tensor::zeros(&[size]) );
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

  fn lstm(&self, num_layers: usize, hidden_size: usize, full: bool) -> Self {
    let dims = [&self.shape()[0..-3], &[hidden_size]].concat();
    let zeros = Tensor::zeros(&dims).tracked();
    let hidden = Arc::new(RwLock::new((0..num_layers).map(|_|
      (zeros.clone(), zeros.clone())
    ).collect()));

    let block = Module::continued(self, move |inputs| vec![
      inputs[0].lstm_block(num_layers, hidden.clone())
    ]);

    let output: Vec<_> = (0..self.dim(1)).map(|t| {
      let t = t as isize;
      let ranges = [vec![0..-1; self.rank() - 2], vec![t..t + 1]].concat();
      let input = self.range(&ranges).squeeze(&[1]);
      block.run_raw(0, &[&input, &zeros])
    }).collect();

    if full {
      Variable::rows(&output).transpose(0, -2)
    } else {
      output
        .last().unwrap()
        .squeeze_all()
    }
  }

  fn lstm_block(&self, num_layers: usize, hidden: Arc<RwLock<Vec<(Self, Self)>>>) -> Self {
    let mut hidden = hidden.write();
    hidden[0] = self.lstm_cell(&hidden[0]);
    for l in 1..num_layers {
      hidden[l] = hidden[l - 1].0.lstm_cell(&hidden[l]);
    }
    hidden.last().unwrap().0.clone()
  }

  fn lstm_cell(&self, state: &(Self, Self)) -> (Self, Self) {
    let (hx, cx) = state;

    let size = hx.dim(-1) * 4;
    let gates = self.dense(size) + hx.dense_taped(size, self);
    let [input_gate, forget_gate, cell_gate, output_gate] = gates.chunks(4, -1).try_into().unwrap();

    let cy = cx * forget_gate.sigmoid() + input_gate.sigmoid() * cell_gate.tanh();
    let hy = output_gate.sigmoid() * cy.tanh();

    (hy, cy)
  }
}
