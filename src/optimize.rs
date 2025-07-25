use std::collections::HashMap;

use crate::{
  scalar::Real,
  tensor::Tensor,
  variable::Variable,
  ops::{ NonOps, NumericHops, RealHops },
};


/// An optimization strategy to be used with [Optimizer].

pub trait Strategy<R: Real> {
  fn update(&mut self, param: &Variable<R>, rate: R, step: usize) -> Tensor<R>;
}


/// Generic optimizer that allows for several optimization [strategies](Strategy) to be used.

#[derive(Debug)]
pub struct Optimizer<R: Real, S: Strategy<R>> {
  strategy: S,
  pub learning_rate: R,
  step: usize,
}

impl<R: Real, S: Strategy<R>> Optimizer<R, S> {
  pub fn new(learning_rate: R, strategy: S) -> Self {
    Self { strategy, learning_rate, step: 1 }
  }

  pub fn minimize(&mut self, loss: &Variable<R>, params: Vec<Variable<R>>, reset: bool) {
    // Compute gradients
    if reset { loss.backward() }

    // Optimize individual parameters
    for mut param in params {
      param.grad().expect("Non-trainable parameters cannot be optimized");

      // Execute strategy
      let change = self.strategy.update(&param, self.learning_rate, self.step);

      // Apply change
      param += change;
    }

    // Reset gradients
    if reset { loss.reset() }

    self.step += 1;
  }
}


/// Stochastic Gradient Descent strategy

#[derive(Debug, Clone, Default)]
pub struct SGD;

impl<R: Real> Strategy<R> for SGD {
  fn update(&mut self, param: &Variable<R>, rate: R, _step: usize) -> Tensor<R> {
    param.grad().unwrap() * -rate
  }
}


/// Stochastic Gradient Descent with momentum

#[derive(Debug, Clone)]
pub struct Momentum<R: Real> {
  pub momentum: R,
  v: HashMap<usize, Tensor<R>>,
}

impl<R: Real> Momentum<R> {
  pub fn new(momentum: R) -> Self {
    Self {
      momentum,
      v: HashMap::new(),
    }
  }
}

impl<R: Real> Default for Momentum<R> {
  fn default() -> Self {
    Self::new(R::from(0.9).unwrap())
  }
}

impl<R: Real> Strategy<R> for Momentum<R> {
  fn update(&mut self, param: &Variable<R>, rate: R, _step: usize) -> Tensor<R> {
    let id = param.id();
    let weights = param.tensor();
    let grad = param.grad().unwrap();
    if self.v.get(&id).is_none() {
      let shape = &weights.shape().dims;
      self.v.insert(id, Tensor::zeros(shape));
    }
    let v = self.v.get(&id).unwrap();
    v.assign(&(v * self.momentum - grad * rate));
    v.clone()
  }
}


/// Stochastic Gradient Descent with Nesterov momentum

#[derive(Debug, Clone)]
pub struct Nesterov<R: Real> {
  pub momentum: R,
  v: HashMap<usize, Tensor<R>>,
  v_prev: HashMap<usize, Tensor<R>>,
}

impl<R: Real> Nesterov<R> {
  pub fn new(momentum: R) -> Self {
    Self {
      momentum,
      v: HashMap::new(),
      v_prev: HashMap::new(),
    }
  }
}

impl<R: Real> Default for Nesterov<R> {
  fn default() -> Self {
    Self::new(R::from(0.9).unwrap())
  }
}

impl<R: Real> Strategy<R> for Nesterov<R> {
  fn update(&mut self, param: &Variable<R>, rate: R, _step: usize) -> Tensor<R> {
    let id = param.id();
    let weights = param.tensor();
    let grad = param.grad().unwrap();
    if self.v.get(&id).is_none() {
      let shape = &weights.shape().dims;
      self.v.insert(id, Tensor::zeros(shape));
      self.v_prev.insert(id, Tensor::zeros(shape));
    }
    let v = self.v.get(&id).unwrap();
    let v_prev = self.v_prev.get(&id).unwrap();
    v_prev.assign(&v);
    v.assign(&(v * self.momentum - grad * rate));
    v_prev * -self.momentum + v * (R::one() + self.momentum)
  }
}


/// Adaptive Movement Estimation strategy (ADAM)

#[derive(Debug, Clone)]
pub struct Adam<R: Real> {
  pub beta1: R,
  pub beta2: R,
  m: HashMap<usize, Tensor<R>>,
  v: HashMap<usize, Tensor<R>>,
}

impl<R: Real> Adam<R> {
  pub fn new(beta1: R, beta2: R) -> Self {
    Self {
      beta1,
      beta2,
      m: HashMap::new(),
      v: HashMap::new(),
    }
  }
}

impl<R: Real> Default for Adam<R> {
  fn default() -> Self {
    Self::new(R::from(0.9).unwrap(), R::from(0.999).unwrap())
  }
}

impl<R: Real> Strategy<R> for Adam<R> {
  fn update(&mut self, param: &Variable<R>, rate: R, step: usize) -> Tensor<R> {
    let id = param.id();
    let weights = param.tensor();
    let grad = param.grad().unwrap();
    if self.m.get(&id).is_none() {
      let shape = &weights.shape().dims;
      self.m.insert(id, Tensor::zeros(shape));
      self.v.insert(id, Tensor::zeros(shape));
    }
    let m = self.m.get(&id).unwrap();
    let v = self.v.get(&id).unwrap();
    m.assign(&(m * self.beta1 + grad                             * (R::one() - self.beta1)));
    v.assign(&(v * self.beta2 + grad.powf(R::from(2.0).unwrap()) * (R::one() - self.beta2)));
    let step = R::from(step).unwrap();
    let mt = m / (R::one() - self.beta1.powf(step));
    let vt = v / (R::one() - self.beta2.powf(step));
    mt * -rate / (vt.sqrt() + R::from(1e-8).unwrap())
  }
}


/// Adaptive Movement Estimation strategy
/// with decoupled weight decay

#[derive(Debug, Clone)]
pub struct AdamW<R: Real> {
  pub beta1: R,
  pub beta2: R,
  pub weight_decay: R,
  m: HashMap<usize, Tensor<R>>,
  v: HashMap<usize, Tensor<R>>,
}

impl<R: Real> AdamW<R> {
  pub fn new(beta1: R, beta2: R, weight_decay: R) -> Self {
    Self {
      beta1,
      beta2,
      weight_decay,
      m: HashMap::new(),
      v: HashMap::new(),
    }
  }
}

impl<R: Real> Default for AdamW<R> {
  fn default() -> Self {
    Self::new(
      R::from(0.9).unwrap(),
      R::from(0.999).unwrap(),
      R::from(0.01).unwrap(),
    )
  }
}

impl<R: Real> Strategy<R> for AdamW<R> {
  fn update(&mut self, param: &Variable<R>, rate: R, step: usize) -> Tensor<R> {
    let id = param.id();
    let weights = param.tensor();
    let grad = param.grad().unwrap();

    if self.m.get(&id).is_none() {
      let shape = &weights.shape().dims;
      self.m.insert(id, Tensor::zeros(shape));
      self.v.insert(id, Tensor::zeros(shape));
    }

    let m = self.m.get_mut(&id).unwrap();
    let v = self.v.get_mut(&id).unwrap();

    m.assign(&(m.clone() * self.beta1 + grad * (R::one() - self.beta1)));
    v.assign(&(v.clone() * self.beta2 + grad.powf(R::from(2.0).unwrap()) * (R::one() - self.beta2)));

    let step = R::from(step).unwrap();
    let mt = m.clone() / (R::one() - self.beta1.powf(step));
    let vt = v.clone() / (R::one() - self.beta2.powf(step));

    // Standard Adam update
    let adam_update = mt * -rate / (vt.sqrt() + R::from(1e-8).unwrap());

    // Decoupled weight decay
    let decay = weights.clone() * -(self.weight_decay * rate);

    adam_update + decay
  }
}
