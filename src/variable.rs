use std::collections::HashSet;
use std::sync::{ Mutex, OnceLock };
use std::sync::atomic::{ AtomicUsize, Ordering };
use std::fmt::Debug;

use serde::{ Serialize, Deserialize };

mod mops;
mod graph;
mod layer;

pub use graph::{ Graph, Module, MultiModule };
pub use layer::Layer;

use crate::{
  internal::*,
  tensor::Tensor,
  scalar::Real,
  ops::{ NonOps, BaseOps, NumericOps, SignedOps, BaseHops, NumericHops, RealHops },
};


pub static LAST_ID: AtomicUsize = AtomicUsize::new(0);

pub fn make_id() -> usize {
  LAST_ID.fetch_add(1, Ordering::Relaxed)
}


static AUTOGRAD: OnceLock<Mutex<bool>> = OnceLock::new();

fn autograd_mutex() -> &'static Mutex<bool> {
  AUTOGRAD.get_or_init(|| Mutex::new(true))
}

pub fn enable_autograd(val: bool) {
  *autograd_mutex().lock().unwrap() = val;
}

pub fn autograd() -> bool {
  *autograd_mutex().lock().unwrap()
}


/// Unary computational operation that can also compute its derivative.

pub trait UnaryOp<T: Real>: Debug + Send + Sync {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T>;
  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T>;
  fn as_enum(self) -> mops::UnaryMops;
}


/// Binary computational operation that can also compute its derivative.

pub trait BinaryOp<T: Real>: Debug + Send + Sync {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>;
  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>);
  fn as_enum(self) -> mops::BinaryMops;
}


/// Computational operation that can also compute its derivative.

pub trait MultiOp<T: Real>: Debug + Send + Sync {
  fn run(&self, inputs: &[&Tensor<T>]) -> Tensor<T>;
  fn derive(&self, inputs: &[&Tensor<T>], grad: &Tensor<T>) -> Vec<Tensor<T>>;
  fn as_enum(self) -> mops::MultiMops;
}


#[derive(Debug, Clone, Serialize, Deserialize)]
enum Op {
  Binary(mops::BinaryMops),
  Unary(mops::UnaryMops),
  Multi(mops::MultiMops),
}


/// Node in a computation graph, containing a [Variable]'s data and gradient,
/// as well as the operation used to create it.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Node<T: Real> {
  pub id: usize,
  cell: NodeCell<T>,
  op: Option<Op>,
  previous: Vec<RcT<Self>>,
  trainable: bool,
  traintape: Option<RcCell<Traintape<T>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeCell<T: Real> {
  data: Tensor<T>,
  grad: Option<Tensor<T>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Traintape<T: Real> {
  tape: Vec<RcT<Node<T>>>,
  counter: usize,
}

impl<T: Real> PartialEq for Node<T> {
  fn eq(&self, rhs: &Self) -> bool {
    self.id == rhs.id
  }
}

impl<T: Real> Node<T> {
  fn new(tensor: Tensor<T>, trainable: bool, traintape: Option<RcCell<Traintape<T>>>) -> Node<T> {
    Node {
      id: make_id(),
      cell: NodeCell {
        grad: if trainable {
          Some(Tensor::zeros(&tensor.shape().dims))
        } else {
          None
        },
        data: tensor,
      },
      op: None,
      previous: vec![],
      trainable,
      traintape,
    }
  }
  fn grad(&self) -> Option<&Tensor<T>> {
    self.cell.grad.as_ref()
  }

  fn reset_gradient(&self, filler: T) {
    if let Some(grad) = &self.cell.grad {
      grad.refill(filler);
    }
  }

  fn forward(&self) {
    if let Some(op) = &self.op {
      let lhs = &self.previous[0].cell.data;
      let value = match op {
        Op::Unary(op) => op.as_unary_op().run(lhs),
        Op::Binary(op) => {
          let rhs = &self.previous[1].cell.data;
          op.as_binary_op().run(lhs, rhs)
        },
        Op::Multi(op) => {
          let tensors: Vec<&Tensor<T>> = self.previous.iter().map(|prev| &prev.cell.data ).collect();
          op.as_multi_op().run(&tensors)
        },
      };
      if !self.cell.data.shared_with(&value) {
        self.cell.data.assign(&value);
        // When our tensor shares storage with its predecessors,
        // that means our value gets calculated using strides from their storage. No need to assign.
      }
    }
  }

  fn backward(&self) {
    if let (Some(op), Some(grad)) = (&self.op, &self.cell.grad) {
      let lhs = &self.previous[0];
      let changes = match op {
        Op::Unary(op) => vec![op.as_unary_op().derive(&lhs.cell.data, grad)],
        Op::Binary(op) => {
          let rhs = &self.previous[1];
          let changes = op.as_binary_op().derive(&lhs.cell.data, &rhs.cell.data, grad);
          vec![changes.0, changes.1]
        },
        Op::Multi(op) => {
          let tensors: Vec<&Tensor<T>> = self.previous.iter().map(|prev| &prev.cell.data ).collect();
          op.as_multi_op().derive(&tensors, grad)
        },
      };
      for (change, prev) in changes.iter().zip(self.previous.iter()) {
        if let Some(grad) = &prev.cell.grad {
          grad.op_assign(&change, |a, b| *a += b );
        }
      }
    }
  }
}


/// Variables track the computational operations used to create them and allow
/// for computing their gradient with respect to all input variables involved.
///
/// They get created by calling [tracked](Tensor::tracked) or
/// [trained](Tensor::trained) on any differentiable [Tensor] type.
///
/// Variables dereference to their underlying [Tensor] automatically for
/// non-differentiable operations. Differentiable operations, on the other hand,
/// will always return another Variable.

#[derive(Debug, Clone)]
pub struct Variable<T: Real> {
  pub(crate) node: RcT<Node<T>>,
}

//XXX disallow clone for trained vars

impl<T: Real> BaseHops<T> for Variable<T> {}
impl<T: Real> NumericHops<T> for Variable<T> {}
impl<T: Real> RealHops<T> for Variable<T> {}

impl<T: Real> std::ops::Deref for Variable<T> {
  type Target = Tensor<T>;

  fn deref(&self) -> &Self::Target {
    &self.node.cell.data
  }
}

impl<T: Real> PartialEq for Variable<T> {
  fn eq(&self, rhs: &Self) -> bool {
    self.node.cell.data == rhs.node.cell.data
  }
}

impl<T: Real> From<T> for Variable<T> {
  fn from(value: T) -> Self {
    Self::scalar(value)
  }
}

impl<T: Real> Variable<T> {
  pub(crate) fn from_tensor(tensor: Tensor<T>, trainable: bool) -> Self {
    Self { node: RcT::new(Node::new(tensor, trainable, None)) }
  }

  pub(crate) fn from_tape(trainable: bool, traintape_rc: &RcCell<Traintape<T>>, generator: impl Fn() -> Tensor<T>) -> Self {
    let node = if trainable {
      let mut traintape = borrow_mut(&traintape_rc);
      traintape.counter += 1;
      if traintape.counter <= traintape.tape.len() {
        let clone = (*traintape.tape[traintape.counter - 1]).clone();
        RcT::new(clone)
      } else {
        let node = RcT::new(Node::new(generator().detach(), true, Some(traintape_rc.clone())));
        traintape.tape.push(node.clone());
        node
      }
    } else {
      RcT::new(Node::new(generator().detach(), false, Some(traintape_rc.clone())))
    };
    Self { node }
  }

  fn operation(op: Op, data: Tensor<T>, grad: bool, previous: Vec<RcT<Node<T>>>) -> Self {
    // For MultiModules to work, trained vars need to retain their original traintapes, while inputs get
    // fresh ones for every copied version of a model. We need to pick one of those.
    let traintape: Option<RcCell<Traintape<T>>> = previous.iter()
      .find(|prev| prev.traintape.is_some() && !prev.trainable )
      .and_then(|node| node.traintape.clone() );
    Self {
      node: RcT::new(Node {
        id: make_id(),
        cell: NodeCell {
          grad: grad.then(|| Tensor::zeros(&data.shape().dims) ), //XXX Can be huge (#windows). Store as strided for ops that expand tensor.
          data,
        },
        op: Some(op),
        previous: if autograd() { previous } else { vec![] },
        trainable: false,
        traintape,
      }),
    }
  }

  pub fn id(&self) -> usize {
    self.node.id
  }

  pub fn grad(&self) -> Option<&Tensor<T>> {
    self.node.grad()
  }

  pub fn unary_op(&self, op: impl UnaryOp<T> + 'static) -> Self {
    let data = op.run(&self.node.cell.data);
    Self::operation(
      Op::Unary(op.as_enum()),
      data,
      self.grad().is_some(),
      vec![self.node.clone()],
    )
  }

  pub fn binary_op(&self, op: impl BinaryOp<T> + 'static, rhs: &Self) -> Self {
    let data = op.run(&self.node.cell.data, &rhs.node.cell.data);
    Self::operation(
      Op::Binary(op.as_enum()),
      data,
      self.grad().is_some() || rhs.grad().is_some(),
      vec![self.node.clone(), rhs.node.clone()],
    )
  }

  pub fn multi_op(op: impl MultiOp<T> + 'static, inputs: &[Self]) -> Self {
    let tensors: Vec<&Tensor<T>> = inputs.iter().map(|input| &input.node.cell.data ).collect();
    let data = op.run(&tensors);
    Self::operation(
      Op::Multi(op.as_enum()),
      data,
      inputs.iter().any(|input| input.grad().is_some() ),
      inputs.iter().map(|input| input.node.clone() ).collect(),
    )
  }

  /// Reevaluate this Variable's graph to produce a new output.

  pub fn forward(&self) {
    for node in self.history() {
      node.forward();
    }
  }

  /// Compute gradients across this Variable's entire graph.

  pub fn backward(&self) {
    if !self.grad().is_some() { panic!("Cannot compute gradients for constant {self}") }
    self.node.reset_gradient(T::one());
    for node in self.history().iter().rev() {
      node.backward();
    }
  }

  /// List all trainable parameters in this Variable's graph.

  pub fn parameters(&self) -> Vec<Self> {
    self.history()
      .into_iter()
      .filter(|node| node.trainable )
      .map(|node| Self { node } )
      .collect()
  }

  pub fn inputs(&self) -> Vec<Self> {
    self.history()
      .into_iter()
      .filter(|node| node.op.is_none() )
      .map(|node| Self { node } )
      .collect()
  }

  /// Set gradients to zero for this Variable's entire graph.

  pub fn reset(&self) {
    for node in self.history() {
      node.reset_gradient(T::zero());
    }
  }

  fn history(&self) -> Vec<RcT<Node<T>>> {
    let mut history = vec![];
    Self::history_recurse(&self.node, &mut history, &mut HashSet::new());
    history
  }

  fn history_recurse(node: &RcT<Node<T>>, history: &mut Vec<RcT<Node<T>>>, visited: &mut HashSet<usize>) {
    if visited.contains(&node.id) { return }
    visited.insert(node.id);
    for prev in &node.previous {
      Self::history_recurse(prev, history, visited);
    }
    history.push(node.clone());
  }

  /// Compute a function's gradient with respect to a generated
  /// input numerically and compare it to the automatically derived
  /// solution.
  ///
  /// Supply any function to check that it gets differentiated correctly.

  pub fn check_gradients<F>(shape: &[usize], generator: F) -> T
  where
    F: Fn(&Self) -> Self
  {
    let eps = T::from(0.01).unwrap();
    let two = T::from(2.0).unwrap();
    // Generate random input
    let input = Tensor::rand(shape);
    let var = input.trained();
    // Compute gradient using auto diff
    let output = generator(&var).sum(0);
    output.reset();
    output.backward();
    let grad = var.grad().unwrap().detach();
    // Compute gradient numerically for every param in input
    let len = input.shape().size();
    let mut num_grad = vec![T::from(0.0).unwrap(); len];
    for i in 0..len {
      let epst = Tensor::hot_encode(i, len).reshape(shape) * eps;
      let prev = generator(&(&input - &epst).tracked()).sum(0);
      let next = generator(&(&input + &epst).tracked()).sum(0);
      num_grad[i] = (next.item() - prev.item()) / (two * eps);
    }
    let num_grad = Tensor::new(&grad.shape().dims, num_grad);
    // Return average difference between both gradients
    (grad - num_grad).abs().mean(0).item()
  }

  pub fn summary(&self) {
    let history = self.history();
    let num_nodes = history.len();
    let num_ops = history.iter().filter(|node| node.op.is_some() ).collect::<Vec<_>>().len();
    let num_grads = history.iter().filter(|node| node.cell.grad.is_some() ).collect::<Vec<_>>().len();
    let params = self.parameters();
    let num_variables = params.len();
    let num_trainable_params: usize = params.iter().map(|param| param.size() ).sum();
    println!("# Nodes: {num_nodes}");
    println!("# Operations: {num_ops}");
    println!("# Gradients: {num_grads}");
    println!("# Trained Tensors: {num_variables}");
    println!("# Total Parameters: {num_trainable_params}");
    for param in params {
      println!("{} -> {}", param.shape(), param.shape().size());
    }
  }

  pub fn tracked(&self) -> Self { panic!("Tensor is already being tracked") }
  pub fn trained(&self) -> Self { panic!("Tensor is already being tracked") }

  pub fn stop_gradient(&self) -> Self {
    self.tensor().tracked()
  }
}

impl<T: Real> std::ops::AddAssign<Tensor<T>> for Variable<T> {
  fn add_assign(&mut self, rhs: Tensor<T>) {
    self.op_assign(&rhs, |a, b| *a += b );
  }
}

impl<T: Real> std::ops::SubAssign<Tensor<T>> for Variable<T> {
  fn sub_assign(&mut self, rhs: Tensor<T>) {
    self.op_assign(&rhs, |a, b| *a -= b );
  }
}

impl<T: Real> std::ops::MulAssign<Tensor<T>> for Variable<T> {
  fn mul_assign(&mut self, rhs: Tensor<T>) {
    self.op_assign(&rhs, |a, b| *a *= b );
  }
}

impl<T: Real> std::ops::DivAssign<Tensor<T>> for Variable<T> {
  fn div_assign(&mut self, rhs: Tensor<T>) {
    self.op_assign(&rhs, |a, b| *a /= b );
  }
}

impl<T: Real> std::iter::Sum for Variable<T> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self where I: Iterator {
    iter.fold(Self::zeros(&[1]), |acc, a| a + acc )
  }
}

impl<T: Real> std::fmt::Display for Variable<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    let title = if self.node.trainable { "Trainable" } else {
      if self.node.cell.grad.is_some() { "Differentiated" } else {
        if self.node.op.is_some() { "Computed" } else { "Tracked" }
      }
    };
    write!(f, "{title} {}", self.tensor())
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn x_squared() {
    let x = Tensor::vec(&[3.0, 5.0]).trained();
    let z = &x * &x + 2.0;
    z.backward();
    assert_eq!(z, Tensor::vec(&[11.0, 27.0]).tracked());
    assert_eq!(x.grad(), Some(&Tensor::from_shape(x.shape().clone(), vec![6.0, 10.0] )));
  }
}
