use std::rc::Rc;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt::Debug;

use serde::{Serialize, Deserialize};

mod mops;
mod graph;

pub use graph::Graph;

use crate::{
  tensor::Tensor,
  scalar::Real,
  ops::{ BaseOps, NumericOps, BaseHops, RealHops },
};


pub fn make_id() -> usize {
  static LAST_ID: AtomicUsize = AtomicUsize::new(0);
  LAST_ID.fetch_add(1, Ordering::Relaxed)
}


/// Unary computational operation that can also compute its derivative.

pub trait UnaryOp<T: Real>: Debug + Send + Sync + serde_traitobject::Serialize + serde_traitobject::Deserialize {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T>;
  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T>;
}


/// Binary computational operation that can also compute its derivative.

pub trait BinaryOp<T: Real>: Debug + Send + Sync + serde_traitobject::Serialize + serde_traitobject::Deserialize {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>;
  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>);
}


/// Node in a computation graph, containing a [Variable]'s data and gradient,
/// as well as the operation used to create it.

#[derive(Debug)]
struct Node<T: Real + 'static> {
  pub id: usize,
  cell: NodeCell<T>,
  op: Option<Op<T>>,
  previous: Vec<Rc<Self>>,
  trainable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeCell<T: Real> {
  data: Tensor<T>,
  grad: Option<Tensor<T>>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Op<T: Real + 'static> {
  Binary(serde_traitobject::Box<dyn BinaryOp<T>>),
  Unary(serde_traitobject::Box<dyn UnaryOp<T>>),
}

impl<T: Real> PartialEq for Node<T> {
  fn eq(&self, rhs: &Self) -> bool {
    self.id == rhs.id
  }
}

impl<T: Real> Node<T> {
  fn grad(&self) -> Option<&Tensor<T>> {
    self.cell.grad.as_ref()
  }

  fn reset_gradient(&self, filler: T) {
    if let Some(grad) = &self.cell.grad {
      grad.feed(&Tensor::fill(&grad.shape().dims, filler));
    }
  }

  fn forward(&self) {
    if let Some(op) = &self.op {
      let lhs = &self.previous[0].cell.data;
      let value = match op {
        Op::Unary(op) => op.run(lhs),
        Op::Binary(op) => {
          let rhs = &self.previous[1].cell.data;
          op.run(lhs, rhs)
        },
      };
      self.cell.data.feed(&value);
    }
  }

  fn backward(&self) {
    if let (Some(op), Some(grad)) = (&self.op, &self.cell.grad) {
      let lhs = &self.previous[0];
      let changes = match op {
        Op::Unary(op) => vec![op.derive(&lhs.cell.data, grad)],
        Op::Binary(op) => {
          let rhs = &self.previous[1];
          let changes = op.derive(&lhs.cell.data, &rhs.cell.data, grad);
          vec![changes.0, changes.1]
        },
      };
      for (change, prev) in changes.iter().zip(self.previous.iter()) {
        if let Some(grad) = &prev.cell.grad {
          grad.feed(&grad.add(change));
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
pub struct Variable<T: Real + 'static> {
  node: Rc<Node<T>>,
}

impl<T: Real> BaseHops<T> for Variable<T> {}
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

impl<T: Real> From<Tensor<T>> for Variable<T> {
  fn from(tensor: Tensor<T>) -> Self {
    Self::from_tensor(tensor, false)
  }
}

impl<T: Real> From<&Tensor<T>> for Variable<T> {
  fn from(tensor: &Tensor<T>) -> Self {
    Self::from_tensor(tensor.clone(), false)
  }
}

impl<T: Real + 'static> Variable<T> {
  pub(crate) fn from_tensor(array: Tensor<T>, trainable: bool) -> Self {
    Self {
      node: Rc::new(Node {
        id: make_id(),
        cell: NodeCell {
          grad: if trainable {
            Some(Tensor::zeros(&array.shape().dims))
          } else {
            None
          },
          data: array,
        },
        op: None,
        previous: vec![],
        trainable,
      }),
    }
  }

  fn operation(op: Op<T>, data: Tensor<T>, grad: bool, previous: Vec<Rc<Node<T>>>) -> Self {
    Self {
      node: Rc::new(Node {
        id: make_id(),
        cell: NodeCell {
          grad: grad.then(|| Tensor::zeros(&data.shape().dims) ),
          data,
        },
        op: Some(op),
        previous,
        trainable: false,
      }),
    }
  }

  pub fn id(&self) -> usize {
    self.node.id
  }

  pub fn tensor(&self) -> &Tensor<T> {
    &self.node.cell.data
  }

  pub fn grad(&self) -> Option<&Tensor<T>> {
    self.node.grad()
  }

  pub fn unary_op(&self, op: impl UnaryOp<T> + 'static) -> Self {
    let data = op.run(&self.node.cell.data);
    Self::operation(
      Op::Unary(serde_traitobject::Box::new(op)),
      data,
      self.grad().is_some(),
      vec![self.node.clone()])
  }

  pub fn binary_op(&self, op: impl BinaryOp<T> + 'static, rhs: &Self) -> Self {
    let data = op.run(&self.node.cell.data, &rhs.node.cell.data);
    Self::operation(
      Op::Binary(serde_traitobject::Box::new(op)),
      data,
      self.grad().is_some() || rhs.grad().is_some(),
      vec![self.node.clone(), rhs.node.clone()],
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

  fn history(&self) -> Vec<Rc<Node<T>>> {
    let mut history = vec![];
    Self::history_recurse(&self.node, &mut history, &mut HashSet::new());
    history
  }

  fn history_recurse(node: &Rc<Node<T>>, history: &mut Vec<Rc<Node<T>>>, visited: &mut HashSet<usize>) {
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
    let eps = T::from(1e-6).unwrap();
    let two = T::from(2.0).unwrap();
    // Generate random input
    let input = Tensor::randn(shape);
    let var = input.trained();
    // Compute gradient using auto diff
    let output = generator(&var).sum(0);
    output.reset();
    output.backward();
    let grad = var.grad().unwrap().detach();
    // Compute gradient numerically for every param in input
    let len = input.shape().size();
    let mut diff = T::zero();
    for i in 0..len {
      let epst = Tensor::hot_encode(i, len).reshape(shape) * eps;
      let prev = generator(&(&input - &epst).tracked()).sum(0);
      let next = generator(&(&input + &epst).tracked()).sum(0);
      let change = (next.item() - prev.item()) / (two * eps);
      diff += (grad.raw()[i] - change).abs();
    }
    // Return average difference between both gradients
    diff / T::from(len).unwrap()
  }

  pub fn statistics(&self) -> (usize, usize, usize, usize, usize) {
    let history = self.history();
    let num_nodes = history.len();
    let num_ops = history.iter().filter(|node| node.op.is_some() ).collect::<Vec<_>>().len();
    let num_grads = history.iter().filter(|node| node.cell.grad.is_some() ).collect::<Vec<_>>().len();
    let params = self.parameters();
    let num_variables = params.len();
    let num_trainable_params = params.iter().map(|param| param.shape().size() ).sum();
    (num_nodes, num_ops, num_grads, num_variables, num_trainable_params)
  }
}

impl<T: Real> std::ops::AddAssign<Tensor<T>> for Variable<T> {
  fn add_assign(&mut self, rhs: Tensor<T>) {
    self.feed(&self.tensor().add(&rhs));
  }
}

impl<T: Real> std::ops::SubAssign<Tensor<T>> for Variable<T> {
  fn sub_assign(&mut self, rhs: Tensor<T>) {
    self.feed(&self.tensor().sub(&rhs));
  }
}

impl<T: Real> std::ops::MulAssign<Tensor<T>> for Variable<T> {
  fn mul_assign(&mut self, rhs: Tensor<T>) {
    self.feed(&self.tensor().mul(&rhs));
  }
}

impl<T: Real> std::ops::DivAssign<Tensor<T>> for Variable<T> {
  fn div_assign(&mut self, rhs: Tensor<T>) {
    self.feed(&self.tensor().div(&rhs));
  }
}

impl<T: Real> std::fmt::Display for Variable<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    let title = if self.node.trainable { "Trainable" } else {
      if self.node.cell.grad.is_some() { "Computed" } else { "Tracked" }
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
