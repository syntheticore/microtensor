use std::ops;
use std::fmt::Debug;

use serde::{Serialize, Deserialize};

use crate::{
  internal::*,
  shape::Shape,
  tensor::Tensor,
  variable::{ Variable, BinaryOp, UnaryOp },
  scalar::Real,
  ops::{ BaseOps, NumericOps, SignedOps, RealOps },
};


impl<T: Real> Variable<T> {
  pub fn squeeze_only(&self, dim: isize) -> Self {
    Self::from_tensor(self.tensor().squeeze_only(dim), false)
  }
}

impl<T: Real> BaseOps<T> for Variable<T> {
  fn scalar(item: T) -> Self {
    Self::from_tensor(Tensor::scalar(item), false)
  }

  fn shape(&self) -> &Shape {
    self.node.cell.data.shape()
  }

  fn broadcast(&self, rhs: &Self) -> Self {
    self.operation_binary(Broadcast, rhs)
  }

  fn reshape(&self, shape: &[usize]) -> Self {
    self.operation_unary(Reshape { shape: shape.to_vec() })
  }

  fn unsqueeze(&self, dim: isize) -> Self {
    let shape = self.shape().unsqueeze(dim);
    self.reshape(&shape.dims) //XXX real unsqueeze works without #contiguous
  }
}

impl<T: Real> NumericOps<T> for Variable<T> {
  fn sum(&self, dim: isize) -> Variable<T> {
    self.operation_unary(Sum { dim })
  }

  fn mm(&self, rhs: &Self) -> Self {
    self.operation_binary(MatMul, rhs)
  }

  fn min(&self, dim: isize) -> Self {
    self.operation_unary(Min { dim })
  }

  fn max(&self, dim: isize) -> Self {
    self.operation_unary(Max { dim })
  }
}

impl<T: Real> SignedOps<T> for Variable<T> {
  fn abs(&self) -> Variable<T> {
    self.operation_unary(Abs)
  }
}

impl<T: Real> RealOps<T> for Variable<T> {
  fn pow(&self, rhs: &Self) -> Variable<T> {
    let (lhs, rhs) = if self.shape().dims != rhs.shape().dims {
      (self.broadcast(rhs), rhs.broadcast(self))
    } else {
      (self.clone(), rhs.clone())
    };
    lhs.operation_binary(Pow, &rhs)
  }

  fn relu(&self) -> Variable<T> {
    self.operation_unary(ReLU)
  }

  fn sigmoid(&self) -> Variable<T> {
    self.operation_unary(Sigmoid)
  }
}

impl<T: Real> std::ops::Neg for &Variable<T> {
  type Output = Variable<T>;

  fn neg(self) -> Self::Output {
    self * -T::one()
  }
}

impl<T: Real> std::ops::Neg for Variable<T> {
  type Output = Variable<T>;

  fn neg(self) -> Self::Output {
    -&self
  }
}

macro_rules! add_operator {
  ($op:ident, $operator:ident, $meth:ident, $symbol:tt, $broadcast:expr) => {
    impl<T: Real> ops::$operator for &Variable<T> { // &tensor * &other
      type Output = Variable<T>;

      fn $meth(self, rhs: Self) -> Variable<T> {
        let (lhs, rhs) = if $broadcast && self.shape().dims != rhs.shape().dims {
          (self.broadcast(rhs), rhs.broadcast(self))
        } else {
          (self.clone(), rhs.clone())
        };
        lhs.operation_binary($op, &rhs)
      }
    }

    impl<T: Real> ops::$operator for Variable<T> { // tensor * other
      type Output = Variable<T>;

      fn $meth(self, rhs: Self) -> Variable<T> {
        &self $symbol &rhs
      }
    }

    impl<T: Real> ops::$operator<Variable<T>> for &Variable<T> { // &tensor * other
      type Output = Variable<T>;

      fn $meth(self, rhs: Variable<T>) -> Variable<T> {
        self $symbol &rhs
      }
    }

    impl<T: Real> ops::$operator<&Variable<T>> for Variable<T> { // tensor * &other
      type Output = Variable<T>;

      fn $meth(self, rhs: &Variable<T>) -> Variable<T> {
        &self $symbol rhs
      }
    }

    impl<T: Real> ops::$operator<T> for &Variable<T> { // &tensor * T
      type Output = Variable<T>;

      fn $meth(self, rhs: T) -> Variable<T> {
        self $symbol &Tensor::scalar(rhs).tracked()
      }
    }

    impl<T: Real> ops::$operator<T> for Variable<T> { // tensor * T
      type Output = Variable<T>;

      fn $meth(self, rhs: T) -> Variable<T> {
        &self $symbol &Tensor::scalar(rhs).tracked()
      }
    }

    impl ops::$operator<&Variable<f32>> for f32 { // T * &tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: &Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol rhs
      }
    }

    impl ops::$operator<Variable<f32>> for f32 { // T * tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol &rhs
      }
    }
  };
}

add_operator!(Add, Add, add, +, true);
add_operator!(Sub, Sub, sub, -, true);
add_operator!(Mul, Mul, mul, *, true);
add_operator!(Div, Div, div, /, true);
add_operator!(MatMul, Rem, rem, %, false);


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Add;

impl<T: Real> BinaryOp<T> for Add {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.add(rhs) }

  fn derivative(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.clone(),
    grad.clone(),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sub;

impl<T: Real> BinaryOp<T> for Sub {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.sub(rhs) }

  fn derivative(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.clone(),
    -grad
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mul;

impl<T: Real> BinaryOp<T> for Mul {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.mul(rhs) }

  fn derivative(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.mul(rhs),
    grad.mul(lhs),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Div;

impl<T: Real> BinaryOp<T> for Div {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.div(rhs) }

  fn derivative(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.div(rhs),
    -grad.mul(lhs).div(rhs).div(rhs)
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul;

impl<T: Real> BinaryOp<T> for MatMul {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.mm(rhs) }

  fn derivative(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.mm(&rhs.transpose_vec(false)),
    lhs.transpose_vec(true).mm(grad),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Broadcast;

impl<T: Real> BinaryOp<T> for Broadcast {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.broadcast(rhs)
  }

  fn derivative(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    unbroadcast(lhs, rhs, grad),
    Tensor::zeros(&rhs.shape().dims),
  )}
}

fn unbroadcast<T: Real>(tensor: &Tensor<T>, other: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
  let shape = tensor.shape().broadcast(&other.shape());
  let mut grad = grad.clone();
  for (d, &stride) in shape.strides.iter().enumerate().rev() {
    if stride == 0 {
      grad = grad.sum_over(d as isize);
    }
  }
  grad
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reshape {
  shape: Vec<usize>,
}

impl<T: Real> UnaryOp<T> for Reshape {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.reshape(&self.shape)
  }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad.reshape(&lhs.shape().dims)
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pow;

impl<T: Real> BinaryOp<T> for Pow {
  fn execute(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> { lhs.pow(rhs) }

  fn derivative(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {(
    grad.mul(rhs).mul(&lhs.pow(&rhs.sub(&Tensor::ones(&rhs.shape().dims)))),
    grad.mul(&lhs.pow(rhs)).mul(&lhs.log()),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sum {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Sum {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sum(self.dim)
  }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    uncollapse(self.dim, lhs, grad)
  }
}

fn uncollapse<T: Real>(dim: isize, tensor: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
  let rank = tensor.shape().rank();
  let dim = negative_index(dim, rank, false);
  let removed = rank - dim;
  let mut grad = grad.clone();
  for _ in 0..removed {
    grad = grad.unsqueeze(-1);
  }
  grad
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Abs;

impl<T: Real> UnaryOp<T> for Abs {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> { lhs.abs() }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * lhs.signum()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Min {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Min {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> { lhs.min(self.dim) }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    // grad * lhs.min_index(self.dim).one_hot(lhs.shape()[self.dim])
    uncollapse(self.dim, lhs, grad) //XXX
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Max {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Max {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> { lhs.max(self.dim) }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    uncollapse(self.dim, lhs, grad) //XXX
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl<T: Real> UnaryOp<T> for ReLU {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.relu()
  }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * lhs.gt(&Tensor::scalar(T::zero())).numeric()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl<T: Real> UnaryOp<T> for Sigmoid {
  fn execute(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sigmoid()
  }

  fn derivative(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let result = lhs.sigmoid();
    grad * (&result * (Tensor::scalar(T::one()) - &result))
  }
}
