use std::fmt::Debug;

use serde::{Serialize, Deserialize};

use crate::{
  internal::*,
  shape::Shape,
  tensor::Tensor,
  variable::{ Variable, BinaryOp, UnaryOp },
  scalar::Real,
  ops::{ BaseOps, NumericOps, SignedOps, RealOps, BaseHops, NumericHops },
};


impl<T: Real> BaseOps<T> for Variable<T> {
  fn scalar(item: T) -> Self {
    Self::from_tensor(Tensor::scalar(item), false)
  }

  fn fill(shape: &[usize], filler: T) -> Self {
    Self::from_tensor(Tensor::fill(shape, filler), false)
  }

  fn shape(&self) -> &Shape {
    self.node.cell.data.shape()
  }

  fn range(&self, ranges: &[std::ops::Range<isize>]) -> Self {
    self.unary_op(Range { ranges: ranges.to_vec() })
  }

  fn broadcast(&self, shape: &Shape, ignore_from: Option<isize>) -> Self {
    if self.shape().broadcast(&shape, ignore_from).dims == self.shape().dims { return self.clone() }
    self.unary_op(Broadcast { dims: shape.dims.clone(), ignore_from })
  }

  fn reshape(&self, dims: &[usize]) -> Self {
    self.unary_op(Reshape { dims: dims.to_vec() })
  }

  fn squeeze(&self, squeezed: &[isize]) -> Self {
    let shape = self.shape().squeeze(squeezed);
    if shape.dims == self.shape().dims { return self.clone() }
    self.reshape(&shape.dims)
  }

  fn unsqueeze(&self, dim: isize) -> Self {
    let shape = self.shape().unsqueeze(dim);
    self.reshape(&shape.dims) //XXX real unsqueeze works without #contiguous
  }

  fn transpose(&self, dim1: isize, dim2: isize) -> Self {
    self.unary_op(Transpose { dim1, dim2 })
  }

  fn concat(&self, rhs: &Self, dim: isize) -> Self {
    self.binary_op(Concat { dim }, rhs)
  }

  fn assign_masked(&self, rhs: &Self, cb: impl Fn(&Shape) -> Shape) -> Self {
    let mask = cb(&Shape::new(&self.shape().dims));
    self.binary_op(AssignMasked { mask }, rhs)
  }

  fn layout(&self, cb: impl Fn(&Shape) -> Shape) -> Self {
    let shape = cb(& if self.shape().contiguous() {
      self.shape().clone()
    } else {
      Shape::new(&self.shape().dims)
    });
    self.unary_op(Layout { shape })
  }
}

impl<T: Real> NumericOps<T> for Variable<T> {
  fn sum(&self, dim: isize) -> Variable<T> {
    self.unary_op(Sum { dim })
  }

  fn mm(&self, rhs: &Self) -> Self {
    self.binary_op(MatMul, rhs)
  }

  fn min(&self, dim: isize) -> Self {
    self.unary_op(Min { dim })
  }

  fn max(&self, dim: isize) -> Self {
    self.unary_op(Max { dim })
  }

  // fn compose(dims: &[usize], inputs: &[(Shape, Self)]) -> Self {
  //   let masks = inputs.iter().map(|(mask, _)| mask.clone() ).collect();
  //   let others: Vec<_> = inputs.iter().map(|(_, other)| other ).collect();
  //   Self::multi_op(Compose { dims: dims.to_vec(), masks }, &others)
  // }
}

impl<T: Real> SignedOps<T> for Variable<T> {
  fn abs(&self) -> Variable<T> {
    self.unary_op(Abs)
  }
}

impl<T: Real> RealOps<T> for Variable<T> {
  fn pow(&self, rhs: &Self) -> Variable<T> {
    let (lhs, rhs) = if self.shape().dims != rhs.shape().dims {
      (self.broadcast(&rhs.shape(), None), rhs.broadcast(&self.shape(), None))
    } else {
      (self.clone(), rhs.clone())
    };
    lhs.binary_op(Pow, &rhs)
  }

  fn sin(&self) -> Self {
    self.unary_op(Sin)
  }

  fn cos(&self) -> Self {
    self.unary_op(Cos)
  }

  fn log(&self) -> Self {
    self.unary_op(Log)
  }

  fn relu(&self) -> Variable<T> {
    self.unary_op(ReLU)
  }

  fn sigmoid(&self) -> Variable<T> {
    self.unary_op(Sigmoid)
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
  ($op:ident, $meth:ident, $symbol:tt) => {
    impl<T: Real> std::ops::$op for &Variable<T> { // &tensor * &other
      type Output = Variable<T>;

      fn $meth(self, rhs: Self) -> Variable<T> {
        let (lhs, rhs) = if self.shape().dims != rhs.shape().dims {
          (self.broadcast(&rhs.shape(), None), rhs.broadcast(&self.shape(), None))
        } else {
          (self.clone(), rhs.clone())
        };
        lhs.binary_op($op, &rhs)
      }
    }

    impl<T: Real> std::ops::$op for Variable<T> { // tensor * other
      type Output = Variable<T>;

      fn $meth(self, rhs: Self) -> Variable<T> {
        &self $symbol &rhs
      }
    }

    impl<T: Real> std::ops::$op<Variable<T>> for &Variable<T> { // &tensor * other
      type Output = Variable<T>;

      fn $meth(self, rhs: Variable<T>) -> Variable<T> {
        self $symbol &rhs
      }
    }

    impl<T: Real> std::ops::$op<&Variable<T>> for Variable<T> { // tensor * &other
      type Output = Variable<T>;

      fn $meth(self, rhs: &Variable<T>) -> Variable<T> {
        &self $symbol rhs
      }
    }

    impl<T: Real> std::ops::$op<T> for &Variable<T> { // &tensor * T
      type Output = Variable<T>;

      fn $meth(self, rhs: T) -> Variable<T> {
        self $symbol &Tensor::scalar(rhs).tracked()
      }
    }

    impl<T: Real> std::ops::$op<T> for Variable<T> { // tensor * T
      type Output = Variable<T>;

      fn $meth(self, rhs: T) -> Variable<T> {
        &self $symbol &Tensor::scalar(rhs).tracked()
      }
    }

    impl std::ops::$op<&Variable<f32>> for f32 { // T * &tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: &Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol rhs
      }
    }

    impl std::ops::$op<Variable<f32>> for f32 { // T * tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol &rhs
      }
    }
  };
}

add_operator!(Add, add, +);
add_operator!(Sub, sub, -);
add_operator!(Mul, mul, *);
add_operator!(Div, div, /);
add_operator!(Rem, rem, %);


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Add;

impl<T: Real> BinaryOp<T> for Add {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs + rhs
  }

  fn derive(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad.clone(),
    grad.clone(),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sub;

impl<T: Real> BinaryOp<T> for Sub {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs - rhs
  }

  fn derive(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad.clone(),
    -grad
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mul;

impl<T: Real> BinaryOp<T> for Mul {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs * rhs
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad * rhs,
    grad * lhs,
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Div;

impl<T: Real> BinaryOp<T> for Div {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs / rhs
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad / rhs,
    -grad * lhs / rhs / rhs
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rem;

impl<T: Real> BinaryOp<T> for Rem {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs % rhs
  }

  fn derive(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad.clone(),
    -grad
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul;

impl<T: Real> BinaryOp<T> for MatMul {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.mm(rhs)
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {
    let mut grad_l = grad.mm(&rhs.transpose_vec(false));
    let mut grad_r = lhs.transpose_vec(true).mm(grad);

    let rank_l = lhs.rank();
    let rank_r = rhs.rank();
    let rank = rank_l.max(rank_r);

    // Sum over batch dimensions if tensor was broadcasted for bmm
    lhs.shape().dims.iter()
      .rev()
      .chain(std::iter::repeat(&1))
      .zip(rhs.shape().dims.iter()
        .rev()
        .chain(std::iter::repeat(&1)))
      .enumerate()
      .skip(2)
      .take(rank)
      .for_each(|(i, (&dl, &dr))| {
        let i = rank as isize - 1 - i as isize;
        if dl == 1 && dr != 1 {
          grad_l = grad_l.sum_over(i).squeeze_only(i)
        } else if dr == 1 && dl != 1 {
          grad_r = grad_r.sum_over(i).squeeze_only(i)
        }
      });

    (grad_l, grad_r)
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
  ranges: Vec<std::ops::Range<isize>>,
}

impl<T: Real> UnaryOp<T> for Range {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.range(&self.ranges)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let out = Tensor::zeros(&lhs.shape().dims);
    let sliced = out.shape().range(&self.ranges);
    {
      let mut out_raw = out.raw_mut();
      for (i, g) in sliced.iter().zip(grad.param_iter()) {
        out_raw[i] = g;
      }
    }
    out
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Broadcast {
  dims: Vec<usize>,
  ignore_from: Option<isize>,
}

impl<T: Real> UnaryOp<T> for Broadcast {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.broadcast(&Shape::new(&self.dims), self.ignore_from)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let shape = lhs.shape().broadcast(&Shape::new(&self.dims), self.ignore_from);

    let rank = lhs.rank().max(self.dims.len());
    let ignore = self.ignore_from.unwrap_or(rank as isize);
    let ignore = negative_index(ignore, rank, false);

    let mut grad = grad.clone();
    for (d, &stride) in shape.strides.iter().enumerate().rev() {
      if stride == 0 && d < ignore {
        grad = grad.sum_over(d as isize);
      }
    }
    grad
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reshape {
  dims: Vec<usize>,
}

impl<T: Real> UnaryOp<T> for Reshape {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.reshape(&self.dims)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad.reshape(&lhs.shape().dims)
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transpose {
  dim1: isize,
  dim2: isize,
}

impl<T: Real> UnaryOp<T> for Transpose {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.transpose(self.dim1, self.dim2)
  }

  fn derive(&self, _lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad.transpose(self.dim1, self.dim2)
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concat {
  dim: isize,
}

impl<T: Real> BinaryOp<T> for Concat {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.concat(rhs, self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let size_l = lhs.dim(self.dim);
    let size_r = rhs.dim(self.dim);
    let dim = negative_index(self.dim, grad.rank(), false);

    let mut ranges_l = vec![0..-1; dim + 1];
    let mut ranges_r = ranges_l.clone();
    let size_l = size_l as isize;
    ranges_l[dim] = 0..size_l;
    ranges_r[dim] = size_l..size_l + size_r as isize;

    (grad.range(&ranges_l), grad.range(&ranges_r))
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignMasked {
  mask: Shape,
}

impl<T: Real> BinaryOp<T> for AssignMasked {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.assign_masked(rhs, |_| self.mask.clone() )
  }

  fn derive(&self, _lhs: &Tensor<T>, _rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let lgrad = grad.detach();
    lgrad.layout(|_| self.mask.clone() ).assign(&Tensor::scalar(T::zero()));
    let rgrad = grad.complete().layout(|_| self.mask.clone() );
    (lgrad, rgrad)
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layout {
  shape: Shape,
}

impl<T: Real> UnaryOp<T> for Layout {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.layout(|_| self.shape.clone() )
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let out = Tensor::zeros(&lhs.shape().dims);
    {
      let mut raw = out.raw_mut();
      for (i, g) in self.shape.iter().zip(grad.param_iter()) {
        raw[i - self.shape.offset] += g;
      }
    }
    out
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pow;

impl<T: Real> BinaryOp<T> for Pow {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.pow(rhs)
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {(
    grad * rhs * lhs.pow(&(rhs - T::one())),
    grad * lhs.pow(rhs) * lhs.log(),
  )}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sin;

impl<T: Real> UnaryOp<T> for Sin {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sin()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * lhs.cos()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cos;

impl<T: Real> UnaryOp<T> for Cos {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.cos()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * -lhs.sin()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log;

impl<T: Real> UnaryOp<T> for Log {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.log()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad / lhs
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sum {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Sum {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sum(self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
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
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.abs()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * lhs.signum()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Min {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Min {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.min(self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    // grad * lhs.min_index(self.dim).one_hot(lhs.shape()[self.dim])
    uncollapse(self.dim, lhs, grad) //XXX
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Max {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for Max {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.max(self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    uncollapse(self.dim, lhs, grad) //XXX
  }
}


// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct Compose {
//   dims: Vec<usize>,
//   masks: Vec<Shape>,
// }

// impl<T: Real> MultiOp<T> for Compose {
//   fn run(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
//     let masked_inputs: Vec<_> = inputs.iter().enumerate().map(|(i, &input)| (self.masks[i].clone(), input.clone())).collect();
//     Tensor::compose(&self.dims, &masked_inputs)
//   }

//   fn derive(&self, inputs: &[&Tensor<T>], grad: &Tensor<T>) -> Vec<Tensor<T>> {
//     inputs.iter().map(|_| grad.clone() ).collect()
//   }
// }


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl<T: Real> UnaryOp<T> for ReLU {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.relu()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad * lhs.gt(&Tensor::scalar(T::zero())).numeric()
  }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl<T: Real> UnaryOp<T> for Sigmoid {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sigmoid()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let result = lhs.sigmoid();
    grad * (&result * (Tensor::scalar(T::one()) - &result))
  }
}
