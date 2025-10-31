use std::fmt::Debug;

use serde::{Serialize, Deserialize};

use crate::{
  internal::*,
  shape::Shape,
  tensor::Tensor,
  variable::{ Variable, UnaryOp, BinaryOp, MultiOp },
  scalar::Real,
  ops::{ NonOps, BaseOps, NumericOps, SignedOps, RealOps, BaseHops, NumericHops },
};


impl<T: Real> NonOps<T> for Variable<T> {
  fn scalar(item: T) -> Self {
    Self::from_tensor(Tensor::scalar(item), false)
  }

  fn fill(shape: &[usize], filler: T) -> Self {
    Self::from_tensor(Tensor::fill(shape, filler), false)
  }

  fn item(&self) -> T {
    self.node.cell.data.item()
  }

  fn shape(&self) -> &Shape {
    self.node.cell.data.shape()
  }

  fn tensor(&self) -> &Tensor<T> {
    &self.node.cell.data
  }

  fn reself(other: Tensor<T>) -> Self { Self::from_tensor(other, false) }
}

impl<T: Real> BaseOps<T> for Variable<T> {
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

  fn stack(inputs: &[Self], dim: isize) -> Self {
    Self::multi_op(Stack { dim }, inputs)
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

  fn min_over(&self, dim: isize) -> Self {
    self.unary_op(MinOver { dim })
  }

  fn max_over(&self, dim: isize) -> Self {
    self.unary_op(MaxOver { dim })
  }

  fn look_up(&self, rhs: &Self) -> Self {
    self.binary_op(LookUp, rhs)
  }
}

impl<T: Real> SignedOps<T> for Variable<T> {
  fn abs(&self) -> Variable<T> {
    self.unary_op(Abs)
  }
}

impl<T: Real> RealOps<T> for Variable<T> {
  fn pow(&self, rhs: &Self) -> Variable<T> {
    let (lhs, rhs) = if self.shape().dims != rhs.shape().dims { //XXX remove, because already checked in broadcast itself
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

  fn tanh(&self) -> Self {
    self.unary_op(Tanh)
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

  fn silu(&self) -> Self {
    self.unary_op(SiLU)
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

    impl std::ops::$op<&Variable<f32>> for f32 { // f32 * &tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: &Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol rhs
      }
    }

    impl std::ops::$op<Variable<f32>> for f32 { // f32 * tensor
      type Output = Variable<f32>;

      fn $meth(self, rhs: Variable<f32>) -> Variable<f32> {
        Tensor::scalar(self).tracked() $symbol &rhs
      }
    }

    impl std::ops::$op<&Variable<f64>> for f64 { // f64 * &tensor
      type Output = Variable<f64>;

      fn $meth(self, rhs: &Variable<f64>) -> Variable<f64> {
        Tensor::scalar(self).tracked() $symbol rhs
      }
    }

    impl std::ops::$op<Variable<f64>> for f64 { // f64 * tensor
      type Output = Variable<f64>;

      fn $meth(self, rhs: Variable<f64>) -> Variable<f64> {
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

  fn as_enum(self) -> BinaryMops { BinaryMops::Add(self) }
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

  fn as_enum(self) -> BinaryMops { BinaryMops::Sub(self) }
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

  fn as_enum(self) -> BinaryMops { BinaryMops::Mul(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Div;

impl<T: Real> BinaryOp<T> for Div {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs / rhs
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let shape = grad.shape();
    let lhs_b = lhs.broadcast(shape, None);
    let rhs_b = rhs.broadcast(shape, None);
    let cap = shape.size();
    let mut lhs_grad = Vec::with_capacity(cap);
    let mut rhs_grad = Vec::with_capacity(cap);
    for ((g, l), r) in grad.param_iter().zip(lhs_b.param_iter()).zip(rhs_b.param_iter()) {
      lhs_grad.push(g / r);
      rhs_grad.push(-g * l / (r * r));
    }
    (Tensor::new(&shape.dims, lhs_grad), Tensor::new(&shape.dims, rhs_grad))
  }

  fn as_enum(self) -> BinaryMops { BinaryMops::Div(self) }
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

  fn as_enum(self) -> BinaryMops { BinaryMops::Rem(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul;

impl<T: Real> BinaryOp<T> for MatMul {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.mm(rhs)
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>)
  {
    let mut grad = grad.clone();

    if grad.rank() == 1 {
      if lhs.rank() == 1 {
        // vector × matrix
        grad = grad.unsqueeze(0);
      } else if rhs.rank() == 1 {
        // matrix × vector
        grad = grad.unsqueeze(-1);
      }
    }

    let mut grad_l = grad.mm(&rhs.transpose_vec(false));
    let mut grad_r = lhs.transpose_vec(true).mm(&grad);

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

  fn as_enum(self) -> BinaryMops { BinaryMops::MatMul(self) }
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
    let out = Tensor::zeros_like(&lhs);
    let sliced = out.shape().range(&self.ranges);
    {
      let mut out_raw = out.raw_mut();
      for (i, g) in sliced.iter().zip(grad.param_iter()) {
        out_raw[i] = g;
      }
    }
    out
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Range(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookUp;

impl<T: Real> BinaryOp<T> for LookUp {
  fn run(&self, table: &Tensor<T>, tokens: &Tensor<T>) -> Tensor<T> {
    table.look_up(tokens)
  }

  fn derive(&self, table: &Tensor<T>, tokens: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let out = Tensor::zeros(&table.shape().dims);
    for (tok_grad, token) in grad.iter(-2).zip(tokens.iter(-1)) {
      out.at(&[token.cast().item()]).op_assign(&tok_grad, |a, b| *a += b )
    }
    (out, Tensor::scalar(T::zero()))
  }

  fn as_enum(self) -> BinaryMops { BinaryMops::LookUp(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Broadcast(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Reshape(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Transpose(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stack {
  dim: isize,
}

impl<T: Real> MultiOp<T> for Stack {
  fn run(&self, inputs: &[&Tensor<T>]) -> Tensor<T> {
    let inputs: Vec<_> = inputs.iter().map(|&input| input.clone() ).collect();
    Tensor::stack(&inputs, self.dim)
  }

  fn derive(&self, inputs: &[&Tensor<T>], grad: &Tensor<T>) -> Vec<Tensor<T>> {
    let dim = negative_index(self.dim, grad.rank(), false);
    let mut offset = 0;
    inputs.iter().map(|input| {
      let size = input.dim(self.dim) as isize;
      let mut ranges = vec![0..-1; dim + 1];
      ranges[dim] = offset..offset + size;
      offset += size;
      grad.range(&ranges)
    }).collect()
  }

  fn as_enum(self) -> MultiMops { MultiMops::Stack(self) }
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

  fn as_enum(self) -> BinaryMops { BinaryMops::AssignMasked(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Layout(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pow;

impl<T: Real> BinaryOp<T> for Pow {
  fn run(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T> {
    lhs.pow(rhs)
  }

  fn derive(&self, lhs: &Tensor<T>, rhs: &Tensor<T>, grad: &Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let shape = grad.shape();
    let lhs_b = lhs.broadcast(shape, None);
    let rhs_b = rhs.broadcast(shape, None);
    let mut lhs_grad = Vec::with_capacity(shape.size());
    let mut rhs_grad = Vec::with_capacity(shape.size());
    let one = T::one();
    for ((g, x), y) in grad.param_iter().zip(lhs_b.param_iter()).zip(rhs_b.param_iter()) {
      lhs_grad.push(g * y * x.powf(y - one));
      rhs_grad.push(g * x.powf(y) * x.ln());
    }
    (Tensor::new(&shape.dims, lhs_grad), Tensor::new(&shape.dims, rhs_grad))
  }

  fn as_enum(self) -> BinaryMops { BinaryMops::Pow(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sin;

impl<T: Real> UnaryOp<T> for Sin {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sin()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad.zip(lhs, |(g, x)| g * x.cos() )
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Sin(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cos;

impl<T: Real> UnaryOp<T> for Cos {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.cos()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    grad.zip(lhs, |(g, x)| -g * x.sin() )
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Cos(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tanh;

impl<T: Real> UnaryOp<T> for Tanh {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.tanh()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let one = T::one();
    grad.zip(lhs, |(g, x)| {
      let th = x.tanh();
      g * (one - th * th)
    })
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Tanh(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Log(self) }
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Sum(self) }
}

fn uncollapse<T: Real>(dim: isize, tensor: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
  let rank = tensor.shape().rank();
  let dim = negative_index(dim, rank, false);
  let removed = rank - dim;
  grad.unsqueeze_n(removed, -1)
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

  fn as_enum(self) -> UnaryMops { UnaryMops::Abs(self) }
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
    let indices: Tensor<usize> = lhs.argmin(self.dim);
    let removed_flat_size = lhs.shape()[self.dim..-1].iter().product();
    uncollapse(self.dim, lhs, grad) * indices.one_hot(removed_flat_size).reshape(&lhs.shape().dims)
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Min(self) }
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
    let indices: Tensor<usize> = lhs.argmax(self.dim);
    let removed_flat_size = lhs.shape()[self.dim..-1].iter().product();
    uncollapse(self.dim, lhs, grad) * indices.one_hot(removed_flat_size).reshape(&lhs.shape().dims)
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Max(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinOver {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for MinOver {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.min_over(self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let data =
      lhs.unsqueeze(0).iter(self.dim as isize)
      .zip(grad.unsqueeze(0).iter(self.dim as isize))
      .flat_map(|(t, g)| {
        let rows: Vec<_> = t.iter(0).collect();
        let argmin = Tensor::linearize(&rows, |col| {
          col.iter()
            .enumerate()
            .min_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
            .map(|(i, _)| i )
            .unwrap()
        });
        (0..rows.len()).flat_map(move |i| {
          argmin.param_iter().zip(g.param_iter()).map(|(a, b)|
            if a == i { b } else { T::zero() }
          ).collect::<Vec<_>>()
        })
      }).collect();
    Tensor::new(&lhs.shape().dims, data)
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::MinOver(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxOver {
  dim: isize,
}

impl<T: Real> UnaryOp<T> for MaxOver {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.max_over(self.dim)
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let data =
      lhs.unsqueeze(0).iter(self.dim as isize)
      .zip(grad.unsqueeze(0).iter(self.dim as isize))
      .flat_map(|(t, g)| {
        let rows: Vec<_> = t.iter(0).collect();
        let argmax = Tensor::linearize(&rows, |col| {
          col.iter()
            .enumerate()
            .max_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
            .map(|(i, _)| i )
            .unwrap()
        });
        (0..rows.len()).flat_map(move |i| {
          argmax.param_iter().zip(g.param_iter()).map(|(a, b)|
            if a == i { b } else { T::zero() }
          ).collect::<Vec<_>>()
        })
      }).collect();
    Tensor::new(&lhs.shape().dims, data)
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::MaxOver(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU;

impl<T: Real> UnaryOp<T> for ReLU {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.relu()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let zero = T::zero();
    grad.zip(lhs, |(g, x)| if x > zero { g } else { zero })
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::ReLU(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sigmoid;

impl<T: Real> UnaryOp<T> for Sigmoid {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.sigmoid()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let one = T::one();
    grad.zip(lhs, |(g, x)| {
      let sig = one / (one + (-x).exp());
      g * sig * (one - sig)
    })
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::Sigmoid(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiLU;

impl<T: Real> UnaryOp<T> for SiLU {
  fn run(&self, lhs: &Tensor<T>) -> Tensor<T> {
    lhs.silu()
  }

  fn derive(&self, lhs: &Tensor<T>, grad: &Tensor<T>) -> Tensor<T> {
    let one = T::one();
    grad.zip(lhs, |(g, x)| {
      let sig = one / (one + (-x).exp());
      g * (sig + x * sig * (one - sig))
    })
  }

  fn as_enum(self) -> UnaryMops { UnaryMops::SiLU(self) }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnaryMops {
  Range(Range),
  Broadcast(Broadcast),
  Reshape(Reshape),
  Transpose(Transpose),
  Layout(Layout),
  Sin(Sin),
  Cos(Cos),
  Tanh(Tanh),
  Log(Log),
  Sum(Sum),
  Abs(Abs),
  Min(Min),
  Max(Max),
  MinOver(MinOver),
  MaxOver(MaxOver),
  ReLU(ReLU),
  Sigmoid(Sigmoid),
  SiLU(SiLU),
}

impl UnaryMops {
  pub fn as_unary_op<T: Real>(&self) -> &dyn UnaryOp<T> {
    match self {
      Self::Range(op) => op,
      Self::Broadcast(op) => op,
      Self::Reshape(op) => op,
      Self::Transpose(op) => op,
      Self::Layout(op) => op,
      Self::Sin(op) => op,
      Self::Cos(op) => op,
      Self::Tanh(op) => op,
      Self::Log(op) => op,
      Self::Sum(op) => op,
      Self::Abs(op) => op,
      Self::Min(op) => op,
      Self::Max(op) => op,
      Self::MinOver(op) => op,
      Self::MaxOver(op) => op,
      Self::ReLU(op) => op,
      Self::Sigmoid(op) => op,
      Self::SiLU(op) => op,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryMops {
  Add(Add),
  Sub(Sub),
  Mul(Mul),
  Div(Div),
  Rem(Rem),
  MatMul(MatMul),
  AssignMasked(AssignMasked),
  LookUp(LookUp),
  Pow(Pow),
}

impl BinaryMops {
  pub fn as_binary_op<T: Real>(&self) -> &dyn BinaryOp<T> {
    match self {
      Self::Add(op) => op,
      Self::Sub(op) => op,
      Self::Mul(op) => op,
      Self::Div(op) => op,
      Self::Rem(op) => op,
      Self::MatMul(op) => op,
      Self::AssignMasked(op) => op,
      Self::LookUp(op) => op,
      Self::Pow(op) => op,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiMops {
  Stack(Stack)
}

impl MultiMops {
  pub fn as_multi_op<T: Real>(&self) -> &dyn MultiOp<T> {
    match self {
      Self::Stack(op) => op,
    }
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn mm_gradient() {
    // 1. Matrix × Matrix
    let err_mat_mat = Variable::check_gradients(&[2, 3], |x| {
      let y = Tensor::new(&[3, 4], vec![
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
      ]).tracked();
      x.mm(&y)
    });
    assert!(err_mat_mat < 1e-2);

    // 2. Matrix × Vector
    let err_mat_vec = Variable::check_gradients(&[2, 3], |x| {
      let y = Tensor::new(&[3], vec![1.0, 2.0, 3.0]).tracked();
      x.mm(&y)
    });
    assert!(err_mat_vec < 1e-2);

    // 3. Vector × Matrix
    let err_vec_mat = Variable::check_gradients(&[3], |x| {
      let y = Tensor::new(&[3, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
      ]).tracked();
      x.mm(&y)
    });
    assert!(err_vec_mat < 1e-2);

    // 4. Batched Matrix × Broadcasted Matrix
    let err_batched = Variable::check_gradients(&[2, 2, 3], |x| {
      let y = Tensor::new(&[3, 1], vec![1.0, 2.0, 3.0]).tracked();
      x.mm(&y)
    });
    assert!(err_batched < 1e-2);

    // 5. Batched Matrix × Vector (vector broadcasted across batch)
    let err_batched_vec = Variable::check_gradients(&[2, 3], |x| {
      let y = Tensor::new(&[3], vec![1.0, 2.0, 3.0]).tracked();
      x.mm(&y)
    });
    assert!(err_batched_vec < 1e-2);
  }

  #[test]
  fn max_over_backward() {
    let a = Tensor::new(&[2, 3, 2], vec![
      1.0, 4.0,
      3.0, 2.0,
      0.0, 1.0,
      5.0, 1.0,
      2.0, 6.0,
      3.0, 4.0
    ]).trained();

    let out = a.max_over(1);
    out.backward();
    let grad = a.grad().unwrap();

    let expected = Tensor::new(&[2, 3, 2], vec![
      0.0, 1.0,
      1.0, 0.0,
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      0.0, 0.0
    ]);

    assert_eq!(*grad, expected);
  }
}
