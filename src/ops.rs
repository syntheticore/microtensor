use std::ops::Range;

use num_traits::NumOps;

use crate::{
  internal::*,
  Shape,
  scalar::{ Inner, Numeric, Signed, Real },
};


/// Low-level compute operations.

pub trait Cops<I: Numeric> {
  fn matmul(&self, rhs: &Self) -> Vec<I>;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Inner] types.

pub trait BaseOps<I: Inner>: Clone + std::fmt::Display + std::fmt::Debug {
  fn scalar(item: I) -> Self;
  fn fill(shape: &[usize], filler: I) -> Self;
  fn item(&self) -> I;
  fn shape(&self) -> &Shape;
  fn range(&self, ranges: &[Range<isize>]) -> Self;
  fn broadcast(&self, shape: &Shape, ignore_from: Option<isize>) -> Self;
  fn reshape(&self, dims: &[usize]) -> Self;
  fn squeeze(&self, dims: &[isize]) -> Self;
  fn unsqueeze(&self, dim: isize) -> Self; //XXX multiple dims
  fn transpose(&self, dim1: isize, dim2: isize) -> Self;
  fn stack(inputs: &[Self], dim: isize) -> Self;
  fn assign_masked(&self, rhs: &Self, cb: impl Fn(&Shape) -> Shape) -> Self;
  fn layout(&self, cb: impl Fn(&Shape) -> Shape) -> Self;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Numeric] inner types.

pub trait NumericOps<I: Numeric>: BaseOps<I> + NumOps + NumOps<I, Self> + Sized {
  fn sum(&self, dim: isize) -> Self; //XXX keep_dim option
  // sum_over or generic form of sum etc., like sum(&[1,2])
  fn mm(&self, rhs: &Self) -> Self;
  fn min(&self, dim: isize) -> Self;
  fn max(&self, dim: isize) -> Self;
  fn min_over(&self, dim: isize) -> Self;
  fn max_over(&self, dim: isize) -> Self;
  fn look_up(&self, rhs: &Self) -> Self;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Signed] inner types.

pub trait SignedOps<I: Signed>: NumericOps<I> + std::ops::Neg {
  fn abs(&self) -> Self;
}


/// Differentiable mid-level operations.

pub trait RealOps<I: Real>: NumericOps<I> + SignedOps<I> {
  fn pow(&self, rhs: &Self) -> Self;
  fn sin(&self) -> Self;
  fn cos(&self) -> Self;
  fn tanh(&self) -> Self;
  fn log(&self) -> Self;
  fn relu(&self) -> Self;
  fn sigmoid(&self) -> Self;
}


fn make_ranges(indices: &[isize], shape: &Shape) -> Vec<Range<isize>> {
  debug_assert!(indices.len() <= shape.rank(), "Too many indices {indices:?} for {shape} Tensor");
  indices.iter()
    .zip(&shape.dims)
    .map(|(&idx, &dim)| {
      let idx = negative_index(idx, dim, false) as isize;
      idx .. idx + 1
    })
    .collect()
}

/// High-level operations, implemented exclusively on top of
/// Mops and other Hops. As a result, these are all
/// differentiable when called on a [Variable](crate::Variable).

pub trait BaseHops<I: Inner>: BaseOps<I> {
  fn rank(&self) -> usize {
    self.shape().rank()
  }

  fn dim(&self, i: isize) -> usize {
    self.shape()[i]
  }

  fn size(&self) -> usize {
    self.shape().size()
  }

  fn range_back(&self, ranges: &[Range<isize>]) -> Self {
    let full_slices = self.rank() - ranges.len();
    let mut vec = vec![0..-1; full_slices];
    vec.append(&mut ranges.to_vec());
    self.range(&vec)
  }

  fn at(&self, indices: &[isize]) -> Self {
    let ranges = make_ranges(indices, self.shape());
    self.range(&ranges).squeeze_first(indices.len())
  }

  fn at_back(&self, indices: &[isize]) -> Self {
    self.range_back(&make_ranges(indices, self.shape()))
  }

  fn select(&self, dim: isize, index: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    let ranges = [vec![0..-1; dim], vec![index..index + 1]].concat();
    self.range(&ranges).squeeze(&[dim as isize])
  }

  fn set(&mut self, indices: &[isize], other: &Self) -> Self {
    self.assign_masked(other, |shape|
      shape
        .range(&make_ranges(indices, shape))
        .squeeze_first(indices.len())
    )
  }

  fn squeeze_only(&self, dim: isize) -> Self {
    self.squeeze(&[dim])
  }

  fn squeeze_but(&self, dim: isize) -> Self {
    let rank = self.rank();
    let dim = negative_index(dim, rank, false) as isize;
    let dims: Vec<_> = (0..rank as isize)
      .filter(|&d| d != dim )
      .collect();
    self.squeeze(&dims)
  }

  fn squeeze_first(&self, n: usize) -> Self {
    let dims: Vec<_> = (0..n as isize).collect();
    self.squeeze(&dims)
  }

  fn squeeze_all(&self) -> Self {
    self.squeeze_first(self.rank())
  }

  fn unsqueeze_n(&self, n: usize, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), true) as isize;
    let mut out = self.clone();
    for _ in 0..n {
      out = out.unsqueeze(dim);
    }
    out
  }

  fn extend(&self, rank: usize) -> Self {
    let n = rank - self.rank();
    self.unsqueeze_n(n, -1)
  }

  fn concat(&self, rhs: &Self, dim: isize) -> Self {
    Self::stack(&[self.clone(), rhs.clone()], dim)
  }

  fn rows(rows: &[Self]) -> Self {
    let rows: Vec<_> = rows.iter()
      .map(|row| row.unsqueeze(0) )
      .collect();
    Self::stack(&rows, 0)
  }

  fn split(&self, size: usize, dim: isize) -> Vec<Self> {
    let n = self.dim(dim) / size;
    let remainder = self.dim(dim) % size;
    let dim = negative_index(dim, self.rank(), false);
    let slices = (0..n as isize)
      .map(|i| {
        let j = i * size as isize;
        let mut ranges = vec![0..-1; dim + 1];
        ranges[dim] = j .. j + size as isize;
        self.range(&ranges)
      })
      .collect();
    if remainder == 0 {
      slices
    } else {
      let j = n as isize * size as isize;
      let mut ranges = vec![0..-1; dim + 1];
      ranges[dim] = j .. j + remainder as isize;
      [slices, vec![self.range(&ranges)]].concat()
    }
  }

  fn chunks(&self, n: usize, dim: isize) -> Vec<Self> {
    let size = self.dim(dim) / n;
    self.split(size, dim)
  }

  fn reshape_keep(&self, dims: &[isize]) -> Self {
    let dims: Vec<_> = dims.iter()
      .enumerate()
      .map(|(i, &n)| if n == -1 { self.shape()[i as isize] } else { n as usize })
      .collect();
    self.reshape(&dims)
  }

  fn flatten(&self, keep_dims: usize) -> Self {
    let dims = [vec![-1; keep_dims], vec![0]].concat();
    self.reshape_keep(&dims).squeeze_all()
  }

  fn windows(&self, shape: [usize; 2], step: [usize; 2]) -> Self {
    self.layout(|shap| shap.windows(shape, step) )
  }

  fn repeat(&self, count: usize, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    let mut dims = vec![1; self.rank() + 1];
    dims[dim] = count;
    self.unsqueeze(dim as isize).broadcast(&Shape::new(&dims), None)
  }

  // fn transpose_vec(&self, extend_front: bool) -> Self {
  //   let mut this = self.clone();
  //   if self.rank() == 1 {
  //     this = self.unsqueeze(if extend_front { -2 } else { -1 })
  //   }
  //   this.transpose(-1, -2)
  // }
}


/// High-level operations, implemented exclusively on top of
/// Mops and other Hops. As a result, these are all
/// differentiable when called on a [Variable](crate::Variable).

pub trait NumericHops<I>: NumericOps<I> + BaseHops<I>
where
  I: Numeric,
  for<'a> &'a Self: NumOps<&'a Self, Self> + NumOps<I, Self>,
{
  fn ones(shape: &[usize]) -> Self {
    Self::fill(shape, I::one())
  }

  fn zeros(shape: &[usize]) -> Self {
    Self::fill(shape, I::zero())
  }

  fn zeros_like(other: &Self) -> Self {
    Self::zeros(&other.shape().dims)
  }

  fn min_with(&self, rhs: &Self) -> Self {
    self
      .broadcast(rhs.shape(), None).unsqueeze(0)
      .concat(&rhs.broadcast(self.shape(), None).unsqueeze(0), 0)
      .min_over(0)
      .squeeze_only(0)
  }

  fn max_with(&self, rhs: &Self) -> Self {
    self
      .broadcast(rhs.shape(), None).unsqueeze(0)
      .concat(&rhs.broadcast(self.shape(), None).unsqueeze(0), 0)
      .max_over(0)
      .squeeze_only(0)
  }

  fn masked(dims: &[usize], source: &Self, cb: impl Fn(&Shape) -> Shape) -> Self {
    let shape = Shape::new(&dims);
    let mask = cb(&shape);
    Self::scalar(I::zero())
      .broadcast(&shape, None)
      .assign_masked(&source.broadcast(&mask, None), |_| mask.clone() )
  }

  fn max_pool(&self, kernel: [usize; 2]) -> Self {
    self.windows(kernel, kernel).max(-2)
  }

  fn pad(&self, padding: &[usize]) -> Self {
    let mut dims = self.shape().dims.clone();
    let dimlen = dims.len();
    let padlen = padding.len();
    for i in 0..padlen {
      let p = padding[i];
      dims[dimlen - (padlen - i)] += p * 2;
    }
    let ranges: Vec<_> = padding.iter().map(|&p| p as isize .. -1 - p as isize ).collect();
    Self::masked(&dims, &self, |mask| mask.range_back(&ranges) )
  }

  fn upscale(&self, kernel: [usize; 2]) -> Self {
    let mut dims = self.shape().dims.clone();
    let len = dims.len();
    dims[len - 2] *= kernel[0];
    dims[len - 1] *= kernel[1];
    Self::masked(
      &dims,
      &self.unsqueeze_n(2, -1),
      |mask| mask.windows(kernel, kernel)
    )
  }
}


/// High-level operations, implemented exclusively on top of
/// Mops and other Hops. As a result, these are all
/// differentiable when called on a [Variable](crate::Variable).

pub trait RealHops<I>: RealOps<I> + NumericHops<I>
where
  I: Real,
  for<'a> &'a Self: NumOps<&'a Self, Self> + NumOps<I, Self>,
{
  fn powf(&self, exp: I) -> Self {
    self.pow(&Self::scalar(exp))
  }

  fn sqr(&self) -> Self {
    self.powf(I::from(2.0).unwrap())
  }

  fn sqrt(&self) -> Self {
    self.powf(I::from(0.5).unwrap())
  }

  fn exp(&self) -> Self {
    let e = I::from(std::f64::consts::E).unwrap();
    Self::scalar(e).pow(self)
  }

  fn norm(&self, dim: isize) -> Self {
    self.sqr().sum(dim).sqrt()
  }

  fn dot(&self, rhs: &Self, dim: isize) -> Self {
    (self * rhs).sum(dim)
  }

  fn mean(&self, dim: isize) -> Self {
    let udim = negative_index(dim, self.rank(), false);
    let n: usize = self.shape().dims[udim..].iter().product();
    let n = I::from(n).unwrap();
    self.sum(dim) / n
  }

  // mean_over

  fn variance(&self, dim: isize) -> Self {
    let mean = self.mean(dim).extend(self.rank());
    (self - &mean).sqr().mean(dim)
  }

  fn clamp_min(&self, min: I) -> Self {
    self.max_with(&Self::scalar(min))
  }

  fn swish(&self, beta: &Self) -> Self {
    self * &(self * beta).sigmoid()
  }

  fn silu(&self) -> Self {
    self * &self.sigmoid()
  }

  fn softmax(&self, dim: isize) -> Self {
    let max = self.max(dim).extend(self.rank());
    let exp = (self - &max).exp();
    &exp / &exp.sum(dim).extend(exp.rank())
  }

  fn convolve2d(&self, kernels: &Self, step: [usize; 2], bias: Option<&Self>, padding: bool) -> Self {
    let kernel_width = kernels.dim(-2);
    let kernel_height = kernels.dim(-1);

    let padded = if padding {
      self.pad(&[(kernel_width - 1) / 2, (kernel_height - 1) / 2])
    } else {
      self.clone()
    };

    let out_width = (padded.dim(-2) - kernel_width) + 1;
    let out_height = (padded.dim(-1) - kernel_height) + 1;

    let windows = padded
      .windows(kernels.shape()[-2..-1].try_into().unwrap(), step)
      .reshape_keep(&[-1, -1, 0, (kernel_width * kernel_height) as isize])
      .transpose(-2, -1);

    let flat_kernels = kernels
      .unsqueeze(-3)
      .reshape_keep(&[-1, -1, 0])
      .transpose(0, 1);

    let conv = flat_kernels.mm(&windows);

    let mut conv = conv
      .transpose(1, 2)
      .transpose(-2, -1)
      .sum(-1);

    if let Some(bias) = bias {
      conv = conv + bias.unsqueeze(-1);
    }

    conv.reshape(&[padded.dim(0), kernels.dim(0), out_width, out_height])
  }

  fn attention(&self, key: &Self, value: &Self, head_dim: usize, mask: &Self) -> Self {
    let scale = I::one() / I::from(head_dim).unwrap().sqrt();
    let scores = self.mm(&key.transpose(-2, -1)) * scale;
    (&scores + mask)
      .softmax(-1)
      .mm(value)
  }

  fn cross_entropy(&self, other: &Self) -> Self {
    let this = self + I::from(1e-9).unwrap();
    (&this.log() * other).sum(-1) * (-I::one())
  }

  fn mse(&self, other: &Self) -> Self {
    (self - other).sqr().mean(0)
  }

  fn mae(&self, other: &Self) -> Self {
    (self - other).abs().mean(0)
  }

  fn huber(&self, other: &Self, delta: I) -> Self {
    let half = I::from(0.5).unwrap();
    let diff = self - other;
    let mae = diff.abs().mean(0);
    if mae.item() < delta {
      diff.sqr().mean(0) * half
    } else {
      (mae - (half * delta)) * delta
    }
  }
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::Tensor;

  #[test]
  fn mean() {
    let a = Tensor::new(&[3,2], vec![1., 2., 3., 4., 5., 6.]).trained();
    assert_eq!(a.mean(0).tensor(), &Tensor::scalar(3.5));
    assert_eq!(a.mean(-1).tensor(), &Tensor::vec(&[1.5, 3.5, 5.5]));
  }

  #[test]
  fn softmax() {
    let a = Tensor::arrange(&[3,2], 1.0, 1.0).softmax(-1);
    for row in a.iter(0) {
      assert_eq!(row.sum(0).item(), 1.0);
    }
  }

  #[test]
  fn stack() {
    let a = Tensor::stack(&[
      Tensor::arrange(&[1,2], 1, 1),
      Tensor::arrange(&[3,2], 3, 1),
    ], 0);
    assert_eq!(a, Tensor::new(&[4,2], vec![1, 2, 3, 4, 5, 6, 7, 8]));
  }

  #[test]
  fn max_with() {
    let a = Tensor::arrange(&[3,2,2], 0, 1);
    let b = Tensor::arrange(&[3,1,2], 13, 1);
    assert_eq!(a.max_with(&b), Tensor::new(&[3, 2, 2], vec![13, 14, 13, 14, 15, 16, 15, 16, 17, 18, 17, 18]));
  }
}
