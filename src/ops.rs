use std::ops::Range;

use num_traits::NumOps;

use crate::internal::*;
use crate::Shape;
use crate::scalar::{ Inner, Numeric, Signed, Real };


/// Low-level compute operations.

pub trait Cops<I: Numeric> {
  fn matmul(&self, rhs: &Self) -> Vec<I>;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Inner] types.

pub trait BaseOps<I: Inner>: Clone + std::fmt::Display {
  fn scalar(item: I) -> Self;
  fn fill(shape: &[usize], filler: I) -> Self;
  fn shape(&self) -> &Shape;
  fn range(&self, ranges: &[Range<isize>]) -> Self;
  fn broadcast(&self, shape: &Shape, ignore_from: Option<isize>) -> Self;
  fn reshape(&self, dims: &[usize]) -> Self;
  fn squeeze(&self, dims: &[isize]) -> Self;
  fn unsqueeze(&self, dim: isize) -> Self; //XXX multiple dims
  fn transpose(&self, dim1: isize, dim2: isize) -> Self;
  fn concat(&self, rhs: &Self, dim: isize) -> Self;
  fn assign_masked(&self, rhs: &Self, shape: &Shape) -> Self;
  fn layout(&self, shape: Shape) -> Self;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Numeric] inner types.

pub trait NumericOps<I: Numeric>: BaseOps<I> + NumOps + NumOps<I, Self> + Sized {
  fn sum(&self, dim: isize) -> Self;
  // sum_over or generic form of sum etc., like sum(&[1,2])
  fn mm(&self, rhs: &Self) -> Self;
  fn min(&self, dim: isize) -> Self;
  fn max(&self, dim: isize) -> Self;
  fn max_over(&self, _dim: isize) -> Self { todo!() }
  // fn compose(dims: &[usize], inputs: &[(Shape, Self)]) -> Self;
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
  fn log(&self) -> Self;
  fn relu(&self) -> Self;
  fn sigmoid(&self) -> Self;
}


fn make_ranges(indices: &[isize], shape: &Shape) -> Vec<Range<isize>> {
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

  fn set(&mut self, indices: &[isize], other: &Self) -> Self {
    self.assign_masked(other, self.at(indices).shape())
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

  fn stack(rows: &[Self], dim: isize) -> Self {
    assert!(rows.len() >= 1);
    let mut out = rows[0].clone();
    for row in &rows[1..] {
      out = out.concat(row, dim);
    }
    out
  }

  fn rows(rows: &[Self]) -> Self {
    assert!(rows.len() >= 1);
    let rows: Vec<_> = rows.iter()
      .map(|row| row.unsqueeze(0) )
      .collect();
    Self::stack(&rows, 0)
  }

  fn reshape_keep(&self, dims: &[isize]) -> Self {
    let dims: Vec<_> = dims.iter()
      .enumerate()
      .map(|(i, &n)| if n == -1 { self.shape()[i as isize] } else { n as usize })
      .collect();
    self.reshape(&dims)
  }

  fn flatten(&self) -> Self {
    self.reshape(&[0]).squeeze_all()
  }

  fn windows(&self, shape: &Shape) -> Self {
    self.layout(self.shape().windows(shape))
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

  fn softmax(&self, dim: isize) -> Self {
    let max = self.max(dim).extend(self.rank());
    let exp = (self - &max).exp();
    &exp / &exp.sum(dim).extend(exp.rank())
  }

  fn max_with(&self, rhs: &Self) -> Self {
    //XXX broadcast
    self.unsqueeze(0).concat(&rhs.unsqueeze(0), 0).max_over(0).squeeze_only(0)
  }

  // fn convolve(&self, kernels: &Self) -> Self {
  //   // self: (batch, width, height, channels)
  //   // kernels: (k, width, height, channels)
  //   let kernel_width = kernels.dim(-3);
  //   let kernel_height = kernels.dim(-2);
  //   let target_width = self.dim(-3) - (kernel_width - 1);
  //   let target_height = self.dim(-2) - (kernel_height - 1);
  //   // let channels = self.dim(-1);
  //   // convolved: (batch, k, width, height)
  //   let dims = &self.shape().dims;
  //   let mut dims = dims[0..dims.len() - 3].to_vec();
  //   dims.append(&mut vec![kernels.dim(0), target_width, target_height]);
  //   let mut convolved = Self::zeros(&dims);
  //   // let mut assignments = vec![];
  //   for x in 0..target_width as isize {
  //     for y in 0..target_height as isize {
  //       let slice = self.range_back(&[
  //         x .. x + kernel_width as isize,
  //         y .. y + kernel_height as isize,
  //         0 .. -1,
  //       ]);
  //       let dot = (&slice.unsqueeze(-4) * kernels).sum(-3); // (batch, w, h, c) -> (batch, k, w, h, c) -> (batch, k)
  //       // convolved.range_back(&[x..x + 1, y..y + 1]).assign(&dot);
  //       let mask = convolved.at_back(&[x, y]);
  //       convolved = convolved.assign_masked(&dot, mask.shape());
  //       // assignments.push((mask.shape().clone(), dot));
  //     }
  //   }
  //   // let convolved = Self::compose(&dims, &assignments);
  //   // (batch, k, width, height) -> (batch, width, height, k)
  //   convolved.unsqueeze(-1).transpose(-4, -1).squeeze(&[-4])
  // }

  fn convolve(&self, kernels: &Self, bias: &Self) -> Self {
    let kernel_width = kernels.dim(-2);
    let kernel_height = kernels.dim(-1);
    let out_width = (self.dim(-2) - kernel_width) + 1;
    let out_height = (self.dim(-1) - kernel_height) + 1;

    let windows = self.windows(kernels.shape());
    let windows = windows.reshape_keep(&[-1, -1, 0, (kernel_width * kernel_height) as isize]);
    let windows = windows.transpose(-2, -1).unsqueeze(2);

    let flat_kernels = kernels.reshape_keep(&[-1, -1, 0]);

    let conv = flat_kernels.mm(&windows);
    let conv = &conv.transpose(-2, -1).squeeze_only(-1).transpose(1, 2).transpose(2, 3).sum(-1) + bias;

    conv.reshape(&[self.dim(0), kernels.dim(0), out_width, out_height])
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
}
