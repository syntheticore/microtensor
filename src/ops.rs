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

pub trait BaseOps<I: Inner>: Clone {
  fn scalar(item: I) -> Self;
  fn shape(&self) -> &Shape;
  fn broadcast(&self, shape: &Shape) -> Self;
  fn reshape(&self, dims: &[usize]) -> Self;
  fn unsqueeze(&self, dim: isize) -> Self;
  fn transpose(&self, dim1: isize, dim2: isize) -> Self;
  fn concat(&self, rhs: &Self, dim: isize) -> Self;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Numeric] inner types.

pub trait NumericOps<I: Numeric>: NumOps + NumOps<I, Self> + Sized {
  fn sum(&self, dim: isize) -> Self;
  // sum_over or generic form of sum etc., like sum(&[1,2])
  fn mm(&self, rhs: &Self) -> Self;
  fn min(&self, dim: isize) -> Self;
  fn max(&self, dim: isize) -> Self;
}


/// Differentiable mid-level operations that are also implemented
/// for non-differentiable [Signed] inner types.

pub trait SignedOps<I: Signed>: std::ops::Neg {
  fn abs(&self) -> Self;
}


/// Differentiable mid-level operations.

pub trait RealOps<I: Real>: std::ops::Neg {
  fn pow(&self, rhs: &Self) -> Self;
  fn sin(&self) -> Self;
  fn cos(&self) -> Self;
  fn relu(&self) -> Self;
  fn sigmoid(&self) -> Self;
}


/// High-level operations, implemented exclusively on top of
/// Mops and other Hops. As a result, these are all
/// differentiable when called on a [Variable](crate::Variable).

pub trait Hops<I>: BaseOps<I> + NumericOps<I> + SignedOps<I> + RealOps<I>
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

  fn mean(&self, dim: isize) -> Self {
    let udim = negative_index(dim, self.shape().rank(), false);
    let n: usize = self.shape().dims[udim..].iter().product();
    let n = I::from(n).unwrap();
    self.sum(dim) / n
  }

  // mean_over

  fn variance(&self, dim: isize) -> Self {
    (self - &self.mean(dim).unsqueeze(-1)).sqr().mean(dim)
  }

  fn softmax(&self, dim: isize) -> Self {
    let exp = (self - &self.max(dim).unsqueeze(-1)).exp();
    &exp / &exp.sum(dim).unsqueeze(-1)
  }

  fn dot(&self, rhs: &Self, dim: isize) -> Self {
    (self * rhs).sum(dim)
  }
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::Tensor;

  #[test]
  fn mean() {
    let a = Tensor::new(&[3,2], vec![1., 2., 3., 4., 5., 6.]).trained();
    assert_eq!(a.mean(0).tensor(), &Tensor::vec(&[3.5]));
    assert_eq!(a.mean(-1).tensor(), &Tensor::vec(&[1.5, 3.5, 5.5]));
  }

  #[test]
  fn softmax() {
    let a = Tensor::arrange(&[3,2], 1.0, 1.0).softmax(-1);
    for row in a.iter(0) {
      assert_eq!(row.sum(0).item(), 1.0);
    }
  }
}
