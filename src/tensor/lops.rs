use std::ops::Range;

use crate::{
  internal::*,
  shape::Shape,
  tensor::Tensor,
  scalar::{ Inner, Numeric, Signed, Real },
  ops::{ Cops, BaseOps, NumericOps, SignedOps, RealOps, BaseHops },
};


impl<T: Inner> BaseOps<T> for Tensor<T> {
  fn scalar(item: T) -> Self {
    Self::vec(&[item])
  }

  fn shape(&self) -> &Shape {
    &self.shape
  }

  fn range(&self, ranges: &[Range<isize>]) -> Self {
    let shape = self.shape.range(ranges);
    let data = self.data.clone();
    Self { shape, data }
  }

  fn broadcast(&self, shape: &Shape) -> Self {
    Self {
      shape: self.shape.broadcast(shape),
      data: self.data.clone(),
    }
  }

  fn reshape(&self, shape: &[usize]) -> Self {
    self.contiguous().view(shape)
  }

  fn squeeze(&self, squeezed: &[isize]) -> Self {
    let shape = self.shape.squeeze(squeezed);
    let data = self.data.clone();
    Self { shape, data }
  }

  fn unsqueeze(&self, dim: isize) -> Self {
    let shape = self.shape.unsqueeze(dim);
    let data = self.data.clone();
    Self { shape, data }
  }

  fn transpose(&self, dim1: isize, dim2: isize) -> Self {
    let shape = self.shape.transpose(dim1, dim2);
    let data = self.data.clone();
    Self { shape, data }
  }

  fn concat(&self, rhs: &Self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    let data = if dim == 0 {
      [self.detach().into_raw(), rhs.detach().into_raw()].concat()
    } else {
      let dim = dim as isize - 1;
      self.iter(dim)
        .zip(rhs.iter(dim))
        .flat_map(|(a, b)| vec![a.detach().into_raw(), b.detach().into_raw()] )
        .collect::<Vec<_>>()
        .concat()
    };
    let mut dims_l = self.shape.dims.clone();
    dims_l[dim] += rhs.shape.dims[dim];
    let mut dims_r = rhs.shape.dims.clone();
    dims_r[dim] += self.shape.dims[dim];
    assert_eq!(dims_l, dims_r,
      "Cannot concat {} & {} tensors. Shapes may only differ in dim {}",
      self.shape, rhs.shape, dim);
    Self::new(&dims_l, data)
  }
}

impl<T: Numeric> NumericOps<T> for Tensor<T> {
  fn sum(&self, dim: isize) -> Self {
    self.collapse(dim, |values| values.param_iter().sum() )
  }

  fn mm(&self, rhs: &Self) -> Self {
    let mut lhs = self.clone();
    let mut rhs = rhs.clone();

    // Unsqueeze vector to match matrix
    //XXX allow batched matrix vector multiply
    let pad_l = lhs.rank() == 1;
    let pad_r = rhs.rank() == 1;
    assert!(!(pad_l && pad_r), "Use Tensor::dot to multiply two vectors");
    if pad_l {
      lhs = lhs.unsqueeze(0);
    }
    if pad_r {
      // rhs = rhs.unsqueeze(-1);
      rhs = rhs.unsqueeze(0);
    }

    // Extend with batch dimension
    let no_batch = lhs.rank() == 2 && rhs.rank() == 2;
    lhs = lhs.extend(3);
    rhs = rhs.extend(3);

    // Batch size must be broadcastable
    //XXX allow arbitrary shape before matrix dims
    assert!(lhs.shape.dims[0] == rhs.shape.dims[0] ||
      lhs.shape.dims[0] == 1 || rhs.shape.dims[0] == 1,
      "Could not broadcast batch sizes {} & {}", lhs.shape.dims[0], rhs.shape.dims[0]);

    let batch_size = lhs.shape.dims[0].max(rhs.shape.dims[0]);

    // Squeeze back result
    let rows_l = lhs.shape[-2];
    let cols_r = rhs.shape[-1];
    let dims = if pad_l {
      vec![cols_r]
    } else if pad_r {
      // vec![rows_l]
      vec![rows_l, cols_r]
    } else {
      vec![rows_l, cols_r]
    };

    let dims_b = if no_batch { vec![] } else { vec![batch_size] };
    let dims = [dims_b, dims].concat();

    let data = (0..batch_size).flat_map(|b|
      lhs.at(&[b.min(lhs.shape.dims[0] - 1)]).matmul(
      &rhs.at(&[b.min(rhs.shape.dims[0] - 1)]))
    ).collect();

    Self::new(&dims, data)
  }

  fn min(&self, dim: isize) -> Self {
    self.collapse(dim, |values| {
      values.param_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
        .unwrap()
    })
  }

  fn max(&self, dim: isize) -> Self {
    self.collapse(dim, |values| {
      values.param_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
        .unwrap()
    })
  }
}

impl<T: Signed> SignedOps<T> for Tensor<T> {
  fn abs(&self) -> Self {
    self.vectorize(|a| a.abs() )
  }
}

impl<T: Real> RealOps<T> for Tensor<T> {
  fn pow(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a.powf(b) )
  }

  fn sin(&self) -> Self {
    self.vectorize(|a| a.sin() )
  }

  fn cos(&self) -> Self {
    self.vectorize(|a| a.cos() )
  }

  fn log(&self) -> Self {
    self.vectorize(|a| a.ln() )
  }

  fn relu(&self) -> Self {
    self.vectorize(|a| a.max(T::zero()) )
  }

  fn sigmoid(&self) -> Self {
    self.vectorize(|a| T::one() / (T::one() + (-a).exp()) )
  }
}

impl<T: Signed> std::ops::Neg for &Tensor<T> {
  type Output = Tensor<T>;

  fn neg(self) -> Self::Output {
    self * (-T::one())
  }
}

impl<T: Signed> std::ops::Neg for Tensor<T> {
  type Output = Tensor<T>;

  fn neg(self) -> Self::Output {
    -&self
  }
}

macro_rules! add_operator {
  ($trait:ident, $meth:ident, $symbol:tt) => {
    impl<T: Numeric> std::ops::$trait for &Tensor<T> { // &self * &other
      type Output = Tensor<T>;

      fn $meth(self, rhs: Self) -> Tensor<T> {
        self.$meth(rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait for Tensor<T> { // tensor * other
      type Output = Tensor<T>;

      fn $meth(self, rhs: Self) -> Tensor<T> {
        (&self).$meth(&rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<Tensor<T>> for &Tensor<T> { // &tensor * other
      type Output = Tensor<T>;

      fn $meth(self, rhs: Tensor<T>) -> Tensor<T> {
        (self).$meth(&rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<&Tensor<T>> for Tensor<T> { // tensor * &other
      type Output = Tensor<T>;

      fn $meth(self, rhs: &Tensor<T>) -> Tensor<T> {
        (&self).$meth(rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<T> for &Tensor<T> { // &tensor * T
      type Output = Tensor<T>;

      fn $meth(self, rhs: T) -> Tensor<T> {
        (self).$meth(&Tensor::scalar(rhs))
      }
    }

    impl<T: Numeric> std::ops::$trait<T> for Tensor<T> { // tensor * T
      type Output = Tensor<T>;

      fn $meth(self, rhs: T) -> Tensor<T> {
        (&self).$meth(&Tensor::scalar(rhs))
      }
    }

    impl std::ops::$trait<&Tensor<f32>> for f32 { // f32 * &tensor
      type Output = Tensor<f32>;

      fn $meth(self, tensor: &Tensor<f32>) -> Tensor<f32> {
        Tensor::scalar(self) $symbol tensor
      }
    }

    impl std::ops::$trait<Tensor<f32>> for f32 { // f32 * tensor
      type Output = Tensor<f32>;

      fn $meth(self, tensor: Tensor<f32>) -> Tensor<f32> {
        Tensor::scalar(self) $symbol &tensor
      }
    }
  };
}

add_operator!(Add, add, +);
add_operator!(Sub, sub, -);
add_operator!(Mul, mul, *);
add_operator!(Div, div, /);
add_operator!(Rem, rem, %);


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sum() {
    let a = Tensor::new(&[3,2], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a.sum(0), Tensor::new(&[], vec![21]));
    assert_eq!(a.sum(-1), Tensor::new(&[3], vec![3, 7, 11]));
  }

  #[test]
  fn concat() {
    let a = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    let b = Tensor::new(&[2,3], vec![7, 8, 9, 10, 11, 12]);
    assert_eq!(a.concat(&b, 1), Tensor::new(&[2,6], vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]));

    let b = Tensor::new(&[2,4], vec![7, 8, 9, 10, 11, 12, 13, 14]);
    assert_eq!(a.concat(&b, 1), Tensor::new(&[2,7], vec![1, 2, 3, 7, 8, 9, 10, 4, 5, 6, 11, 12, 13, 14]));
  }

  #[test]
  fn split() {
    let a = Tensor::new(&[2,6], vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]);
    assert_eq!(a.split(3, 1), vec![
      Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]),
      Tensor::new(&[2,3], vec![7, 8, 9, 10, 11, 12]),
    ]);
  }
}
