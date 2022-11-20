use crate::{
  shape::Shape,
  tensor::Tensor,
  scalar::{ Inner, Numeric, Signed, Real },
  ops::{ Cops, BaseOps, NumericOps, SignedOps, RealOps },
};


impl<T: Inner> BaseOps<T> for Tensor<T> {
  fn scalar(item: T) -> Self {
    Self::vec(&[item])
  }

  fn shape(&self) -> &Shape {
    &self.shape
  }

  fn broadcast(&self, rhs: &Self) -> Self {
    Self {
      shape: self.shape.broadcast(&rhs.shape),
      data: self.data.clone(),
    }
  }

  fn reshape(&self, shape: &[usize]) -> Self {
    self.contiguous().view(shape)
  }

  fn unsqueeze(&self, dim: isize) -> Self {
    let shape = self.shape.unsqueeze(dim);
    let data = self.data.clone();
    Self { shape, data }
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
    let pad_l = lhs.rank() == 1;
    let pad_r = rhs.rank() == 1;
    if pad_l {
      lhs = lhs.unsqueeze(0);
    }
    if pad_r {
      // rhs = rhs.unsqueeze(-1);
      rhs = rhs.unsqueeze(0);
    }

    // Form result shape
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

    let data = lhs.matmul(&rhs);

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
  ($trait:ident, $trait_meth:ident, $meth:ident, $symbol:tt) => {
    impl<T: Numeric> std::ops::$trait for &Tensor<T> { // &self * &other
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: Self) -> Tensor<T> {
        self.$meth(rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait for Tensor<T> { // tensor * other
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: Self) -> Tensor<T> {
        (&self).$meth(&rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<Tensor<T>> for &Tensor<T> { // &tensor * other
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: Tensor<T>) -> Tensor<T> {
        (self).$meth(&rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<&Tensor<T>> for Tensor<T> { // tensor * &other
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: &Tensor<T>) -> Tensor<T> {
        (&self).$meth(rhs)
      }
    }

    impl<T: Numeric> std::ops::$trait<T> for &Tensor<T> { // &tensor * T
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: T) -> Tensor<T> {
        (self).$meth(&Tensor::scalar(rhs))
      }
    }

    impl<T: Numeric> std::ops::$trait<T> for Tensor<T> { // tensor * T
      type Output = Tensor<T>;

      fn $trait_meth(self, rhs: T) -> Tensor<T> {
        (&self).$meth(&Tensor::scalar(rhs))
      }
    }

    impl std::ops::$trait<&Tensor<f32>> for f32 { // f32 * &tensor
      type Output = Tensor<f32>;

      fn $trait_meth(self, tensor: &Tensor<f32>) -> Tensor<f32> {
        Tensor::scalar(self) $symbol tensor
      }
    }

    impl std::ops::$trait<Tensor<f32>> for f32 { // f32 * tensor
      type Output = Tensor<f32>;

      fn $trait_meth(self, tensor: Tensor<f32>) -> Tensor<f32> {
        Tensor::scalar(self) $symbol &tensor
      }
    }
  };
}

add_operator!(Add, add, add, +);
add_operator!(Sub, sub, sub, -);
add_operator!(Mul, mul, mul, *);
add_operator!(Div, div, div, /);
add_operator!(Rem, rem, mm, %);


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sum() {
    let a = Tensor::new(&[3,2], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a.sum(0), Tensor::new(&[], vec![21]));
    assert_eq!(a.sum(-1), Tensor::new(&[3], vec![3, 7, 11]));
  }
}
