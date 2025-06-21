use std::ops::Range;

#[cfg(feature = "threading")]
use std::thread;

use crate::{
  internal::*,
  shape::{ Shape, DimensionIterator },
  tensor::Tensor,
  scalar::{ Inner, Numeric, Signed, Real },
  ops::{ Cops, BaseOps, NumericOps, SignedOps, RealOps, BaseHops },
};


impl<T: Inner> BaseOps<T> for Tensor<T> {
  fn scalar(item: T) -> Self {
    Self::new(&[], vec![item])
  }

  fn fill(shape: &[usize], filler: T) -> Self {
    Self::new(shape, vec![filler; shape.iter().product()])
  }

  fn item(&self) -> T {
    debug_assert!(self.shape.squeeze_all().rank() == 0,
      "Can't extract item from non-scalar {}", self.shape);
    self.raw()[self.shape.offset].clone()
  }

  fn shape(&self) -> &Shape {
    &self.shape
  }

  fn range(&self, ranges: &[Range<isize>]) -> Self {
    let shape = self.shape.range(ranges);
    let data = self.data.clone();
    Self { shape, data }
  }

  fn broadcast(&self, shape: &Shape, ignore_from: Option<isize>) -> Self {
    Self {
      shape: self.shape.broadcast(shape, ignore_from),
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

  fn stack(inputs: &[Self], dim: isize) -> Self {
    assert!(inputs.len() >= 1, "Inputs may not be empty");

    let dim = negative_index(dim, inputs[0].rank(), false);
    let mut dims = inputs[0].shape.dims.clone();
    dims[dim] = inputs.iter().map(|input| input.shape.dims[dim] ).sum();

    debug_assert!(inputs.windows(2).all(|w|
      w[0].shape.dims.len() == w[1].shape.dims.len() &&
      w[0].shape.dims.iter().zip(&w[1].shape.dims).filter(|(a,b)| *a == *b ).count() >= dims.len() - 1
    ), "Cannot stack tensors. Shapes may only differ in dim {dim}");

    let data = if dim == 0 {
      inputs.into_iter().flat_map(|input| input.param_iter() ).collect::<Vec<_>>()
    } else {
      let iter = DimensionIterator::new(&inputs[0].shape.dims[0..dim]);
      iter.flat_map(|dims| {
        inputs.iter().flat_map(move |input| input.at(&dims).extract() )
      }).collect::<Vec<_>>()
    };

    Self::new(&dims, data)
  }

  fn assign_masked(&self, other: &Self, cb: impl Fn(&Shape) -> Shape) -> Self {
    if RcT::ptr_eq(&self.data, &other.data) { panic!("Tensor was fed from shared storage") }
    let out = self.detach();
    let mask = cb(out.shape());
    debug_assert!(mask.squeeze_all().dims == other.shape.squeeze_all().dims,
      "Could not assign {} tensor to {} mask", other.shape, mask);
    {
      let mut data = out.raw_mut();
      let other_data = other.raw();
      for (i, j) in mask.iter().zip(other.shape.iter()) {
        data[i] = other_data[j].clone();
      }
    }
    out
  }

  fn layout(&self, cb: impl Fn(&Shape) -> Shape) -> Self {
    //XXX check if shape overflows self's memory
    let this = self.contiguous();
    Self { shape: cb(&this.shape), data: this.data.clone() }
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
      rhs = rhs.unsqueeze(-1);
      // rhs = rhs.unsqueeze(0);
    }

    // Extend with batch dimension
    let no_batch = lhs.rank() == 2 && rhs.rank() == 2;
    if no_batch {
      lhs = lhs.unsqueeze(0);
      rhs = rhs.unsqueeze(0);
    }

    // Batch size must be broadcastable
    let lhs = lhs.broadcast(&rhs.shape, Some(-2));
    let rhs = rhs.broadcast(&lhs.shape, Some(-2));

    assert_eq!(lhs.shape[-1], rhs.shape[-2], "Cannot matrix multiply {} Tensor by {} Tensor", lhs.shape(), rhs.shape());

    // Batched multiplication
    let iter = lhs.iter(-3).zip(rhs.iter(-3));

    #[cfg(feature = "threading")]
    let data = if lhs.shape[-3] == 1 {
      iter
        .flat_map(|(ml, mr)| ml.matmul(&mr) )
        .collect()
    } else {
      thread::scope(|s| {
        iter
          .map(|(ml, mr)|
            s.spawn(move || ml.matmul(&mr) )
          ).collect::<Vec<_>>().into_iter()
          .flat_map(|h| h.join().unwrap() )
          .collect()
      })
    };

    #[cfg(not(feature = "threading"))]
    let data = iter
      .flat_map(|(ml, mr)| ml.matmul(&mr) )
      .collect();

    // Construct result shape
    let rows_l = lhs.shape[-2];
    let cols_r = rhs.shape[-1];
    let last_dims = if pad_l {
      vec![cols_r]
    } else if pad_r {
      // vec![rows_l]
      vec![rows_l, cols_r]
    } else {
      vec![rows_l, cols_r]
    };

    let n = lhs.rank() - 2;
    let start = if no_batch { 1 } else { 0 };
    let mut dims = lhs.shape.dims[start..n].to_vec();
    dims.extend(last_dims);

    if pad_r && dims.last() == Some(&1) { dims.pop(); }

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

  fn min_over(&self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    if dim == self.rank() - 1 {
      // Optimize basic case
      self.min(-1).unsqueeze(-1)
    } else {
      self.collapse_only(dim as isize, |t| {
        let rows: Vec<_> = t.iter(0).collect();
        Tensor::linearize(&rows, |col| *col.iter().min_by(|a, b|
          a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        ).unwrap())
      })
    }
  }

  fn max_over(&self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    if dim == self.rank() - 1 {
      // Optimize basic case
      self.max(-1).unsqueeze(-1)
    } else {
      self.collapse_only(dim as isize, |t| {
        let rows: Vec<_> = t.iter(0).collect();
        Tensor::linearize(&rows, |col| *col.iter().max_by(|a, b|
          a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        ).unwrap())
      })
    }
  }

  fn look_up(&self, tokens: &Self) -> Self {
    let tokens = tokens.cast();
    tokens.expand(|t| self.at(&[t]).extract() )
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

  fn tanh(&self) -> Self {
    self.vectorize(|a| a.tanh() )
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

    impl std::ops::$trait<&Tensor<f64>> for f64 { // f64 * &tensor
      type Output = Tensor<f64>;

      fn $meth(self, tensor: &Tensor<f64>) -> Tensor<f64> {
        Tensor::scalar(self) $symbol tensor
      }
    }

    impl std::ops::$trait<Tensor<f64>> for f64 { // f64 * tensor
      type Output = Tensor<f64>;

      fn $meth(self, tensor: Tensor<f64>) -> Tensor<f64> {
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
  fn mm() {
    let x = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    let y = Tensor::new(&[3,2], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,2], vec![22, 28, 49, 64]));
  }

  #[test]
  fn mm_vector() {
    let x = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    let y = Tensor::new(&[3,1], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,1], vec![14, 32]));
  }

  #[test]
  fn mm_vector2() {
    let x = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    let y = Tensor::new(&[3], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2], vec![14, 32]));
  }

  #[test]
  fn mm_vector_batched() {
    let x = Tensor::new(&[1,2,3], vec![1, 2, 3, 4, 5, 6]);
    let y = Tensor::new(&[1,3,1], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[1,2,1], vec![14, 32]));
  }

  #[test]
  fn mm_vector_broadcast() {
    let x = Tensor::arrange(&[2,2,3], 1, 1);

    let y = Tensor::new(&[3,1], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,2,1], vec![14, 32, 50, 68]));

    let y = Tensor::new(&[3], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,2], vec![14, 32, 50, 68]));
  }

  #[test]
  fn mm_vector_left_squeeze() {
    let x = Tensor::new(&[3], vec![1, 2, 3]);
    let y = Tensor::new(&[3,2], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.mm(&y).shape().dims, vec![2]);
  }

  #[test]
  fn mm_shapes() {
    let t = |shape: &[usize], val: f32| Tensor::fill(shape, val);

    // 1. Matrix × Matrix → [2, 3] × [3, 4] → [2, 4]
    assert_eq!(t(&[2, 3], 1.0).mm(&t(&[3, 4], 1.0)).shape().dims, vec![2, 4]);

    // 2. Vector × Matrix → [3] × [3, 4] → [4]
    assert_eq!(t(&[3], 1.0).mm(&t(&[3, 4], 1.0)).shape().dims, vec![4]);

    // 3. Matrix × Vector → [2, 3] × [3] → [2]
    assert_eq!(t(&[2, 3], 1.0).mm(&t(&[3], 1.0)).shape().dims, vec![2]);

    // 4. Batched Matrix × Matrix → [5, 2, 3] × [5, 3, 4] → [5, 2, 4]
    assert_eq!(t(&[5, 2, 3], 1.0).mm(&t(&[5, 3, 4], 1.0)).shape().dims, vec![5, 2, 4]);

    // 5. Broadcasted batch → [1, 2, 3] × [5, 3, 4] → [5, 2, 4]
    assert_eq!(t(&[1, 2, 3], 1.0).mm(&t(&[5, 3, 4], 1.0)).shape().dims, vec![5, 2, 4]);

    // 6. Vector × Vector → should panic
    assert!(std::panic::catch_unwind(|| t(&[3], 1.0).mm(&t(&[3], 1.0))).is_err());

    // 7. Higher-dimensional broadcasting → [2, 1, 2, 3] × [1, 4, 3, 4] → [2, 4, 2, 4]
    assert_eq!(t(&[2, 1, 2, 3], 1.0).mm(&t(&[1, 4, 3, 4], 1.0)).shape().dims, vec![2, 4, 2, 4]);

    // 8. Standard matrix multiply → [3, 2] × [2, 5] → [3, 5]
    assert_eq!(t(&[3, 2], 1.0).mm(&t(&[2, 5], 1.0)).shape().dims, vec![3, 5]);

    // 9. Explicit unsqueeze vector × matrix → [1, 2] × [2, 3] → [1, 3]
    assert_eq!(t(&[2], 1.0).unsqueeze(0).mm(&t(&[2, 3], 1.0)).shape().dims, vec![1, 3]);

    // 10. Matrix × unsqueezed vector → [3, 2] × [2, 1] → [3, 1]
    assert_eq!(t(&[3, 2], 1.0).mm(&t(&[2], 1.0).unsqueeze(-1)).shape().dims, vec![3, 1]);
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

  #[test]
  fn max() {
    let a = Tensor::arrange(&[3,2,2], 0, 1);
    assert_eq!(a.max(1), Tensor::new(&[3], vec![3, 7, 11]));
  }

  #[test]
  fn max_over() {
    let a = Tensor::arrange(&[3,2,2], 0, 1);
    assert_eq!(a.max_over(1), Tensor::new(&[3, 1, 2], vec![2, 3, 6, 7, 10, 11]));
  }
}
