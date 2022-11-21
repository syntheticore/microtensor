use crate::{
  tensor::Tensor,
  scalar::{ Numeric },
  ops::{ Cops },
};


impl<T: Numeric> Cops<T> for Tensor<T> {
  default fn matmul(&self, rhs: &Self) -> Vec<T> {
    let lhs = self.contiguous();
    let rhs = rhs.contiguous();

    let rows_l = lhs.shape[-2];
    let rows_r = rhs.shape[-2];
    let cols_l = lhs.shape[-1];
    let cols_r = rhs.shape[-1];

    let data_l = lhs.data.borrow();
    let data_r = rhs.data.borrow();
    let offset_l = lhs.shape.offset;
    let offset_r = rhs.shape.offset;

    let mut data = vec![T::zero(); rows_l * cols_r];
    for i in 0..rows_l {
      for j in 0..cols_r {
        for k in 0..rows_r {
          data[i * cols_r + j] +=
            data_l[offset_l + i * cols_l + k] *
            data_r[offset_r + k * cols_r + j];
        }
      }
    }

    data
  }
}

impl Cops<f32> for Tensor<f32> {
  fn matmul(&self, rhs: &Self) -> Vec<f32> {
    let rows_l = self.shape[-2];
    let cols_l = self.shape[-1];
    let cols_r = rhs.shape[-1];

    let mut data = vec![0.0; rows_l * cols_r];

    unsafe {
      matrixmultiply::sgemm(
        rows_l,
        cols_l,
        cols_r,
        1.0,
        self.raw_mut().as_mut_ptr().add(self.shape.offset),
        self.shape.strides[0],
        self.shape.strides[1],
        rhs.raw_mut().as_mut_ptr().add(rhs.shape.offset),
        rhs.shape.strides[0],
        rhs.shape.strides[1],
        0.0,
        data.as_mut_ptr(),
        cols_r as isize,
        1,
      );
    };

    data
  }
}

impl Cops<f64> for Tensor<f64> {
  fn matmul(&self, rhs: &Self) -> Vec<f64> {
    let rows_l = self.shape[-2];
    let cols_l = self.shape[-1];
    let cols_r = rhs.shape[-1];

    let mut data = vec![0.0; rows_l * cols_r];

    unsafe {
      matrixmultiply::dgemm(
        rows_l,
        cols_l,
        cols_r,
        1.0,
        self.raw_mut().as_mut_ptr().add(self.shape.offset),
        self.shape.strides[0],
        self.shape.strides[1],
        rhs.raw_mut().as_mut_ptr().add(rhs.shape.offset),
        rhs.shape.strides[0],
        rhs.shape.strides[1],
        0.0,
        data.as_mut_ptr(),
        cols_r as isize,
        1,
      );
    };

    data
  }
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::ops::NumericOps;

  #[test]
  fn matmul() {
    let x = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    let y = Tensor::new(&[3,2], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,2], vec![22, 28, 49, 64]));
  }

  #[test]
  fn matmul_vector() {
    let x = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);

    let y = Tensor::new(&[3,1], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,1], vec![14, 32]));

    // let y = Tensor::new(&[3], vec![1, 2, 3]);
    // assert_eq!(x.mm(&y), Tensor::new(&[2], vec![14, 32]));
  }

  #[test]
  #[ignore]
  fn matmul_vector_broadcast() {
    let x = Tensor::arrange(&[2,2,3], 1, 1);

    let y = Tensor::new(&[3,1], vec![1, 2, 3]);
    assert_eq!(x.mm(&y), Tensor::new(&[2,2,1], vec![14, 32, 50, 68]));

    // let y = Tensor::new(&[3], vec![1, 2, 3]);
    // assert_eq!(x.mm(&y), Tensor::new(&[2,2], vec![14, 32, 50, 68]));

    let x = Tensor::new(&[2], vec![1, 2]);
    let y = Tensor::new(&[2,3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.mm(&y), Tensor::new(&[3], vec![9, 12, 15]));
  }
}
