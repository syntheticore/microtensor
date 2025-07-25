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

    let data_l = lhs.raw();
    let data_r = rhs.raw();
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

macro_rules! matmul {
  ($T:ident, $method:ident) => {

    #[cfg(feature = "unsafe")]
    impl Cops<$T> for Tensor<$T> {

      fn matmul(&self, rhs: &Self) -> Vec<$T> {
        let rows_l = self.shape[-2];
        let cols_l = self.shape[-1];
        let cols_r = rhs.shape[-1];

        let mut data = vec![0.0; rows_l * cols_r];

        unsafe {
          matrixmultiply::$method(
            rows_l,
            cols_l,
            cols_r,
            1.0,
            self.raw().as_ptr().add(self.shape.offset),
            self.shape.strides[0],
            self.shape.strides[1],
            rhs.raw().as_ptr().add(rhs.shape.offset),
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
  };
}

matmul!(f32, sgemm);
matmul!(f64, dgemm);
