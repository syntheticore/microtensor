use std::ops::Range;
use std::fmt::Debug;

use serde::{Serialize, Deserialize};

use crate::internal::*;


/// The shape of a [Tensor](crate::Tensor).

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape {
  pub dims: Vec<usize>,
  pub(crate) strides: Vec<isize>,
  pub(crate) offset: usize,
}

impl Shape {
  pub fn new(dims: &[usize]) -> Self {
    let strides = Self::make_strides(dims);
    Self {
      dims: dims.to_vec(),
      strides,
      offset: 0,
    }
  }

  pub fn strided(dims: &[usize], strides: &[isize]) -> Self {
    Self {
      dims: dims.to_vec(),
      strides: strides.to_vec(),
      offset: Self::make_offset(dims, strides),
    }
  }

  pub fn offset(dims: &[usize], strides: &[isize], offset: usize) -> Self {
    Self {
      dims: dims.to_vec(),
      strides: strides.to_vec(),
      offset,
    }
  }

  fn make_strides(dims: &[usize]) -> Vec<isize> {
    if dims.len() == 0 { return vec![] }
    let mut strides = vec![0; dims.len()];
    strides[dims.len() - 1] = 1;
    for i in (1..dims.len()).rev() {
      strides[i - 1] = dims[i] as isize * strides[i];
    }
    strides
  }

  fn make_offset(dims: &[usize], strides: &[isize]) -> usize {
    strides.iter()
      .enumerate()
      .map(|(d, &s)| if s < 0 {
        ((dims[d] as isize - 1) * s.abs()) as usize
      } else { 0 }).sum()
  }

  pub fn size(&self) -> usize {
    self.dims.iter().product()
  }

  pub fn rank(&self) -> usize {
    self.dims.len()
  }

  pub(crate) fn index(&self, indices: &[usize]) -> usize {
    assert!(indices.len() <= self.rank());
    // Append missing dimensions as zero
    (indices.iter()
      .chain(std::iter::repeat(&0))
      .zip(&self.strides)
      .map(|(&i, &s)| i as isize * s)
      .sum::<isize>() + self.offset as isize
    ) as usize
  }

  // pub(crate) fn indices(&self, _index: usize) -> Vec<usize> {
  //   todo!()
  // }

  pub fn contiguous(&self) -> bool {
    self.strides == Self::make_strides(&self.dims)
  }

  pub fn iter(&self) -> Box<dyn Iterator<Item=usize> + '_> {
    if self.contiguous() {
      Box::new(self.offset..self.offset + self.size())
    } else {
      Box::new(ShapeIterator::new(self))
    }
  }

  pub fn at_or(&self, idx: isize, or: usize) -> usize {
    let off_bounds = if idx < 0 {
      idx.abs() as usize > self.rank()
    } else {
      idx as usize >= self.rank()
    };
    if off_bounds { or } else { self[idx] }
  }

  pub fn view(&self, shape: &[usize]) -> Self { //XXX allow -1 to keep dimension
    assert!(self.contiguous()); //XXX
    // Calculate size of placeholders
    let dims: Vec<usize> = shape.iter().enumerate().map(|(i, &n)| if n == 0 {
      let product: usize =
        shape[0..i].iter()
        .chain(shape[i + 1..shape.len()].iter())
        .product();
      self.size() / product
    } else {
      n
    }).collect();
    let strides = Self::make_strides(&dims);
    Self { dims, strides, offset: self.offset }
  }

  pub fn take(&self, indices: &[usize]) -> Self {
    let dims = self.dims[indices.len()..].to_vec();
    let strides = self.strides[indices.len()..].to_vec();
    let offset = self.index(indices);
    // // Using all indices results in a scalar -> Wrap it with shape [1]
    // dims.resize(1.max(dims.len()), 1);
    // strides.resize(1.max(strides.len()), 1);
    Self { dims, strides, offset }
  }

  pub fn range(&self, ranges: &[Range<isize>]) -> Self {
    let mut offset = 0;
    let mut dims = self.dims.clone();
    for (d, range) in ranges.iter().enumerate() {
      let dim = self.dims[d];
      let start = negative_index(range.start, dim, true);
      let end = negative_index(range.end, dim, true);
      offset += self.strides[d] * start as isize;
      dims[d] = end - start
    }
    Self { dims, strides: self.strides.clone(), offset: (self.offset as isize + offset) as usize }
  }

  pub fn step(&self, steps: &[isize]) -> Self {
    let strides: Vec<_> = self.strides.iter()
      .zip(steps.iter())
      .map(|(&stride, &step)| stride * step )
      .collect();

    let offset = steps.iter()
      .enumerate()
      .map(|(d, &step)| if step < 0 {
        (self.dims[d] as isize - 1) * strides[d].abs() * self.strides[d].signum()
      } else { 0 })
      .sum::<isize>();

    Self::offset(&self.dims, &strides, (offset + self.offset as isize) as usize)
  }

  pub fn squeeze(&self) -> Self {
    self.squeeze_if(|_| false )
  }

  pub fn squeeze_only(&self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    self.squeeze_if(|d| d != dim )
  }

  pub fn squeeze_but(&self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    self.squeeze_if(|d| d == dim )
  }

  fn squeeze_if(&self, cb: impl Fn(usize) -> bool) -> Self {
    let mut dims = vec![];
    let mut strides = vec![];
    for (d, &n) in self.dims.iter().enumerate() {
      if n != 1 || cb(d) {
        dims.push(n);
        strides.push(self.strides[d]);
      }
    }
    Self { dims, strides, offset: self.offset }
  }

  pub fn unsqueeze(&self, dim: isize) -> Self {
    let d = negative_index(dim, self.rank(), true);
    let mut shape = self.clone();
    shape.strides.insert(d, if d < shape.dims.len() {
      shape.strides[d].abs() * shape.dims[d] as isize
    } else { 1 });
    shape.dims.insert(d, 1);
    shape
  }

  pub fn extend(&self, size: usize) -> Self {
    assert!(self.rank() <= size);
    let mut shape = self.clone();
    for _ in 0..size - self.rank() {
      shape.strides.insert(0, shape.strides[0].abs() * shape.dims[0] as isize);
      shape.dims.insert(0, 1);
    }
    shape
  }

  pub fn broadcast(&self, other: &Self) -> Self {
    let mut dims = vec![];
    let mut strides = vec![];
    self.dims.iter()
      .rev()
      .chain(std::iter::repeat(&1))
      .zip(other.dims.iter()
        .rev()
        .chain(std::iter::repeat(&1)))
      .inspect(|(&a, &b)|
        assert!(a == b || a == 1 || b == 1, "Could not broadcast {} & {}", self, other) )
      .take(self.rank().max(other.rank()))
      .zip(self.strides.iter()
        .rev()
        .chain(std::iter::repeat(&0)))
      .for_each(|((&dl, &dr), &stride)| {
        dims.push(dl.max(dr));
        strides.push(if dl == 1 && dr != 1 { 0 } else { stride });
      });
      let dims: Vec<_> = dims.into_iter().rev().collect();
      let strides: Vec<_> = strides.into_iter().rev().collect();
    Self { dims, strides, offset: self.offset }
  }

  pub fn transpose(&self, dim1: isize, dim2: isize) -> Self {
    let dim1 = negative_index(dim1, self.rank(), false);
    let dim2 = negative_index(dim2, self.rank(), false);
    let mut shape = self.clone();
    shape.dims.swap(dim1, dim2);
    shape.strides.swap(dim1, dim2);
    shape
  }
}

impl std::ops::Index<isize> for Shape {
  type Output = usize;

  fn index<'a>(&'a self, i: isize) -> &'a usize {
    let idx = negative_index(i, self.rank(), false);
    &self.dims[idx]
  }
}

impl std::fmt::Display for Shape {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Shape{:?}", self.dims)
  }
}


/// Iterate through a [Shape]'s indices.

pub struct ShapeIterator<'a> {
  shape: &'a Shape,
  counter: Vec<usize>,
  idx: isize,
  finished: bool,
}

impl<'a> ShapeIterator<'a> {
  fn new(shape: &'a Shape) -> Self {
    Self {
      counter: vec![0; shape.rank()],
      idx: shape.offset as isize,
      shape,
      finished: false,
    }
  }
}

impl<'a> Iterator for ShapeIterator<'a> {
  type Item = usize;

  fn next(&mut self) -> Option<Self::Item> {
    if self.finished { return None }
    let out = self.idx as usize;
    let len = self.counter.len();
    // Walk backward through dimensions
    for cd in (0..len).rev() {
      // Increment counter on full turn of right hand dimension
      if cd == len - 1 || self.counter[cd + 1] == 0 {
        let count = &mut self.counter[cd];
        // Full turn?
        if *count == self.shape.dims[cd] - 1 {
          if cd == 0 { self.finished = true; break }
          *count = 0;
          let backstride = (self.shape.dims[cd] as isize - 1) * self.shape.strides[cd];
          self.idx -= backstride;
        } else {
          *count += 1;
          self.idx += self.shape.strides[cd];
        }
      } else {
        break
      }
    }
    Some(out)
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn strides() {
    let shape = Shape::new(&[3,2,2]);
    assert_eq!(shape.strides, vec![4,2,1]);

    let shape = Shape::new(&[2,3,2]);
    assert_eq!(shape.strides, vec![6,2,1]);
  }

  #[test]
  fn offset() {
    let shape = Shape::strided(&[3,2,2], &[4,2,1]);
    assert_eq!(shape.offset, 0);

    let shape = Shape::strided(&[3,2,2], &[4,-2,1]);
    assert_eq!(shape.offset, 2);

    let shape = Shape::strided(&[2,3,2], &[6,-2,1]);
    assert_eq!(shape.offset, 4);
  }

  #[test]
  fn index() {
    let shape = Shape::new(&[2,3]);
    assert_eq!(shape.index(&[0]), 0);
    assert_eq!(shape.index(&[1,0]), 3);
  }

  #[test]
  fn index_negative_stride() {
    let shape = Shape::offset(&[3,2,2], &[4,-2,1], 2);
    assert_eq!(shape.index(&[0,0,0]), 2);
    assert_eq!(shape.index(&[0,0,1]), 3);
    assert_eq!(shape.index(&[0,1,0]), 0);
    assert_eq!(shape.index(&[0,1,1]), 1);
    assert_eq!(shape.index(&[1,0,0]), 6);
    assert_eq!(shape.index(&[1,1,0]), 4);
  }

  #[test]
  fn iterate() {
    let shape = Shape::offset(&[3,2,2], &[4,-2,1], 2);
    let indices: Vec<_> = shape.iter().collect();
    assert_eq!(indices, vec![2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]);
  }

  #[test]
  fn step() {
    let shape = Shape::new(&[3,2,2]).step(&[1,-1,1]);
    assert_eq!(shape.strides, vec![4,-2,1]);
    assert_eq!(shape.offset, 2);
    assert_eq!(shape.step(&[1, 1,1]).offset, 2);
    assert_eq!(shape.step(&[1,-1,1]).offset, 0);
  }

  #[test]
  fn range() {
    let shape = Shape::new(&[4,4,4]).range(&[1..3, 1..3, 1..3]);
    assert_eq!(shape.dims, vec![2,2,2]);
    assert_eq!(shape.offset, 21);
    let indices: Vec<_> = shape.iter().collect();
    assert_eq!(indices, vec![21, 22, 25, 26, 37, 38, 41, 42]);
  }

  #[test]
  fn unsqueeze() {
    let shape = Shape::new(&[3,2,2]).unsqueeze(-1);
    assert_eq!(shape.dims, vec![3,2,2,1]);
    assert_eq!(shape.strides, vec![4,2,1,1]);

    let shape = Shape::new(&[2,3,2]).unsqueeze(-3);
    assert_eq!(shape.dims, vec![2,1,3,2]);
    assert_eq!(shape.strides, vec![6,6,2,1]);

    let shape = Shape::new(&[2,3,2]).unsqueeze(0);
    assert_eq!(shape.dims, vec![1,2,3,2]);
    assert_eq!(shape.strides, vec![12,6,2,1]);
  }

  #[test]
  fn squeeze() {
    let shape = Shape::new(&[3,2,1]).squeeze();
    assert_eq!(shape.dims, vec![3,2]);
    assert_eq!(shape.strides, vec![2,1]);

    let shape = Shape::new(&[1,2,3,2]).squeeze();
    assert_eq!(shape.dims, vec![2,3,2]);
    assert_eq!(shape.strides, vec![6,2,1]);

    let shape = Shape::new(&[2,1,3,1,2]).squeeze_only(-2);
    assert_eq!(shape.dims, vec![2,1,3,2]);
    assert_eq!(shape.strides, vec![6,6,2,1]);
  }

  #[test]
  fn extend() {
    let shape = Shape::new(&[2,3]).extend(4);
    assert_eq!(shape.dims, vec![1,1,2,3]);
    assert_eq!(shape.strides, vec![6,6,3,1]);
  }

  #[test]
  fn broadcast() {
    let shape = Shape::new(&[2,3,2]).broadcast(&Shape::new(&[2,1,2]));
    assert_eq!(shape.dims, vec![2,3,2]);
    assert_eq!(shape.strides, vec![6,2,1]);

    let shape = Shape::new(&[2,1,2]).broadcast(&Shape::new(&[2,3,1]));
    assert_eq!(shape.dims, vec![2,3,2]);
    assert_eq!(shape.strides, vec![2,0,1]);

    let indices: Vec<_> = shape.iter().collect();
    assert_eq!(indices, vec![0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3]);
  }

  #[test]
  fn transpose() {
    let shape = Shape::new(&[2,3]).transpose(0,1);
    assert_eq!(shape.dims, vec![3,2]);
    assert_eq!(shape.strides, vec![1,3]);
    assert_eq!(shape.index(&[1,0]), 1);
    assert_eq!(shape.index(&[1,1]), 4);
  }
}
