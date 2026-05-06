use crate::{MatrixDynamic, MatrixError, Quaternion, Vec3, Vector};
use std::{
  iter::Sum,
  ops::{Add, Mul},
};

/// # Matrix
/// A RxC matrix of type T
/// Row major
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize, T>
where
  T: Clone + PartialEq,
{
  data: [[T; C]; R],
}

/// Generic f32 mat4 type
pub type Mat4 = Matrix<4, 4, f32>;

impl<const R: usize, const C: usize, T> Matrix<R, C, T>
where
  T: Clone + PartialEq,
{
  pub fn new(data: [[T; C]; R]) -> Self
  {
    Self { data }
  }
  /// [Transpose](https://en.wikipedia.org/wiki/Transpose)
  pub fn transpose(&self) -> Matrix<C, R, T>
  {
    let mut ret = Vec::new();
    let mut col = 0;
    ret.resize_with(C, || {
      col += 1;
      self
        .col(col - 1)
        .into_iter()
        .cloned()
        .collect::<Vec<T>>()
        .try_into()
        .map_err(|_| "SHOULD BE IMPOSSIBLE")
        .unwrap()
    });
    Matrix {
      data: ret.try_into().map_err(|_| "SHOULD BE IMPOSSIBLE").unwrap(),
    }
  }
  /// Gets a column
  pub fn col(&self, col: usize) -> Vec<&T>
  {
    assert!(col < C);

    self.data.iter().map(|c| &c[col]).collect()
  }
  /// Gets a row
  pub fn row(&self, row: usize) -> Vec<&T>
  {
    assert!(row < R);
    self.data[row].iter().collect()
  }
}

impl<const D: usize> Matrix<D, D, f32>
{
  /// Creates an identity matrix
  pub fn identity() -> Self
  {
    let mut dvec = Vec::new();

    for i in 0..D
    {
      let mut row = Vec::new();
      let mut j = 0;
      row.resize_with(D, || {
        let ret = if j == i { 1.0 } else { 0.0 };
        j += 1;
        ret
      });
      dvec.push(row.try_into().map_err(|_| "SHOULD BE IMPOSSIBLE").unwrap());
    }
    Self {
      data: dvec.try_into().map_err(|_| "SHOULD BE IMPOSSIBLE").unwrap(),
    }
  }

  pub fn det(&self) -> f32
  {
    MatrixDynamic::from(self.clone().into()).det().unwrap() as f32
  }
  pub fn inverse(&self) -> Self
  {
    MatrixDynamic::from(self.clone().into())
      .inverse()
      .unwrap()
      .try_into()
      .unwrap()
  }
}

impl Mat4
{
  /// Affine translation matrix
  pub fn translation(pos: Vec3) -> Self
  {
    Matrix::new([
      [1.0, 0.0, 0.0, pos.x()],
      [0.0, 1.0, 0.0, pos.y()],
      [0.0, 0.0, 1.0, pos.z()],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  /// Affine rotation matrix around axis by theta degrees
  pub fn rotation(axis: Vec3, theta: f32) -> Self
  {
    Quaternion::rotation(theta, axis).rotation_matrix()
  }

  /// Affine scale matrix with component wise scale
  pub fn scale(scale: Vec3) -> Self
  {
    Matrix::new([
      [scale.x(), 0.0, 0.0, 0.0],
      [0.0, scale.y(), 0.0, 0.0],
      [0.0, 0.0, scale.z(), 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  /// [perspective projection](https://en.wikipedia.org/wiki/3D_projection)
  pub fn perspective(fov: f32, near: f32, far: f32, aspect: f32) -> Self
  {
    Matrix::new([
      [1.0 / (aspect * f32::tan(fov / 2.0)), 0.0, 0.0, 0.0],
      [0.0, 1.0 / (f32::tan(fov / 2.0)), 0.0, 0.0],
      [
        0.0,
        0.0,
        -(far) / (far - near),
        -(far * near) / (far - near),
      ],
      [0.0, 0.0, -1.0, 0.0],
    ])
  }

  pub fn euler_angles(angles: Vec3) -> Self
  {
    Quaternion::euler_angles(angles).rotation_matrix()
  }
}

impl<const R: usize> Into<Matrix<R, 1, f32>> for Vector<R>
{
  fn into(self) -> Matrix<R, 1, f32>
  {
    let mut dvec = Vec::new();
    for i in 0..R
    {
      dvec.push([self.components[i]]);
    }
    Matrix {
      data: dvec.try_into().map_err(|_| "SHOULDNT BE POSSIBLE").unwrap(),
    }
  }
}
impl<const R: usize> Into<Vector<R>> for Matrix<R, 1, f32>
{
  fn into(self) -> Vector<R>
  {
    Vector {
      components: self
        .data
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>()
        .try_into()
        .map_err(|_| "SHOULDNT BE POSSIBLE")
        .unwrap(),
    }
  }
}

impl<const R: usize, const C: usize> TryInto<Matrix<R, C, f32>> for MatrixDynamic
{
  type Error = MatrixError;

  fn try_into(self) -> Result<Matrix<R, C, f32>, Self::Error>
  {
    if self.rows() != R || self.cols() != C
    {
      return Err(MatrixError::MissmatchDimension {
        lhs: (self.rows(), self.cols()),
        rhs: (R, C),
      });
    }

    Ok(Matrix {
      data: self
        .data()
        .into_iter()
        .map(|x| {
          x.into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>()
            .try_into()
            .map_err(|_| "")
            .unwrap()
        })
        .collect::<Vec<[f32; C]>>()
        .try_into()
        .map_err(|_| "")
        .unwrap(),
    })
  }
}

impl<const R: usize, const C: usize> Mul<Vector<C>> for Matrix<R, C, f32>
{
  type Output = Matrix<R, 1, f32>;
  fn mul(self, rhs: Vector<C>) -> Self::Output
  {
    self * Matrix::from(rhs.into())
  }
}

impl<const R1: usize, const CR: usize, const C2: usize, T> Mul<Matrix<CR, C2, T>>
  for Matrix<R1, CR, T>
where
  T: Clone + PartialEq + Mul<T, Output = T> + Add + Sum,
{
  type Output = Matrix<R1, C2, T>;

  fn mul(self, rhs: Matrix<CR, C2, T>) -> Self::Output
  {
    let mut data = Vec::new();
    for row in 0..R1
    {
      let mut col = 0;
      let mut ret = Vec::new();
      ret.resize_with(C2, || {
        col += 1;
        self
          .row(row)
          .iter()
          .zip(rhs.col(col - 1).iter())
          .map(|(x, y)| (*x).clone() * (*y).clone())
          .sum::<T>()
      });
      data.push(ret.try_into().map_err(|_| "SHOULDNT BE POSSIBLE").unwrap())
    }
    Self::Output {
      data: data.try_into().map_err(|_| "SHOULDNT BE POSSIBLE").unwrap(),
    }
  }
}

impl<const D: usize> Default for Matrix<D, D, f32>
{
  fn default() -> Self
  {
    Self::identity()
  }
}

impl<const R: usize, const C: usize> Mul<f32> for Matrix<R, C, f32>
{
  type Output = Self;

  fn mul(mut self, rhs: f32) -> Self::Output
  {
    self
      .data
      .iter_mut()
      .for_each(|v| v.iter_mut().for_each(|x| *x *= rhs));
    self
  }
}
impl<const R: usize, const C: usize> Mul<Matrix<R, C, f32>> for f32
{
  type Output = Matrix<R, C, f32>;

  fn mul(self, rhs: Matrix<R, C, f32>) -> Self::Output
  {
    rhs * self
  }
}

impl<const R: usize, const C: usize> Into<MatrixDynamic> for Matrix<R, C, f32>
{
  fn into(self) -> MatrixDynamic
  {
    MatrixDynamic::new(
      self
        .data
        .into_iter()
        .map(|x| x.into_iter().map(|x| x as f64).collect())
        .collect(),
    )
    .unwrap()
  }
}

#[cfg(test)]
mod tests
{
  use crate::static_matrix::Mat4;

  #[test]
  fn mul()
  {
    let a = Mat4::new([
      [21.0, 16.0, 61.0, 80.0],
      [69.0, 68.0, 76.0, 26.0],
      [60.0, 85.0, 82.0, 94.0],
      [94.0, 87.0, 51.0, 49.0],
    ]);
    let b = Mat4::new([
      [79.0, 49.0, 20.0, 64.0],
      [28.0, 7.0, 39.0, 95.0],
      [55.0, 35.0, 63.0, 92.0],
      [51.0, 50.0, 82.0, 90.0],
    ]);
    let expected = Mat4::new([
      [9542.0, 7276.0, 11447.0, 15676.0],
      [12861.0, 7817.0, 10952.0, 20208.0],
      [16424.0, 11105.0, 17389.0, 27919.0],
      [15166.0, 9450.0, 12504.0, 23383.0],
    ]);

    assert_eq!(a * b, expected);
  }
}
