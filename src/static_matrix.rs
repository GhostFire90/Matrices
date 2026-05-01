use crate::{Quaternion, Vec3, Vector};
use std::{
  iter::Sum,
  ops::{Add, Mul},
};

#[derive(PartialEq, Debug)]
pub struct Matrix<const R: usize, const C: usize, T>
where
  T: Clone + PartialEq,
  [T; R * C]:,
{
  data: [T; R * C],
}

pub type Mat4 = Matrix<4, 4, f32>;

impl<const R: usize, const C: usize, T> Matrix<R, C, T>
where
  T: Clone + PartialEq,
  [T; R * C]:,
{
  pub fn new(data: &[[T; C]; R]) -> Self
  {
    let flattened: Vec<T> = data.iter().flatten().cloned().collect();

    Self {
      data: flattened
        .try_into()
        .map_err(|_| "SHOULD BE IMPOSSIBLE")
        .unwrap(),
    }
  }
  pub fn transpose(&self) -> Matrix<C, R, T>
  where
    [T; C * R]:,
  {
    let mut dvec = Vec::new();
    for c in 0..C
    {
      dvec.append(&mut self.col(c).into_iter().cloned().collect());
    }
    Matrix::<C, R, T> {
      data: dvec
        .try_into()
        .map_err(|_| "SHOULD NOT BE POSSIBLE")
        .unwrap(),
    }
  }
  pub fn col(&self, col: usize) -> Vec<&T>
  {
    assert!(col < C);

    self.data.iter().skip(col).step_by(C).collect()
  }
  pub fn row(&self, row: usize) -> Vec<&T>
  {
    assert!(row < R);
    self.data.iter().skip(C * row).take(R).collect()
  }
}

impl<const D: usize> Matrix<D, D, f32>
where
  [f32; D * D]:,
{
  pub fn identity() -> Self
  {
    let mut dvec = Vec::new();
    for i in 0..D
    {
      for j in 0..D
      {
        if i == j
        {
          dvec.push(1.0);
        }
        else
        {
          dvec.push(0.0);
        }
      }
    }
    Self {
      data: dvec.try_into().map_err(|_| "SHOULD BE IMPOSSIBLE").unwrap(),
    }
  }
}

impl Mat4
{
  pub fn translation(pos: Vec3) -> Self
  {
    Matrix::new(&[
      [1.0, 0.0, 0.0, pos.x()],
      [0.0, 1.0, 0.0, pos.y()],
      [0.0, 0.0, 1.0, pos.z()],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn rotation(axis: Vec3, theta: f32) -> Self
  {
    let axis = axis.matrix();
    let dvec: Vec<f32> = Quaternion::rotation(theta as f64, axis)
      .unwrap()
      .rotation_matrix()
      .expand()
      .flatten()
      .into_iter()
      .map(|x| x as f32)
      .collect();
    Self {
      data: dvec.try_into().map_err(|_| "SHOULD BE IMPOSSIBLE").unwrap(),
    }
  }
  pub fn scale(scale: Vec3) -> Self
  {
    Self::new(&[
      [scale.x(), 0.0, 0.0, 0.0],
      [0.0, scale.y(), 0.0, 0.0],
      [0.0, 0.0, scale.z(), 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn perspective(fov: f32, near: f32, far: f32, aspect: f32) -> Self
  {
    Self::new(&[
      [1.0 / (aspect * f32::tan(fov / 2.0)), 0.0, 0.0, 0.0],
      [0.0, 1.0 / (aspect * f32::tan(fov / 2.0)), 0.0, 0.0],
      [
        0.0,
        0.0,
        -(far + near) / (far - near),
        -(2.0 * far * near) / (far - near),
      ],
      [0.0, 0.0, -1.0, 0.0],
    ])
  }
}

impl<const R: usize> Into<Matrix<R, 1, f32>> for Vector<{ R * 1 }>
where
  [f32; R * 1]:,
{
  fn into(self) -> Matrix<R, 1, f32>
  {
    Matrix {
      data: self.components,
    }
  }
}
impl<const R: usize> Into<Vector<{ R * 1 }>> for Matrix<R, 1, f32>
{
  fn into(self) -> Vector<{ R * 1 }>
  {
    Vector {
      components: self.data,
    }
  }
}

impl<const R: usize, const C: usize> Mul<Vector<{ C * 1 }>> for Matrix<R, C, f32>
where
  [f32; R * 1]:,
  [f32; R * C]:,
  [f32; C * 1]:,
{
  type Output
    = Matrix<R, 1, f32>
  where
    [f32; R * 1]:;

  fn mul(self, rhs: Vector<{ C * 1 }>) -> Self::Output
  {
    self * Matrix::from(rhs.into())
  }
}

impl<const R1: usize, const CR: usize, const C2: usize, T> Mul<Matrix<CR, C2, T>>
  for Matrix<R1, CR, T>
where
  T: Clone + PartialEq + Mul<T, Output = T> + Add + Sum,
  [T; R1 * CR]:,
  [T; R1 * C2]:,
  [T; CR * C2]:,
{
  type Output = Matrix<R1, C2, T>;

  fn mul(self, rhs: Matrix<CR, C2, T>) -> Self::Output
  {
    let mut data = Vec::new();
    for row in 0..R1
    {
      for col in 0..C2
      {
        let value = self
          .row(row)
          .iter()
          .zip(rhs.col(col).iter())
          .map(|(x, y)| (*x).clone() * (*y).clone())
          .sum::<T>();
        data.push(value)
      }
    }
    Self::Output {
      data: data.try_into().map_err(|_| "SHOULDNT BE POSSIBLE").unwrap(),
    }
  }
}

#[cfg(test)]
mod tests
{
  use crate::static_matrix::Mat4;

  #[test]
  fn mul()
  {
    let a = Mat4::new(&[
      [21.0, 16.0, 61.0, 80.0],
      [69.0, 68.0, 76.0, 26.0],
      [60.0, 85.0, 82.0, 94.0],
      [94.0, 87.0, 51.0, 49.0],
    ]);
    let b = Mat4::new(&[
      [79.0, 49.0, 20.0, 64.0],
      [28.0, 7.0, 39.0, 95.0],
      [55.0, 35.0, 63.0, 92.0],
      [51.0, 50.0, 82.0, 90.0],
    ]);
    let expected = Mat4::new(&[
      [9542.0, 7276.0, 11447.0, 15676.0],
      [12861.0, 7817.0, 10952.0, 20208.0],
      [16424.0, 11105.0, 17389.0, 27919.0],
      [15166.0, 9450.0, 12504.0, 23383.0],
    ]);

    assert_eq!(a * b, expected);
  }
}
