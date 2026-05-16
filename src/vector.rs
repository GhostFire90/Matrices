use crate::{Matrix, MatrixDynamic};
use std::ops::{Div, Mul, Sub};

#[derive(Clone)]
pub struct Vector<const S: usize>
{
  pub(crate) components: [f32; S],
}
impl<const S: usize> Into<MatrixDynamic> for Vector<S>
{
  fn into(self) -> MatrixDynamic
  {
    MatrixDynamic::new(vec![
      self
        .components
        .to_vec()
        .into_iter()
        .map(|x| x as f64)
        .collect(),
    ])
    .unwrap()
    .transpose()
  }
}

impl<const S: usize> Vector<S>
{
  pub fn matrix(&self) -> Matrix<S, 1, f32>
  {
    Matrix {
      data: [self.components],
    }
    .transpose()
  }
  pub fn dot(&self, rhs: &Self) -> f32
  {
    (self.matrix().transpose() * rhs.matrix()).data[0][0]
  }
  pub fn magnitude_squared(&self) -> f32
  {
    self.dot(self)
  }
  pub fn magnitude(&self) -> f32
  {
    self.magnitude_squared().sqrt()
  }
  pub fn normalize(&self) -> Self
  {
    self.clone() / self.magnitude()
  }
  pub fn distance_squared(&self, rhs: &Self) -> f32
  {
    (rhs.clone() - self.clone()).magnitude_squared()
  }
  pub fn distance(&self, rhs: &Self) -> f32
  {
    self.distance_squared(rhs).sqrt()
  }

  pub fn angle(&self, rhs: &Self) -> f32
  {
    self.dot(rhs) / (self.magnitude() * rhs.magnitude())
  }
}
impl<const S: usize> Mul<f32> for Vector<S>
{
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output
  {
    Self {
      components: self
        .components
        .into_iter()
        .map(|x| x * rhs)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap(),
    }
  }
}
impl<const S: usize> Div<f32> for Vector<S>
{
  type Output = Self;

  fn div(self, rhs: f32) -> Self::Output
  {
    Self {
      components: self
        .components
        .into_iter()
        .map(|x| x / rhs)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap(),
    }
  }
}

impl<const S: usize> Mul<Vector<S>> for f32
{
  type Output = Vector<S>;

  fn mul(self, rhs: Vector<S>) -> Self::Output
  {
    rhs * self
  }
}

impl<const R: usize> From<Matrix<R, 1, f32>> for Vector<R>
{
  fn from(value: Matrix<R, 1, f32>) -> Self
  {
    Self {
      components: value
        .data
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .try_into()
        .unwrap(),
    }
  }
}

impl<const S: usize> Sub for Vector<S>
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output
  {
    Self {
      components: self
        .components
        .into_iter()
        .zip(rhs.components.into_iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap(),
    }
  }
}

pub type Vec4 = Vector<4>;
pub type Vec3 = Vector<3>;
pub type Vec2 = Vector<2>;

impl Vec4
{
  pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self
  {
    Self {
      components: [x, y, z, w],
    }
  }
  pub fn x(&self) -> f32
  {
    self.components[0]
  }
  pub fn y(&self) -> f32
  {
    self.components[1]
  }
  pub fn z(&self) -> f32
  {
    self.components[2]
  }
  pub fn w(&self) -> f32
  {
    self.components[3]
  }
}

impl Vec3
{
  pub const fn new(x: f32, y: f32, z: f32) -> Self
  {
    Self {
      components: [x, y, z],
    }
  }
  pub fn x(&self) -> f32
  {
    self.components[0]
  }
  pub fn y(&self) -> f32
  {
    self.components[1]
  }
  pub fn z(&self) -> f32
  {
    self.components[2]
  }

  pub fn cross(&self, rhs: &Self) -> Self
  {
    Self {
      components: [
        self.y() * rhs.z() - rhs.y() * self.z(),
        self.z() * rhs.x() - rhs.z() * self.x(),
        self.x() * rhs.y() - rhs.x() * self.y(),
      ],
    }
  }
}

impl Vec2
{
  pub fn new(x: f32, y: f32) -> Self
  {
    Self { components: [x, y] }
  }
  pub fn x(&self) -> f32
  {
    self.components[0]
  }
  pub fn y(&self) -> f32
  {
    self.components[1]
  }
}
