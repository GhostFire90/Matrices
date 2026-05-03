use std::ops::Mul;

use crate::MatrixDynamic;

#[derive(Clone)]
pub struct Vector<const S: usize> {
  pub(crate) components: [f32; S],
}
impl<const S: usize> Into<MatrixDynamic> for Vector<S> {
  fn into(self) -> MatrixDynamic {
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

impl<const S: usize> Vector<S> {
  pub fn matrix(&self) -> MatrixDynamic {
    MatrixDynamic::from(self.clone().into())
  }
  pub fn dot(&self, rhs: &Self) -> f32 {
    (rhs.matrix() * self.matrix().transpose())
      .unwrap()
      .get(0, 0)
      .unwrap()
      .clone() as f32
  }
  pub fn magnitude(&self) -> f32 {
    self.dot(self)
  }
}
impl<const S: usize> Mul<f32> for Vector<S> {
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output {
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
impl<const S: usize> Mul<Vector<S>> for f32 {
  type Output = Vector<S>;

  fn mul(self, rhs: Vector<S>) -> Self::Output {
    rhs * self
  }
}

pub type Vec4 = Vector<4>;
pub type Vec3 = Vector<3>;
pub type Vec2 = Vector<2>;

impl Vec4 {
  pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
    Self {
      components: [x, y, z, w],
    }
  }
  pub fn x(&self) -> f32 {
    self.components[0]
  }
  pub fn y(&self) -> f32 {
    self.components[1]
  }
  pub fn z(&self) -> f32 {
    self.components[2]
  }
  pub fn w(&self) -> f32 {
    self.components[3]
  }
}

impl Vec3 {
  pub fn new(x: f32, y: f32, z: f32) -> Self {
    Self {
      components: [x, y, z],
    }
  }
  pub fn x(&self) -> f32 {
    self.components[0]
  }
  pub fn y(&self) -> f32 {
    self.components[1]
  }
  pub fn z(&self) -> f32 {
    self.components[2]
  }
}

impl Vec2 {
  pub fn new(x: f32, y: f32) -> Self {
    Self { components: [x, y] }
  }
  pub fn x(&self) -> f32 {
    self.components[0]
  }
  pub fn y(&self) -> f32 {
    self.components[1]
  }
}
