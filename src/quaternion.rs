use crate::{Mat4, MatrixDynamic, MatrixError, Vec3};
use std::ops::{Div, Mul};

#[derive(Debug)]
pub enum QuaternionErr {
  MatrixErr(MatrixError),
  InvalidMatrix(MatrixDynamic),
  InvalidVector(MatrixDynamic),
}
impl From<MatrixError> for QuaternionErr {
  fn from(value: MatrixError) -> Self {
    Self::MatrixErr(value)
  }
}

// Q = a + bi + cj + dk
#[derive(Clone, Debug, PartialEq)]
pub struct Quaternion {
  components: [f32; 4],
}

impl Quaternion {
  pub const fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
    Self {
      components: [a, b, c, d],
    }
  }

  pub fn rotation(angle: f32, axis: Vec3) -> Self {
    let a = (angle / 2.0).cos();
    let components: Vec<f32> = (axis * ((angle / 2.0).sin()))
      .components
      .into_iter()
      .collect();
    Self {
      components: [a, components[0], components[1], components[2]],
    }
  }

  pub fn rotate(&self, vector: Vec3) -> Vec3 {
    (self.clone() * Quaternion::from(vector.into()) * self.conjugate()).into()
  }

  pub fn rotation_matrix(&self) -> Mat4 {
    let (w, x, y, z) = (self.a(), self.b(), self.c(), self.d());
    Mat4::new([
      [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        0.0,
      ],
      [
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        0.0,
      ],
      [
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
      ],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn euler_angles(angles: Vec3) -> Self {
    Self::rotation(angles.x() as f32, Vec3::new(1.0, 0.0, 0.0))
      * Self::rotation(angles.y() as f32, Vec3::new(0.0, 1.0, 0.0))
      * Self::rotation(angles.z() as f32, Vec3::new(0.0, 0.0, 1.0))
  }

  pub fn conjugate(&self) -> Self {
    Self {
      components: [self.a(), -self.b(), -self.c(), -self.d()],
    }
  }

  pub fn matrix(&self) -> Mat4 {
    self.clone().into()
  }

  pub fn magnitude(&self) -> f32 {
    (self.clone() * self.conjugate()).a().sqrt()
  }
  pub fn magnitude_square(&self) -> f32 {
    (self.clone() * self.conjugate()).a()
  }

  pub fn inverse(&self) -> Self {
    self.conjugate() / self.magnitude_square()
  }

  pub fn a(&self) -> f32 {
    self.components[0]
  }

  pub fn b(&self) -> f32 {
    self.components[1]
  }

  pub fn c(&self) -> f32 {
    self.components[2]
  }

  pub fn d(&self) -> f32 {
    self.components[3]
  }
}

impl Mul for Quaternion {
  type Output = Quaternion;

  fn mul(self, rhs: Self) -> Self::Output {
    (self.matrix() * rhs.matrix()).into()
  }
}

impl Div for Quaternion {
  type Output = Self;

  fn div(self, rhs: Self) -> Self::Output {
    self * rhs.inverse()
  }
}

impl Div<f32> for Quaternion {
  type Output = Quaternion;

  fn div(self, rhs: f32) -> Self::Output {
    Self {
      components: [
        self.a() / rhs,
        self.b() / rhs,
        self.c() / rhs,
        self.d() / rhs,
      ],
    }
  }
}

impl Mul<f32> for Quaternion {
  type Output = Quaternion;

  fn mul(self, rhs: f32) -> Self::Output {
    Self {
      components: [
        self.a() * rhs,
        self.b() * rhs,
        self.c() * rhs,
        self.d() * rhs,
      ],
    }
  }
}

impl Into<Mat4> for Quaternion {
  fn into(self) -> Mat4 {
    let (a, b, c, d) = (self.a(), self.b(), self.c(), self.d());
    Mat4::new([[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]])
  }
}

impl Into<Quaternion> for Mat4 {
  fn into(self) -> Quaternion {
    Quaternion {
      components: self
        .col(0)
        .into_iter()
        .cloned()
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap(),
    }
  }
}

impl Into<Quaternion> for Vec3 {
  fn into(self) -> Quaternion {
    Quaternion {
      components: [0.0, self.x(), self.y(), self.z()],
    }
  }
}
impl Into<Vec3> for Quaternion {
  fn into(self) -> Vec3 {
    Vec3 {
      components: [self.b(), self.c(), self.d()],
    }
  }
}

#[cfg(test)]
mod tests {
  use core::f32;

  use crate::{Quaternion, Vec3};

  #[test]
  pub fn multest() {
    const EXPECTED: Quaternion = Quaternion::new(-60.0, 12.0, 30.0, 24.0);
    let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let b = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    assert!((a * b) == EXPECTED)
  }

  #[test]
  pub fn conjtest() {
    const EXPECTED: Quaternion = Quaternion::new(1.0, -2.0, -3.0, -4.0);
    assert!(Quaternion::new(1.0, 2.0, 3.0, 4.0).conjugate() == EXPECTED);
  }

  #[test]
  pub fn invtest() {
    const EXPECTED: Quaternion = Quaternion::new(1.0 / 30.0, -1.0 / 15.0, -1.0 / 10.0, -2.0 / 15.0);
    let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert!(a.inverse() == EXPECTED);
  }

  #[test]
  pub fn rotatetest() {
    let expected: Quaternion = Quaternion::new(
      (f32::consts::FRAC_PI_4).cos(),
      (f32::consts::FRAC_PI_4).sin(),
      (f32::consts::FRAC_PI_4).sin(),
      (f32::consts::FRAC_PI_4).sin(),
    );
    let a = Quaternion::rotation(f32::consts::FRAC_PI_2, Vec3::new(1.0, 1.0, 1.0));
    dbg!(&a);
    dbg!(&expected);

    assert!(a == expected)
  }
}
