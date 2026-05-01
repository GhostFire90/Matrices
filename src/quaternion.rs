use crate::{MatrixDynamic, MatrixError};
use std::ops::{Div, Mul};

#[derive(Debug)]
pub enum QuaternionErr
{
  MatrixErr(MatrixError),
  InvalidMatrix(MatrixDynamic),
  InvalidVector(MatrixDynamic),
}
impl From<MatrixError> for QuaternionErr
{
  fn from(value: MatrixError) -> Self
  {
    Self::MatrixErr(value)
  }
}

// Q = a + bi + cj + dk
#[derive(Clone, Debug, PartialEq)]
pub struct Quaternion
{
  components: [f64; 4],
}

impl Quaternion
{
  pub const fn new(a: f64, b: f64, c: f64, d: f64) -> Self
  {
    Self {
      components: [a, b, c, d],
    }
  }

  pub fn rotation(angle: f64, axis: MatrixDynamic) -> Result<Self, QuaternionErr>
  {
    if axis.rows() > 3 || axis.cols() != 1
    {
      return Err(QuaternionErr::InvalidVector(axis));
    }

    let a = (angle / 2.0).cos();
    let components: Vec<f64> = (axis * ((angle / 2.0).sin()))
      .col(0)
      .unwrap()
      .into_iter()
      .cloned()
      .collect();
    Ok(Self {
      components: [
        a,
        components.get(0).cloned().unwrap_or_default(),
        components.get(1).cloned().unwrap_or_default(),
        components.get(2).cloned().unwrap_or_default(),
      ],
    })
  }

  pub fn rotate(&self, vector: MatrixDynamic) -> Result<MatrixDynamic, QuaternionErr>
  {
    if vector.rows() != 3 || vector.cols() != 1
    {
      return Err(QuaternionErr::InvalidVector(vector));
    }
    let col: Vec<f64> = vector.col(0)?.into_iter().cloned().collect();
    let qvec = Self {
      components: [0.0, col[0], col[1], col[2]],
    };
    let res = self.clone() * qvec * self.conjugate();
    Ok(MatrixDynamic::new(
      res.components[1..]
        .to_vec()
        .into_iter()
        .map(|x| vec![x])
        .collect(),
    )?)
  }
  pub fn rotation_matrix(&self) -> MatrixDynamic
  {
    let (a, b, c, d) = (self.a(), self.b(), self.c(), self.d());

    MatrixDynamic::new(vec![
      vec![
        1.0 - 2.0 * (c * c + d * d),
        2.0 * (b * c - a * d),
        2.0 * (b * d + a * c),
      ],
      vec![
        2.0 * (b * c + a * d),
        1.0 - 2.0 * (b * b + d * d),
        2.0 * (c * d - a * c),
      ],
      vec![
        2.0 * (b * d - a * c),
        2.0 * (c * d + a * d),
        1.0 - 2.0 * (b * b + c * c),
      ],
    ])
    .unwrap()
    .transpose()
  }

  pub fn conjugate(&self) -> Self
  {
    Self {
      components: [self.a(), -self.b(), -self.c(), -self.d()],
    }
  }

  pub fn matrix(&self) -> MatrixDynamic
  {
    self.clone().into()
  }

  pub fn magnitude(&self) -> f64
  {
    (self.clone() * self.conjugate()).a().sqrt()
  }
  pub fn magnitude_square(&self) -> f64
  {
    (self.clone() * self.conjugate()).a()
  }

  pub fn inverse(&self) -> Self
  {
    self.conjugate() / self.magnitude_square()
  }

  pub fn a(&self) -> f64
  {
    self.components[0]
  }

  pub fn b(&self) -> f64
  {
    self.components[1]
  }

  pub fn c(&self) -> f64
  {
    self.components[2]
  }

  pub fn d(&self) -> f64
  {
    self.components[3]
  }
}

impl Mul for Quaternion
{
  type Output = Quaternion;

  fn mul(self, rhs: Self) -> Self::Output
  {
    (self.matrix() * rhs.matrix()).unwrap().try_into().unwrap()
  }
}

impl Div for Quaternion
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self::Output
  {
    self * rhs.inverse()
  }
}

impl Div<f64> for Quaternion
{
  type Output = Quaternion;

  fn div(self, rhs: f64) -> Self::Output
  {
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

impl Mul<f64> for Quaternion
{
  type Output = Quaternion;

  fn mul(self, rhs: f64) -> Self::Output
  {
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

impl Into<MatrixDynamic> for Quaternion
{
  fn into(self) -> MatrixDynamic
  {
    let (a, b, c, d) = (self.a(), self.b(), self.c(), self.d());
    MatrixDynamic::new(vec![
      vec![a, -b, -c, -d],
      vec![b, a, -d, c],
      vec![c, d, a, -b],
      vec![d, -c, b, a],
    ])
    .unwrap()
  }
}

impl TryFrom<MatrixDynamic> for Quaternion
{
  type Error = QuaternionErr;

  fn try_from(value: MatrixDynamic) -> Result<Self, Self::Error>
  {
    if value.cols() != value.rows() || value.rows() != 4
    {
      return Err(QuaternionErr::InvalidMatrix(value));
    }
    let col: Vec<f64> = value.col(0)?.iter().cloned().cloned().collect();
    Ok(Self {
      components: [col[0], col[1], col[2], col[3]],
    })
  }
}

#[cfg(test)]
mod tests
{
  use crate::{MatrixDynamic, Quaternion};

  #[test]
  pub fn multest()
  {
    const EXPECTED: Quaternion = Quaternion::new(-60.0, 12.0, 30.0, 24.0);
    let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let b = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    assert!((a * b) == EXPECTED)
  }

  #[test]
  pub fn conjtest()
  {
    const EXPECTED: Quaternion = Quaternion::new(1.0, -2.0, -3.0, -4.0);
    assert!(Quaternion::new(1.0, 2.0, 3.0, 4.0).conjugate() == EXPECTED);
  }

  #[test]
  pub fn invtest()
  {
    const EXPECTED: Quaternion = Quaternion::new(1.0 / 30.0, -1.0 / 15.0, -1.0 / 10.0, -2.0 / 15.0);
    let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert!(a.inverse() == EXPECTED);
  }

  #[test]
  pub fn rotatetest()
  {
    let expected: Quaternion = Quaternion::new(
      0.7073882691672,
      0.235608393701789 * (3.0_f64.sqrt()),
      0.235608393701789 * (3.0_f64.sqrt()),
      0.235608393701789 * (3.0_f64.sqrt()),
    );
    let a = Quaternion::rotation(
      3.14 / 2.0,
      MatrixDynamic::new(vec![vec![1.0, 1.0, 1.0]])
        .unwrap()
        .transpose(),
    )
    .unwrap();
    dbg!(&a);
    dbg!(&expected);

    assert!(a == expected)
  }
}
