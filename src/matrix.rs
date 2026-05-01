use crate::{Quaternion, Vec3, Vector};
use std::{
  fmt::Display,
  ops::{Add, Mul, Neg, Sub},
};

/// Error type for various Matrix operations
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixError
{
  /// Incompatible dimensions for mul
  MissmatchDimension
  {
    lhs: (usize, usize),
    rhs: (usize, usize),
  },
  /// 2d vec given does not have uniform lengths
  NonUniformColumnLength,
  /// index out of bounds for row or col
  OutOfBoundsRC(usize),
  /// index out of bounds for get
  OutOfBounds(usize, usize),
  /// The matrix is required to be a square and isnt
  NonSquareMatrix,
}

/// # MatrixReal
/// a matrix type using only real numbers represented by f64<br>
/// it is represented in a row-col 2d matrix for ease of use
#[derive(Clone, PartialEq, Debug)]
pub struct MatrixDynamic
{
  row_count: usize,
  col_count: usize,
  data: Vec<Vec<f64>>,
}

pub type Result<T> = std::result::Result<T, MatrixError>;

impl MatrixDynamic
{
  pub fn new(data: Vec<Vec<f64>>) -> Result<Self>
  {
    // data[row][col]
    if data.len() == 0
    {
      return Ok(Self {
        row_count: 0,
        col_count: 0,
        data,
      });
    }

    let col_len = data[0].len();
    if data.iter().any(|x| x.len() != col_len)
    {
      return Err(MatrixError::NonUniformColumnLength);
    }

    Ok(Self {
      row_count: data.len(),
      col_count: col_len,
      data,
    })
  }

  /// Creates a square identity matrix of dim x dim dimension
  pub fn identity(dim: usize) -> Self
  {
    let mut data = Vec::new();
    let mut index = 0;
    data.resize_with(dim, || {
      let mut res = Vec::new();
      let mut current = 0;
      res.resize_with(dim, || {
        let old = current;
        current += 1;
        if old == index
        {
          return 1.0;
        }
        else
        {
          return 0.0;
        }
      });
      index += 1;
      res
    });
    Self {
      row_count: dim,
      col_count: dim,
      data,
    }
  }

  /// gives rows count
  pub const fn rows(&self) -> usize
  {
    self.row_count
  }

  /// gives cols count
  pub const fn cols(&self) -> usize
  {
    self.col_count
  }

  /// Get a row
  pub fn row(&self, idx: usize) -> Result<Vec<&f64>>
  {
    if idx >= self.row_count
    {
      return Err(MatrixError::OutOfBoundsRC(idx));
    }
    Ok(self.data[idx].iter().collect())
  }
  pub fn row_mut(&mut self, idx: usize) -> Result<Vec<&mut f64>>
  {
    if idx >= self.row_count
    {
      return Err(MatrixError::OutOfBoundsRC(idx));
    }
    Ok(self.data[idx].iter_mut().collect())
  }

  /// Get a col
  pub fn col(&self, idx: usize) -> Result<Vec<&f64>>
  {
    if idx >= self.col_count
    {
      return Err(MatrixError::OutOfBoundsRC(idx));
    }
    Ok(self.data.iter().map(|x| &x[idx]).collect())
  }
  pub fn col_mut(&mut self, idx: usize) -> Result<Vec<&mut f64>>
  {
    if idx >= self.col_count
    {
      return Err(MatrixError::OutOfBoundsRC(idx));
    }
    Ok(self.data.iter_mut().map(|x| &mut x[idx]).collect())
  }

  /// Returns a matrix with the given row removed
  pub fn remove_row(mut self, row: usize) -> Result<Self>
  {
    if row >= self.rows()
    {
      return Err(MatrixError::OutOfBoundsRC(row));
    }
    self.data.remove(row);
    self.row_count -= 1;
    Ok(self)
  }

  /// Returns a matrix with the given col removed
  pub fn remove_col(mut self, col: usize) -> Result<Self>
  {
    if col >= self.cols()
    {
      return Err(MatrixError::OutOfBoundsRC(col));
    }
    self.data.iter_mut().for_each(|x| {
      x.remove(col);
    });
    self.col_count -= 1;
    Ok(self)
  }

  /// [Determinant](https://en.wikipedia.org/wiki/Determinant)
  pub fn det(&self) -> Result<f64>
  {
    // non-square matrices dont have a determinant
    if self.rows() != self.cols()
    {
      return Err(MatrixError::NonSquareMatrix);
    }

    // base case
    if self.rows() == 2
    {
      return Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]);
    }

    // get the coefficient row
    let coeff: Vec<f64> = self.row(0)?.iter().cloned().cloned().collect();

    // the matrix without the top row
    let working = self.clone().remove_row(0)?;

    // start positive
    let mut sign = 1.0;
    let mut ret = 0.0;
    for i in 0..coeff.len()
    {
      // the eq that is part of the sum
      // sign * coefficient of the removed column * determinant of the submatrix
      ret += sign * coeff[i] * working.clone().remove_col(i)?.det()?;

      // flip the sign for the next one
      sign = -sign;
    }

    Ok(ret)
  }

  /// [Cofactor matrix](https://en.wikipedia.org/wiki/Minor_(linear_algebra))
  pub fn cofactor_matrix(&self) -> Result<Self>
  {
    // should get tossed out by det but dont wanna do all the extra stuff if we know its uneccesary
    if self.row_count != self.col_count
    {
      return Err(MatrixError::NonSquareMatrix);
    }

    let mut data = Vec::new();
    data.resize(self.row_count, Vec::new());
    for i in 0..self.row_count
    {
      for j in 0..self.col_count
      {
        // C_{i,j} == (-1)^{i+j} * det(SM_{i,j})
        // get the determinant of the submatrix missing row i and col j, mul by the right sign
        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        data[i].push(sign * self.clone().remove_row(i)?.remove_col(j)?.det()?);
      }
    }

    Self::new(data)
  }

  /// [Transpose](https://en.wikipedia.org/wiki/Transpose)
  pub fn transpose(&self) -> Self
  {
    let mut i = 0;
    let mut data = Vec::new();
    data.resize_with(self.cols(), || {
      let old = i;
      i += 1;
      self.col(old).unwrap().iter().map(|x| **x).collect()
    });

    Self::new(data).unwrap()
  }

  /// [Inverse](https://en.wikipedia.org/wiki/Invertible_matrix)
  pub fn inverse(&self) -> Result<Self>
  {
    // A^{-1} = 1/det(A) * CofactorMatrix(A)^{T}
    // CofactorMatrix(A)^T is called the adj matrix
    Ok(1.0 / self.det()? * self.cofactor_matrix()?.transpose())
  }

  pub fn trace(&self) -> Result<f64>
  {
    if self.rows() != self.cols()
    {
      return Err(MatrixError::NonSquareMatrix);
    }
    Ok(
      self
        .data
        .iter()
        .flatten()
        .step_by(self.cols() + 1)
        .cloned()
        .sum(),
    )
  }

  /// Getters
  pub fn get(&self, row: usize, col: usize) -> Result<&f64>
  {
    if row >= self.row_count || col >= self.col_count
    {
      Err(MatrixError::OutOfBounds(row, col))
    }
    else
    {
      Ok(&self.data[row][col])
    }
  }

  pub fn get_mut(&mut self, row: usize, col: usize) -> Result<&mut f64>
  {
    if row >= self.row_count || col >= self.col_count
    {
      Err(MatrixError::OutOfBounds(row, col))
    }
    else
    {
      Ok(&mut self.data[row][col])
    }
  }

  pub fn data(&self) -> Vec<Vec<f64>>
  {
    self.data.clone()
  }

  // increases dimension by 1
  pub fn expand(&self) -> Self
  {
    let mut ret = self.clone();
    let mut row = Vec::new();
    row.resize(self.col_count + 1, 0.0f64);
    row[self.col_count] = 1.0;
    for x in &mut ret.data
    {
      x.push(0.0);
    }
    ret.data.push(row);
    ret.row_count += 1;
    ret.col_count += 1;
    ret
  }

  pub fn translation(trans: Vec3) -> MatrixDynamic
  {
    let mut ret = MatrixDynamic::identity(4);
    let mut col = ret.col_mut(3).unwrap();
    *col[0] = trans.x() as f64;
    *col[1] = trans.y() as f64;
    *col[2] = trans.z() as f64;
    ret
  }

  pub fn rotation(axis: Vec3, theta: f32) -> MatrixDynamic
  {
    let axis = axis.matrix();
    Quaternion::rotation(theta as f64, axis)
      .unwrap()
      .rotation_matrix()
      .expand()
  }
  pub fn scale(scale: Vec3) -> MatrixDynamic
  {
    let mut ret = MatrixDynamic::identity(4);
    ret.data[0][0] = scale.x() as f64;
    ret.data[1][1] = scale.y() as f64;
    ret.data[2][2] = scale.z() as f64;
    ret
  }
  pub fn perspective(fov: f64, near: f64, far: f64, aspect: f64) -> MatrixDynamic
  {
    MatrixDynamic::new(vec![
      vec![1.0 / (aspect * f64::tan(fov / 2.0)), 0.0, 0.0, 0.0],
      vec![0.0, 1.0 / (aspect * f64::tan(fov / 2.0)), 0.0, 0.0],
      vec![
        0.0,
        0.0,
        -(far + near) / (far - near),
        -(2.0 * far * near) / (far - near),
      ],
      vec![0.0, 0.0, -1.0, 0.0],
    ])
    .unwrap()
  }
  pub fn flatten(&self) -> Vec<f64>
  {
    self.data.iter().cloned().flatten().collect()
  }
}

impl Add for MatrixDynamic
{
  type Output = Result<Self>;

  fn add(self, rhs: Self) -> Self::Output
  {
    if self.col_count != rhs.col_count || self.row_count != rhs.row_count
    {
      Err(MatrixError::MissmatchDimension {
        lhs: (self.row_count, self.col_count),
        rhs: (rhs.row_count, rhs.col_count),
      })
    }
    else
    {
      let mut ret = self;
      ret
        .data
        .iter_mut()
        .flatten()
        .zip(rhs.data.iter().flatten())
        .for_each(|(x, y)| *x += *y);
      Ok(ret)
    }
  }
}

impl Sub for MatrixDynamic
{
  type Output = Result<Self>;

  fn sub(self, rhs: Self) -> Self::Output
  {
    self + (-rhs)
  }
}

impl Mul<MatrixDynamic> for f64
{
  type Output = MatrixDynamic;

  fn mul(self, rhs: MatrixDynamic) -> Self::Output
  {
    let mut ret = rhs;
    ret.data.iter_mut().flatten().for_each(|x| *x *= self);
    ret
  }
}
impl Mul<f64> for MatrixDynamic
{
  type Output = Self;

  fn mul(self, rhs: f64) -> Self::Output
  {
    let mut ret = self;
    ret.data.iter_mut().flatten().for_each(|x| *x *= rhs);
    ret
  }
}

impl<const S: usize> Mul<Vector<S>> for MatrixDynamic
{
  type Output = Result<Self>;

  fn mul(self, rhs: Vector<S>) -> Self::Output
  {
    self * MatrixDynamic::from(rhs.into())
  }
}

impl Mul for MatrixDynamic
{
  type Output = Result<Self>;

  fn mul(self, rhs: Self) -> Self::Output
  {
    if self.cols() != rhs.rows()
    {
      return Err(MatrixError::MissmatchDimension {
        lhs: (self.rows(), self.cols()),
        rhs: (rhs.rows(), rhs.cols()),
      });
    }

    let mut data = Vec::new();
    data.resize(self.rows(), Vec::new());
    data.iter_mut().for_each(|x| x.resize(rhs.cols(), 0.0));

    for row in 0..self.rows()
    {
      for col in 0..rhs.cols()
      {
        let value = self
          .row(row)?
          .iter()
          .zip(rhs.col(col)?.iter())
          .map(|(x, y)| *x * *y)
          .sum::<f64>();
        data[row][col] = value;
      }
    }

    Self::new(data)
  }
}

impl Neg for MatrixDynamic
{
  type Output = Self;

  fn neg(self) -> Self::Output
  {
    -1.0 * self
  }
}

impl Display for MatrixDynamic
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    write!(f, "[")?;
    for i in 0..self.data.len()
    {
      let v = &self.data[i];
      write!(f, "[")?;
      for j in 0..v.len()
      {
        write!(f, "{}", v[j])?;
        if j + 1 != v.len()
        {
          write!(f, ", ")?;
        }
      }
      if i + 1 != self.data.len()
      {
        write!(f, ", ")?;
      }
      write!(f, "]")?;
    }
    write!(f, "]")
  }
}

#[cfg(test)]
mod tests
{
  use crate::{Vec3, matrix::MatrixDynamic};

  #[test]
  fn identity()
  {
    assert_eq!(
      MatrixDynamic::identity(2),
      MatrixDynamic::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap()
    );
    assert_eq!(
      MatrixDynamic::identity(3),
      MatrixDynamic::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0]
      ])
      .unwrap()
    );
  }
  #[test]
  fn det()
  {
    assert_eq!(
      MatrixDynamic::new(vec![
        vec![1.0, 4.0, 2.0, 3.0],
        vec![2.0, 3.0, 7.0, 0.0],
        vec![3.0, 7.0, 4.0, 4.0],
        vec![9.0, 1.0, 5.0, 0.0]
      ])
      .unwrap()
      .det()
      .unwrap(),
      205.0
    );
  }

  #[test]
  fn cofactor_matrix()
  {
    assert_eq!(
      MatrixDynamic::new(vec![
        vec![1.0, 4.0, 2.0, 3.0],
        vec![2.0, 3.0, 7.0, 0.0],
        vec![3.0, 7.0, 4.0, 4.0],
        vec![9.0, 1.0, 5.0, 0.0]
      ])
      .unwrap()
      .cofactor_matrix()
      .unwrap(),
      MatrixDynamic::new(vec![
        vec![-32.0, -212.0, 100.0, 295.0],
        vec![-21.0, -11.0, 40.0, -5.0],
        vec![24.0, 159.0, -75.0, -170.0],
        vec![23.0, -27.0, 5.0, 25.0]
      ])
      .unwrap()
    );
  }

  #[test]
  fn transpose()
  {
    assert_eq!(
      MatrixDynamic::new(vec![
        vec![1.0, 4.0, 2.0, 3.0],
        vec![2.0, 3.0, 7.0, 0.0],
        vec![3.0, 7.0, 4.0, 4.0],
        vec![9.0, 1.0, 5.0, 0.0]
      ])
      .unwrap()
      .transpose(),
      MatrixDynamic::new(vec![
        vec![1.0, 2.0, 3.0, 9.0],
        vec![4.0, 3.0, 7.0, 1.0],
        vec![2.0, 7.0, 4.0, 5.0],
        vec![3.0, 0.0, 4.0, 0.0]
      ])
      .unwrap()
    );
  }

  #[test]
  fn inverse()
  {
    assert_eq!(
      MatrixDynamic::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, -2.0, 1.0],
        vec![4.0, 1.0, 1.0],
      ])
      .unwrap()
      .inverse()
      .unwrap(),
      MatrixDynamic::new(vec![
        vec![-3.0 / 32.0, 1.0 / 32.0, 1.0 / 4.0],
        vec![1.0 / 32.0, -11.0 / 32.0, 1.0 / 4.0],
        vec![11.0 / 32.0, 7.0 / 32.0, -1.0 / 4.0]
      ])
      .unwrap()
    );
  }
  #[test]
  fn rotate()
  {
    let my_rot = MatrixDynamic::rotation(Vec3::new(0.0, 0.0, 1.0), 75.0f32.to_radians());
    assert_eq!(
      my_rot,
      MatrixDynamic::new(vec![
        vec![0.258819045102521, 0.965925826289068, 0.0],
        vec![-0.965925826289068, 0.258819045102521, 0.0],
        vec![0.0, 0.0, 1.0]
      ])
      .unwrap()
      .expand()
    )
  }
}
