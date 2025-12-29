use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::tensor::{Shape, Storage};

/// A multi-dimensional array with automatic differentiation support.
///
/// Tensor combines:
/// - `storage`: The raw data as a flat array
/// - `shape`: The logical dimensions
/// - `strides`: How to navigate memory for each dimension
/// - `offset`: Starting position in storage (for views)
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    ///
    /// # Example
    /// ```
    /// use delta::tensor::Tensor;
    /// let t = Tensor::zeros(&[2, 3]); // 2x3 matrix of zeros
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let shape = Shape::new(shape);
        let strides = shape.strides();
        let storage = Storage::zeros(shape.nelems());
        Self {
            storage,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create a tensor from a vector of data.
    ///
    /// # Panics
    /// Panics if data length doesn't match shape.
    ///
    /// # Example
    /// ```
    /// use delta::tensor::Tensor;
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let shape = Shape::new(shape);
        assert_eq!(
            data.len(),
            shape.nelems(),
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape.dims(),
            shape.nelems()
        );
        let strides = shape.strides();
        let storage = Storage::from_vec(data);
        Self {
            storage,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Returns the shape as a slice.
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements.
    pub fn nelems(&self) -> usize {
        self.shape.nelems()
    }

    /// Convert multi-dimensional indices to linear memory index.
    ///
    /// Uses strides: index = offset + sum(indices[i] * strides[i])
    pub fn linear_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.ndim(),
            "Expected {} indices, got {}",
            self.ndim(),
            indices.len()
        );
        self.offset
            + indices
                .iter()
                .zip(&self.strides)
                .map(|(i, s)| i * s)
                .sum::<usize>()
    }

    /// Get element at the given indices.
    ///
    /// # Panics
    /// Panics if indices are out of bounds or wrong number of indices.
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.linear_index(indices);
        self.storage.as_slice()[idx]
    }

    /// Set element at the given indices.
    ///
    /// # Panics
    /// Panics if indices are out of bounds or wrong number of indices.
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.linear_index(indices);
        self.storage.as_mut_slice()[idx] = value;
    }

    /// Element-wise addition: self + other
    ///
    /// # Panics
    /// Panics if shapes do not match.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );

        let data: Vec<f32> = self
            .storage
            .as_slice()
            .iter()
            .zip(other.storage.as_slice())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::from_vec(data, self.shape())
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );

        let data: Vec<f32> = self
            .storage
            .as_slice()
            .iter()
            .zip(other.storage.as_slice())
            .map(|(a, b)| a - b)
            .collect();

        Tensor::from_vec(data, self.shape())
    }

    /// Element-wise multiplication: self * other (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );

        let data: Vec<f32> = self
            .storage
            .as_slice()
            .iter()
            .zip(other.storage.as_slice())
            .map(|(a, b)| a * b)
            .collect();

        Tensor::from_vec(data, self.shape())
    }

    /// Element-wise division: self / other
    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch: {:?} vs {:?}",
            self.shape(),
            other.shape()
        );

        let data: Vec<f32> = self
            .storage
            .as_slice()
            .iter()
            .zip(other.storage.as_slice())
            .map(|(a, b)| a / b)
            .collect();

        Tensor::from_vec(data, self.shape())
    }

    /// Add a scalar to all elements
    pub fn scalar_add(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.storage.as_slice().iter().map(|x| x + scalar).collect();
        Tensor::from_vec(data, self.shape())
    }

    /// Multiply all elements by a scalar
    pub fn scalar_mul(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.storage.as_slice().iter().map(|x| x * scalar).collect();
        Tensor::from_vec(data, self.shape())
    }

    /// Negate all elements: -self
    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = self.storage.as_slice().iter().map(|x| -x).collect();
        Tensor::from_vec(data, self.shape())
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            self.ndim()
        );
        assert_eq!(
            other.ndim(),
            2,
            "matmul requires 2D tensors, got {}D",
            other.ndim()
        );

        let (m, k1) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        assert_eq!(
            k1, k2,
            "Inner dimensions must match: ({}, {}) @ ({}, {})",
            m, k1, k2, n
        );

        let mut result = Tensor::zeros(&[m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self.get(&[i, k]) * other.get(&[k, j]);
                }
                result.set(&[i, j], sum);
            }
        }
        result
    }

    /// Helper for recursive tensor formatting
    fn fmt_recursive(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        dim: usize,
        offset: &mut usize,
    ) -> std::fmt::Result {
        if dim == self.ndim() {
            // Base case: print single element
            write!(f, "{:.4}", self.storage.as_slice()[*offset])?;
            *offset += 1;
            return Ok(());
        }

        write!(f, "[")?;
        let size = self.shape()[dim];

        // Truncate large dimensions
        let max_items = 6;
        let truncated = size > max_items;

        for i in 0..size {
            if truncated && i == max_items / 2 {
                // Skip middle elements
                let skip = size - max_items;
                *offset += skip * self.strides[dim];
                write!(f, "..., ")?;
                continue;
            }

            if truncated && i > max_items / 2 && i < size - max_items / 2 {
                continue;
            }

            self.fmt_recursive(f, dim + 1, offset)?;

            if i < size - 1 {
                write!(f, ", ")?;
                // Add newline for 2D+ tensors between rows
                if dim < self.ndim() - 1 {
                    writeln!(f)?;
                    // Indent based on depth
                    for _ in 0..=dim {
                        write!(f, " ")?;
                    }
                }
            }
        }

        write!(f, "]")
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(")?;
        self.fmt_recursive(f, 0, &mut 0)?;
        write!(f, ", shape={:?})", self.shape())
    }
}

// ----- Neg (unary minus) -----

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

// ----- Add -----

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(&self, &rhs)
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(&self, rhs)
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::add(self, &rhs)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

// ----- Sub -----

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(&self, &rhs)
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(&self, rhs)
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::sub(self, &rhs)
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

// ----- Mul (element-wise) -----

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        Tensor::mul(&self, &rhs)
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::mul(&self, rhs)
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        Tensor::mul(self, &rhs)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::mul(self, rhs)
    }
}

// ----- Div -----

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        Tensor::div(&self, &rhs)
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        Tensor::div(&self, rhs)
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        Tensor::div(self, &rhs)
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        Tensor::div(self, rhs)
    }
}

// ----- Scalar multiplication: Tensor * f32 -----

impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        self.scalar_mul(rhs)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        self.scalar_mul(rhs)
    }
}

// f32 * Tensor
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        rhs.scalar_mul(self)
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        rhs.scalar_mul(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.nelems(), 6);
        assert_eq!(t.get(&[0, 0]), 0.0);
        assert_eq!(t.get(&[1, 2]), 0.0);
    }

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Row 0: [1, 2, 3]
        // Row 1: [4, 5, 6]
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_set() {
        let mut t = Tensor::zeros(&[2, 2]);
        t.set(&[0, 1], 42.0);
        assert_eq!(t.get(&[0, 1]), 42.0);
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_linear_index() {
        let t = Tensor::zeros(&[2, 3]);
        // strides = [3, 1]
        assert_eq!(t.linear_index(&[0, 0]), 0);
        assert_eq!(t.linear_index(&[0, 1]), 1);
        assert_eq!(t.linear_index(&[1, 0]), 3);
        assert_eq!(t.linear_index(&[1, 2]), 5);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_from_vec_shape_mismatch() {
        Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 3]); // 3 != 6
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
        let c = a.add(&b);
        assert_eq!(c.get(&[0]), 5.0);
        assert_eq!(c.get(&[1]), 7.0);
        assert_eq!(c.get(&[2]), 9.0);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let c = a.sub(&b);
        assert_eq!(c.get(&[0]), 4.0);
        assert_eq!(c.get(&[1]), 5.0);
        assert_eq!(c.get(&[2]), 6.0);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0], &[3]);
        let c = a.mul(&b);
        assert_eq!(c.get(&[0]), 10.0);
        assert_eq!(c.get(&[1]), 18.0);
        assert_eq!(c.get(&[2]), 28.0);
    }

    #[test]
    fn test_div() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);
        let b = Tensor::from_vec(vec![2.0, 4.0, 5.0], &[3]);
        let c = a.div(&b);
        assert_eq!(c.get(&[0]), 5.0);
        assert_eq!(c.get(&[1]), 5.0);
        assert_eq!(c.get(&[2]), 6.0);
    }

    #[test]
    fn test_scalar_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.scalar_add(10.0);
        assert_eq!(b.get(&[0]), 11.0);
        assert_eq!(b.get(&[1]), 12.0);
        assert_eq!(b.get(&[2]), 13.0);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let b = a.scalar_mul(3.0);
        assert_eq!(b.get(&[0]), 3.0);
        assert_eq!(b.get(&[1]), 6.0);
        assert_eq!(b.get(&[2]), 9.0);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3]);
        let b = a.neg();
        assert_eq!(b.get(&[0]), -1.0);
        assert_eq!(b.get(&[1]), 2.0);
        assert_eq!(b.get(&[2]), -3.0);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_add_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let _ = a.add(&b);
    }

    #[test]
    fn test_display() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = format!("{}", t);
        assert!(s.contains("Tensor"));
        assert!(s.contains("shape=[2, 3]"));
    }

    #[test]
    fn test_operator_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2]);

        // All 4 variants
        let c1 = &a + &b;
        let c2 = a.clone() + &b;
        let c3 = &a + b.clone();
        let c4 = a.clone() + b.clone();

        assert_eq!(c1.get(&[0]), 4.0);
        assert_eq!(c2.get(&[0]), 4.0);
        assert_eq!(c3.get(&[0]), 4.0);
        assert_eq!(c4.get(&[0]), 4.0);
    }

    #[test]
    fn test_operator_neg() {
        let a = Tensor::from_vec(vec![1.0, -2.0], &[2]);
        let b = -&a; // reference version
        assert_eq!(b.get(&[0]), -1.0);
        assert_eq!(b.get(&[1]), 2.0);

        let c = -a; // owned version
        assert_eq!(c.get(&[0]), -1.0);
    }

    #[test]
    fn test_operator_scalar_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]);

        let b = &a * 3.0; // Tensor * f32
        let c = 3.0 * &a; // f32 * Tensor

        assert_eq!(b.get(&[0]), 3.0);
        assert_eq!(c.get(&[0]), 3.0);
    }
}
