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
    /// Panics if shapes don't match.
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
}
