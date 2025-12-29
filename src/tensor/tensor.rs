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

    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.linear_index(indices);
        self.storage.as_slice()[idx]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.linear_index(indices);
        self.storage.as_mut_slice()[idx] = value;
    }
}
