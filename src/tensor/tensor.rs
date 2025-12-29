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
}
