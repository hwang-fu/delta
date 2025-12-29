/// Represents the dimensions of a tensor.
///
/// For example, a 2x3 matrix has shape [2, 3].
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creat a new shape from a slice of dimensions.
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    /// Returns the number of dimensions (rank of the tensor).
    ///
    /// Scalar = 0, Vector = 1, Matrix = 2, etc.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// For shape [2, 3], nelems = 6.
    pub fn nelems(&self) -> usize {
        self.dims.iter().product()
    }
}
