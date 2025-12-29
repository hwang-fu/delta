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

    /// Returns a reference to the dimensions slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Computes row-major (C-style) strides.
    ///
    /// Strides tell us how many elements to skip in memory
    /// to move one step along each dimension.
    ///
    /// For shape [2, 3]:
    ///   - stride[0] = 3 (skip 3 elements to go to next row)
    ///   - stride[1] = 1 (skip 1 element to go to next column)
    pub fn strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }
}
