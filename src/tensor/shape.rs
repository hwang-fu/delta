/// Represents the dimensions of a tensor.
///
/// For example, a 2x3 matrix has shape [2, 3].
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creat a new shape from a slice of dimensions.
    ///
    /// # Example
    /// ```
    /// let shape = Shape::new(&[2, 3]); // 2 rows, 3 columns
    /// ```
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
    /// For shape [2, 3], strides = [3, 1]:
    ///   - stride[0] = 3 (skip 3 elements to go to next row)
    ///   - stride[1] = 1 (skip 1 element to go to next column)
    ///
    /// Logical view (2D matrix):
    /// ```text
    ///           col 0   col 1   col 2
    ///          ┌───────────────────────┐
    ///   row 0  │   A       B       C   │
    ///   row 1  │   D       E       F   │
    ///          └───────────────────────┘
    /// ```
    ///
    /// Memory layout (flat array):
    /// ```text
    ///   index:   0     1     2     3     4     5
    ///          ┌─────┬─────┬─────┬─────┬─────┬─────┐
    ///          │  A  │  B  │  C  │  D  │  E  │  F  │
    ///          └─────┴─────┴─────┴─────┴─────┴─────┘
    /// ```
    ///
    /// Strides = [3, 1]:
    ///   - strides[0] = 3: move one row down (A -> D), skip 3 elements
    ///   - strides[1] = 1: move one column right (A -> B), skip 1 element
    ///
    /// To access element at [row, col]:
    ///   - memory_index = row * strides[0] + col * strides[1]
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
