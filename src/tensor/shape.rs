/// Represents the dimensions of a tensor.
///
/// For example, a 2x3 matrix has shape [2, 3].
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}
