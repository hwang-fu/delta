/// Raw data storage for tensor elements.
///
/// Storage is a simple wrapper around a flat Vec<f32>.
/// The interpretation of this data (shape, strides) is handled by Tensor.
#[derive(Debug, Clone)]
pub struct Storage {
    data: Vec<f32>,
}

impl Storage {
    /// Create storage initialized with zeros.
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }
}
