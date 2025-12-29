/// Raw data storage for tensor elements.
///
/// Storage is a simple wrapper around a flat Vec<f32>.
/// The interpretation of this data (shape, strides) is handled by Tensor.
#[derive(Debug, Clone)]
pub struct Storage {
    data: Vec<f32>,
}
