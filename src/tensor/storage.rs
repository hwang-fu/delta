/// Raw data storage for tensor elements.
///
/// Storage is a simple wrapper around a flat Vec<f32>.
/// The interpretation of this data (shape, strides) is handled by Tensor.
#[derive(Debug, Clone)]
pub struct Storage {
    data: Vec<f32>,
}

impl Storage {
    /// Create storage initialized with zeros (0.0).
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    /// Create storage from an existing vector.
    ///
    /// Takes ownership of the data (no copy).
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Returns an immutable slice of the underlying data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Returns the number of elements in storage.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let storage = Storage::zeros(5);
        assert_eq!(storage.len(), 5);
        assert_eq!(storage.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_from_vec() {
        let storage = Storage::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut storage = Storage::zeros(3);
        storage.as_mut_slice()[0] = 42.0;
        assert_eq!(storage.as_slice()[0], 42.0);
    }

    #[test]
    fn test_is_empty() {
        assert!(Storage::zeros(0).is_empty());
        assert!(!Storage::zeros(1).is_empty());
    }
}
