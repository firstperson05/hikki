pub mod autograd;
pub mod ops;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Creates a new tensor with the given shape and contiguous data.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        let expected_len: usize = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        if !shape.is_empty() && data.len() < expected_len {
            return Err(format!(
                "Data length {} is less than expected shape product {}",
                data.len(),
                expected_len
            ));
        }

        let strides = Self::compute_strides(&shape);
        Ok(Tensor {
            shape,
            strides,
            data,
        })
    }

    /// Computes contiguous strides for a given shape.
    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        strides
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = if shape.is_empty() {
            0
        } else {
            shape.iter().product()
        };
        Self::new(shape, vec![0.0; size]).unwrap()
    }

    /// Checks if the tensor memory layout is contiguous.
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.strides, vec![3, 1]);
        assert_eq!(t.data.len(), 6);
        assert!(t.is_contiguous());
    }
}
