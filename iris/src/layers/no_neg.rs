use getset::CopyGetters;
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

/// Layer that replaces all negative inputs with 0 then returns them as outputs.
/// AKA ReLU.
#[derive(Copy, Clone, Debug, CopyGetters, Serialize, Deserialize)]
pub struct NoNegative {
    #[getset(get_copy = "pub")]
    width: usize
}

impl NoNegative {
    /// Create a no negatives layer.
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

#[typetag::serde]
impl Layer for NoNegative {
    fn input_size(&self) -> usize { self.width }
    fn output_size(&self) -> usize { self.width }

    fn transform(&self, inputs: &[f32]) -> Box<[f32]> {
        inputs.iter().map(|x| x.max(0.0)).collect()
    }

    /// Apply the same rule as transform by zipping inputs with gradients to
    /// effectively reverse this layer.
    fn back_propagate(
        &mut self,
        inputs: &[f32],
        _outputs: &[f32],
        output_gradient: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        Ok(
            inputs.iter()
                .zip(output_gradient)
                .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
                .collect()
        )
    }
}
