use getset::CopyGetters;
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

/// ZeroNegative (ReLU activation layer)
///
/// # Purpose
///
/// Replaces every negative input with 0 and passes positive inputs through unchanged.  Also
/// known as ReLU (Rectified Linear Unit).  Stacking [`Weights`](crate::layers::weights::Weights)
/// layers directly on top of each other collapses into a single linear equation no matter how
/// many there are, so the model can only learn straight-line relationships.  Placing this layer
/// between them adds a "bend", letting the model learn curved and more complex patterns.
///
/// # How it works
///
/// For each input x_j:
///
/// ```text
/// y_j = max(0, x_j)
/// ```
///
/// Its derivative is a step function, so during back propagation with g_j = ∂L/∂y_j:
///
/// ```text
/// ∂y_j/∂x_j = { 1 if x_j > 0
///             { 0 otherwise
///
/// ∂L/∂x_j = g_j if x_j > 0, otherwise 0
/// ```
///
/// In plain algebra:
/// * The rule is "if the number is negative, make it 0; otherwise keep it".  Inputs
///   [−3, 0.5, 2, −1] become [0, 0.5, 2, 0].
/// * On a graph it is a flat line along 0 for negatives that bends into the line y = x for
///   positives.  That single bend is what lets stacked layers draw non-straight boundaries.
/// * Going backwards: where the input was positive, the output moved exactly as much as the
///   input (slope 1), so the gradient passes through unchanged.  Where the input was negative,
///   the output was stuck at 0 (slope 0), so no gradient passes back.  There is nothing to learn
///   in this layer.
///
/// # Example uses
/// * Activation between two `Weights` layers to form a hidden layer, e.g.
///   `Weights(4→8) → ZeroNegative(8) → Weights(8→3)`.
/// * Hidden layers in image, text, or tabular classifiers; it is the most common default
///   activation because it is cheap and trains well.
/// * Clamping any intermediate value that must not be negative.
///
/// Fields:
/// * width: Number of inputs, which is also the number of outputs.
#[derive(Copy, Clone, Debug, CopyGetters, Serialize, Deserialize)]
pub struct ZeroNegative {
    #[getset(get_copy = "pub")]
    width: usize
}

impl ZeroNegative {
    /// Create a no negatives layer.
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

#[typetag::serde]
impl Layer for ZeroNegative {
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
