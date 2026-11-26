use getset::CopyGetters;
use ndarray::{Array2, ArrayView2, Zip};
use serde::{Deserialize, Serialize};

use crate::{data::DataSheet, layers::{Layer, LayerType}};

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
/// This layer has no fields, its number of outputs is always its number of inputs.
#[derive(Copy, Clone, Debug, CopyGetters, Serialize, Deserialize)]
pub struct ZeroNegative;

inventory::submit! {
    LayerType { name: "ZeroNegative", create: || Box::new(ZeroNegative) }
}

#[typetag::serde]
impl Layer for ZeroNegative {

    /// One output per input.
    fn output_size(&self, input_count: usize) -> usize { input_count }

    /// Nothing to prepare, the layer has no parameters.
    fn prepare_train(&mut self, _sample_data: &DataSheet, _input_count: usize) -> anyhow::Result<()> {
        Ok(())
    }

    /// Replaces every negative input with 0.
    fn transform(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        Ok(inputs.mapv(|x| x.max(0.0)))
    }

    /// Passes back the gradient of every positive input and 0 for every other, the derivative of
    /// `transform`.
    fn back_propagate(
        &mut self,
        inputs: ArrayView2<f32>,
        _outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>> {
        anyhow::ensure!(
            inputs.dim() == output_gradient.dim(),
            "ZeroNegative got {:?} inputs but {:?} gradients", inputs.dim(), output_gradient.dim()
        );

        // only rows whose input was positive pass their gradient back
        let mut input_gradient = output_gradient.to_owned();
        Zip::from(&mut input_gradient)
            .and(&inputs)
            .for_each(|gradient, &x| if x <= 0.0 { *gradient = 0.0 });
        Ok(input_gradient)
    }
}
