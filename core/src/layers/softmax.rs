use serde::{Deserialize, Serialize};

use crate::layers::Layer;

/// SoftMax (probability output layer)
///
/// # Purpose
///
/// Turns a list of raw scores (any real numbers, sometimes called logits) into a list of
/// probabilities.  Every output is between 0 and 1 and all outputs add up to exactly 1, so the
/// result can be read as "how likely is each option".  It is normally the last layer of a
/// classification model, placed after a [`Weights`](crate::layers::weights::Weights) layer.
///
/// # How it works
///
/// For K input scores z_1 … z_K, the output for score i is:
///
/// ```text
/// y_i = e^(z_i) / Σ_{k=1..K} e^(z_k)
/// ```
///
/// To avoid overflow, the largest score m = max_k z_k is subtracted first.  This gives the
/// exact same answer because the e^(−m) factor cancels between top and bottom:
///
/// ```text
/// y_i = e^(z_i − m) / Σ_{k=1..K} e^(z_k − m)
/// ```
///
/// During back propagation, given g_i = ∂L/∂y_i, the gradient for the previous layer is:
///
/// ```text
/// ∂L/∂z_i = y_i · ( g_i − Σ_{k=1..K} g_k · y_k )
/// ```
///
/// In plain algebra:
/// * e ≈ 2.718 is a constant.  Raising e to a power always gives a positive number, and bigger
///   scores give much bigger results, so the highest score stands out.
/// * Each output is "my positive number divided by the total of all positive numbers", which is
///   the same as finding what fraction of a pie each slice is.  That is why the outputs always
///   add up to 1.
/// * Example: scores [2, 1, 0] become e² ≈ 7.39, e¹ ≈ 2.72, e⁰ = 1.  The total is ≈ 11.11, so
///   the probabilities are ≈ [0.67, 0.24, 0.09].
/// * Subtracting the largest score first keeps the powers small so the computer does not run
///   out of room for huge numbers.  Scores [1002, 1001, 1000] give the same answer as [2, 1, 0].
/// * Going backwards, raising one score raises its own probability but lowers all the others
///   (the pie is still one pie).  The formula accounts for this by taking each output's own
///   gradient and subtracting the weighted average gradient of all outputs, then scaling by
///   that output's probability.  There is nothing to learn in this layer.
///
/// # Example uses
/// * Picking which of three iris species a flower most likely is.
/// * Digit recognition (which of 0–9 is in an image).
/// * Any multi-class classification where exactly one answer is correct.
///
/// Fields:
/// * num_inputs: Number of scores taken in, which is also the number of probabilities output.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SoftMax;

#[typetag::serde]
impl Layer for SoftMax {

    fn output_size(&self, previous_layer_num_outputs: usize) -> usize {
        previous_layer_num_outputs
    }

    fn prepare_train(
        &mut self,
        _sample_data: &[(Box<[f32]>, usize)],
        _previous_layer_num_outputs: usize
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Turn some random result scores into a probably scaled from [0, 1]
    /// with the total sum of the output being 1.0.
    fn transform(&self, scores: &[f32]) -> anyhow::Result<Box<[f32]>> {
        let mut sum = 0.0;
        let mut exps = vec![0.0; scores.len()];
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for c in 0 .. scores.len() {
            exps[c] = (scores[c] - max).exp();
            sum += exps[c];
        }

        for c in 0 .. scores.len() {
            exps[c] /= sum;
        }

        return Ok(exps.into_boxed_slice());
    }

    /// Translate the outputs we got to the outputs the previous layer would have given.
    /// Do the same for the gradients (dLoss / dOutput).
    fn back_propagate(
        &mut self,
        _inputs: &[f32],
        outputs: &[f32],
        output_gradient: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        // Σ g·y appears in every input's gradient, so compute it once
        let dot: f32 = output_gradient
            .iter()
            .zip(outputs)
            .map(|(g, y)| g * y)
            .sum();

        // dL/dz_i = y_i · (g_i − Σ g·y)
        let input_grads = outputs
            .iter()
            .zip(output_gradient)
            .map(|(y, g)| y * (g - dot))
            .collect::<Vec<_>>();

        Ok(input_grads.into_boxed_slice())
    }
}
