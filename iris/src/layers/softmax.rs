use serde::{Deserialize, Serialize};

use crate::layers::Layer;


#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SoftMax {
    num_inputs: usize
}

impl SoftMax {
    pub fn new(num_inputs: usize) -> Self {
        Self { num_inputs }
    }
}

#[typetag::serde]
impl Layer for SoftMax {
    fn input_size(&self) -> usize {
        self.num_inputs
    }

    fn output_size(&self) -> usize {
        self.num_inputs
    }

    /// Turn some random result scores into a probably scaled from [0, 1]
    /// with the total sum of the output being 1.0.
    fn transform(&self, scores: &[f32]) -> Box<[f32]> {
        let mut sum = 0.0;
        let mut exps = vec![0.0; self.num_inputs];
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for c in 0 .. self.num_inputs {
            exps[c] = (scores[c] - max).exp();
            sum += exps[c];
        }

        for c in 0 .. self.num_inputs {
            exps[c] /= sum;
        }

        return exps.into_boxed_slice();
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
