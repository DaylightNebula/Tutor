use getset::{CopyGetters, Getters};

use crate::layers::Layer;

/// The raw weights of the model, the numbers that we use various matrix maths
/// to calculate the probabilities of an option being the best result for a given input.
#[derive(CopyGetters, Getters, Clone, Debug)]
pub struct Weights {
    #[getset(get = "pub")]
    weights: Vec<Vec<f32>>,
    #[getset(get = "pub")]
    biases: Box<[f32]>,
    #[getset(get = "pub")]
    gradient_weights: Vec<Vec<f32>>,
    #[getset(get = "pub")]
    gradient_biases: Box<[f32]>,
    #[getset(get_copy = "pub")]
    width: usize,
    #[getset(get_copy = "pub")]
    depth: usize
}

impl Weights {
    /// Create a new weights instance of the given size.
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            weights: vec![vec![0.0; width]; depth],
            biases: vec![0.0; depth].into_boxed_slice(),
            gradient_weights: vec![vec![0.0; width]; depth],
            gradient_biases: vec![0.0; depth].into_boxed_slice(),
            width, depth
        }
    }
}

impl Layer for Weights {

    fn input_size(&self) -> usize {
        self.width
    }

    fn output_size(&self) -> usize {
        self.depth
    }

    /// Use the model to take <width> inputs, score, then return the <depth> outputs.
    ///
    /// The output numbers from this will be kind of random, not a probability, use
    /// a function like a softmax to turn it into a probability of one output being the
    /// most likely.
    fn transform(&self, input: &[f32]) -> Box<[f32]> {
        let mut s = vec![0.0; self.depth];

        for c in 0 .. self.depth {
            s[c] = self.biases[c];
            for j in 0 .. self.width {
                s[c] += self.weights[c][j] * input[j];
            }
        }

        return s.into_boxed_slice();
    }

    fn back_propagate(
        &mut self,
        inputs: &[f32],
        _outputs: &[f32],
        output_grads: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        // add this row's parameter gradients and pass dLoss/dInput back
        let mut input_grads = vec![0.0; self.width];
        for c in 0 .. self.depth {
            for j in 0 .. self.width {
                self.gradient_weights[c][j] += output_grads[c] * inputs[j];
                input_grads[j] += self.weights[c][j] * output_grads[c];
            }
            self.gradient_biases[c] += output_grads[c];
        }

        Ok(input_grads.into_boxed_slice())
    }

    /// Steps the weights against the averaged gradients, then clears them for the next epoch.
    fn perform_epoch_step(&mut self, learning_rate: f32, n: f32) {
        for c in 0 .. self.depth {
            for j in 0 .. self.width {
                self.weights[c][j] -= learning_rate * self.gradient_weights[c][j] / n;
                self.gradient_weights[c][j] = 0.0;
            }
            self.biases[c] -= learning_rate * self.gradient_biases[c] / n;
            self.gradient_biases[c] = 0.0;
        }
    }
}
