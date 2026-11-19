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
            width, depth
        }
    }

    /// Back propogate modification to the weights of models.
    /// Used to modify weights to produce more accurate results for the
    /// same inputs in the future.
    pub fn back_propagate(
        &mut self,
        weight_grades: &Vec<Vec<f32>>,
        bias_grades: &[f32],
        learning_rate: f32,
        n: f32
    ) {
        for c in 0 .. self.depth {
            for j in 0 .. self.width {
                self.weights[c][j] -= learning_rate * weight_grades[c][j] / n;
            }
            self.biases[c] -= learning_rate * bias_grades[c] / n;
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
}
