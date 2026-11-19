use getset::CopyGetters;

/// The raw weights of the model, the numbers that we use various matrix maths
/// to calculate the probabilities of an option being the best result for a given input.
#[derive(CopyGetters, Clone, Copy, Debug)]
pub struct Weights<const WIDTH: usize, const DEPTH: usize> {
    #[getset(get_copy = "pub")]
    weights: [[f32; WIDTH]; DEPTH],
    #[getset(get_copy = "pub")]
    biases: [f32; DEPTH]
}

impl <const WIDTH: usize, const DEPTH: usize> Default for Weights<WIDTH, DEPTH> {
    fn default() -> Self {
        Self {
            weights: [[0.0; WIDTH]; DEPTH],
            biases: [0.0; DEPTH]
        }
    }
}

impl <const WIDTH: usize, const DEPTH: usize> Weights<WIDTH, DEPTH> {

    /// Use the model to take <width> inputs, score, then return the <depth> outputs.
    ///
    /// The output numbers from this will be kind of random, not a probability, use
    /// a function like a softmax to turn it into a probability of one output being the
    /// most likely.
    pub fn transform(&self, input: &[f32; WIDTH]) -> [f32; DEPTH] {
        let mut s = [0.0; DEPTH];

        for c in 0 .. DEPTH {
            s[c] = self.biases[c];
            for j in 0 .. WIDTH {
                s[c] += self.weights[c][j] * input[j];
            }
        }

        return s;
    }

    /// Back propogate modification to the weights of models.
    /// Used to modify weights to produce more accurate results for the
    /// same inputs in the future.
    pub fn back_propagate(
        &mut self,
        grad_w: [[f32; WIDTH]; DEPTH],
        grad_b: [f32; DEPTH],
        learning_rate: f32,
        n: f32
    ) {
        for c in 0 .. DEPTH {
            for j in 0 .. WIDTH {
                self.weights[c][j] -= learning_rate * grad_w[c][j] / n;
            }
            self.biases[c] -= learning_rate * grad_b[c] / n;
        }
    }
}
