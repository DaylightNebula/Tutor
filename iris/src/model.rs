use std::time::Instant;

use getset::Getters;

use crate::{layers::{Layer, scalars::Scalars, softmax::SoftMax, weights::Weights}};

#[derive(Getters, Clone, Debug)]
pub struct Model<const WIDTH: usize, const DEPTH: usize> {
    scalars: Scalars,
    weights: Weights,
    softmax: SoftMax
}

impl <const WIDTH: usize, const DEPTH: usize> Model<WIDTH, DEPTH> {

    /// Build a new "empty" model from some scalars.
    pub fn from_scalars(scalars: Scalars) -> Self {
        Self {
            scalars,
            weights: Weights::new(WIDTH, DEPTH),
            softmax: SoftMax::new(DEPTH)
        }
    }

    /// Take a piece of row of data and give the probabilities of its outputs.
    pub fn find_probabilities(&self, data: &[f32; WIDTH]) -> Box<[f32]> {
        let scalars_result = self.scalars.transform(data);
        let weights_result = self.weights.transform(&scalars_result);
        return self.softmax.transform(&*weights_result);
    }

    /// Predict which of the <depth> possible options for the given input data
    /// is most likely.
    pub fn predict(&self, data: &[f32; WIDTH]) -> usize {
        let mut best = 0;
        let probs = self.find_probabilities(data);

        for c in 1..DEPTH {
            if probs[c] > probs[best] {
                best = c;
            }
        }

        return best;
    }

    /// Perform a training epoch from a list of data.  The data should be given a list of inputs
    /// paired with the index of what should be the index of the highest output.  The average loss
    /// (miss) is returned at the end of the epoch, useful to see if training is giving better
    /// results..
    pub fn train(&mut self, data: &[([f32; WIDTH], usize)], learning_rate: f32) -> f32 {
        let mut grad_w = vec![vec![0.0; WIDTH]; DEPTH];
        let mut grad_b = [0.0; DEPTH];
        let mut total_loss = 0.0;
        let n = data.len() as f32;

        // loop through each row and target output pair, execute on that data row, then find the loss (how much we missed).
        for (input, target_output_idx) in data {
            let probabilities = self.find_probabilities(input);
            total_loss += -(probabilities[*target_output_idx] + 1e-7).ln(); // add a small amount to never have an even 0 in the data

            // track how far off the weights and biases where
            let scaled = self.scalars.transform(input); // we need the inputs the weights saw, not our raw inputs
            for c in 0 .. DEPTH {
                let target = if c == *target_output_idx { 1.0 } else { 0.0 };
                let error = probabilities[c] - target;
                for j in 0 .. WIDTH {
                    grad_w[c][j] += error * scaled[j];
                }
                grad_b[c] += error;
            }
        }

        // tweak the biases and weights by the learning rate multiplied by how wrong they where
        // done via weights back propogation step
        self.weights.back_propagate(&grad_w, &grad_b, learning_rate, n);

        total_loss / n
    }

    /// Trains this model until its stops improving by the `min_req_improve` threshold by
    /// loss for `max_epoch_no_improve` epochs.
    pub fn train_until_stop_improvement(
        &mut self,
        data: &[([f32; WIDTH], usize)],
        learning_rate: f32,
        min_req_improve: f32,
        max_epoch_no_improve: u32
    ) {
        let start_time = Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch = 0;

        loop {
            let loss = self.train(&data, learning_rate);
            epoch += 1;

            if loss < best_loss * (1.0 - min_req_improve) {
                best_loss = loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if epoch % 100 == 0 {
                println!("Epoch {epoch}:  Loss {loss:.4}  Accuracy {:.1}%", self.est_accuracy(&data) * 100.0);
            }

            if epochs_without_improvement >= max_epoch_no_improve || self.est_accuracy(&data) >= 1.0 {
                let end = start_time.elapsed();
                println!("Stopped at epoch {epoch}: loss plateaued at {best_loss:.4}, training took {} ms", end.as_millis());
                break;
            }
        }
    }

    /// Use a given list of inputs and expected output pairs, estimate the accuracy of this model
    /// as a percentage.
    pub fn est_accuracy(&self, data: &[([f32; WIDTH], usize)]) -> f32 {
        let correct = data
            .iter()
            .filter(|(x, label)| self.predict(x) == *label)
            .count();
        return correct as f32 / data.len() as f32;
    }
}
