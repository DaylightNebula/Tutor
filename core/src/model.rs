use std::{collections::LinkedList, time::Instant};

use anyhow::Context;
use getset::{CopyGetters, Getters};
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

#[derive(Getters, CopyGetters, Serialize, Deserialize)]
pub struct Model {
    #[getset(get = "pub")]
    layers: LinkedList<Box<dyn Layer>>,
    #[getset(get_copy = "pub")]
    num_inputs: usize,
    #[getset(get_copy = "pub")]
    num_outputs: usize,
    #[getset(get_copy = "pub")]
    cur_width: usize
}

impl Model {

    /// Create a new blank model.
    pub fn blank(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            layers: LinkedList::new(),
            num_inputs,
            num_outputs,
            cur_width: num_inputs
        }
    }

    /// Add a new layer to this model, this adds to the end of the layer stack.
    pub fn add_layer<L: Layer + 'static>(mut self, layer: L) -> anyhow::Result<Self> {
        if layer.input_size() != self.cur_width() { anyhow::bail!("Input size detached") }
        self.cur_width = layer.output_size();
        self.layers.push_back(Box::new(layer));
        return Ok(self);
    }

    /// Take a piece of row of data and give the probabilities of its outputs.
    pub fn find_probabilities(&self, inputs: &[f32]) -> Box<[f32]> {
        let mut results = inputs.to_vec().into_boxed_slice();
        for layer in self.layers.iter() {
            results = layer.transform(&results);
        }
        return results;
    }

    /// Predict which of the <depth> possible options for the given input data
    /// is most likely.
    pub fn predict(&self, data: &[f32]) -> usize {
        let mut best = 0;
        let probs = self.find_probabilities(data);

        for c in 1 .. self.num_outputs {
            if probs[c] > probs[best] {
                best = c;
            }
        }

        return best;
    }

    /// Perform a training epoch from a list of data.  The data should be given a list of inputs
    /// paired with the index of what should be the index of the highest output.  The average loss
    /// (miss) is returned at the end of the epoch, useful to see if training is giving better
    /// results.
    pub fn train(&mut self, data: &[(Box<[f32]>, usize)], learning_rate: f32) -> anyhow::Result<f32> {
        let mut total_loss = 0.0;
        let n = data.len() as f32;

        for (input, target) in data {

            // forward pass, keeping every layer's output; activations[i] is layer i's input
            let mut activations: Vec<Box<[f32]>> = vec![input.to_vec().into_boxed_slice()];
            for (i, layer) in self.layers.iter().enumerate() {
                let next = layer.transform(&activations[i]);
                activations.push(next);
            }

            // cross-entropy loss and its gradient with respect to the final probabilities
            let probabilities = &activations[self.layers.len()];
            let p = probabilities[*target] + 1e-7;
            total_loss += -p.ln();
            let mut grads = vec![0.0; self.num_outputs].into_boxed_slice();
            grads[*target] = -1.0 / p;

            // backward pass, each layer handing dLoss/dInput to the one before it
            for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                grads = layer
                    .back_propagate(&activations[i], &activations[i + 1], &grads)
                    .with_context(|| format!("backward pass failed at layer {i}"))?;
            }
        }

        // step every layer's parameters once per epoch
        for layer in &mut self.layers {
            layer.perform_epoch_step(learning_rate, n);
        }

        Ok(total_loss / n)
    }

    /// Trains this model until its stops improving by the `min_req_improve` threshold by
    /// loss for `max_epoch_no_improve` epochs.
    pub fn train_until_stop_improvement(
        &mut self,
        data: &[(Box<[f32]>, usize)],
        learning_rate: f32,
        min_req_improve: f32,
        max_epoch_no_improve: u32
    ) -> anyhow::Result<()> {
        let start_time = Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch = 0;

        loop {
            let loss = self.train(&data, learning_rate)?;
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

            if epochs_without_improvement >= max_epoch_no_improve {
                let end = start_time.elapsed();
                println!("Stopped at epoch {epoch}: loss plateaued at {best_loss:.4}, accuracy {:.2}%, training took {} ms", self.est_accuracy(&data) * 100.0, end.as_millis());
                break;
            }
        }

        Ok(())
    }

    /// Use a given list of inputs and expected output pairs, estimate the accuracy of this model
    /// as a percentage.
    pub fn est_accuracy(&self, data: &[(Box<[f32]>, usize)]) -> f32 {
        let correct = data
            .iter()
            .filter(|(x, label)| self.predict(x) == *label)
            .count();
        return correct as f32 / data.len() as f32;
    }
}
