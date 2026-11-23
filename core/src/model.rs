use std::{collections::LinkedList, time::Instant};

use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};

use crate::{data::{DataSheet, DataStorage}, designer::ModelDesigner, layers::Layer};

/// A trained or trainable stack of layers that predicts the output columns of a data sheet row
/// from its input columns.
///
/// Fields:
/// * layers: The layers in the order data flows through them, starting with an `InputSchema` and
///   ending with an `OutputSchema`.
#[derive(Getters, Serialize, Deserialize)]
pub struct Model {
    #[getset(get = "pub")]
    layers: LinkedList<Box<dyn Layer>>
}

impl Model {

    /// Create a new `Model` instance.
    pub fn new(layers: LinkedList<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    /// Create a new `ModelDesigner`.
    pub fn designer() -> ModelDesigner {
        ModelDesigner::new()
    }

    /// Runs a data sheet row through every layer and gives the final layer's outputs, the
    /// probability of every option of every output column.
    pub fn find_probabilities(&self, row: &[DataStorage]) -> anyhow::Result<Box<[f32]>> {
        let mut results = self.first_layer()?.encode_row(row)?;
        for layer in self.layers.iter() {
            results = layer.transform(&results)?;
        }
        return Ok(results);
    }

    /// Predicts the value of every output column for a data sheet row.
    pub fn predict(&self, row: &[DataStorage]) -> anyhow::Result<Vec<(usize, DataStorage)>> {
        let probabilities = self.find_probabilities(row)?;
        self.last_layer()?.decode_outputs(&probabilities)
    }

    /// Perform a training epoch over every row of a data sheet.  The average loss (miss) is
    /// returned at the end of the epoch, useful to see if training is giving better results.
    pub fn train(&mut self, data: &DataSheet, learning_rate: f32) -> anyhow::Result<f32> {
        let mut total_loss = 0.0;
        let mut rows = 0;

        for row in data.iter() {
            rows += 1;

            // forward pass, keeping every layer's output; activations[i] is layer i's input
            let mut activations: Vec<Box<[f32]>> = vec![self.first_layer()?.encode_row(row)?];
            for (i, layer) in self.layers.iter().enumerate() {
                let next = layer.transform(&activations[i])?;
                activations.push(next);
            }

            // the last layer scores its outputs against the row's expected values
            let (loss, mut grads) = self
                .last_layer()?
                .output_loss(&activations[self.layers.len()], row)?;
            total_loss += loss;

            // backward pass, each layer handing dLoss/dInput to the one before it
            for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                grads = layer
                    .back_propagate(&activations[i], &activations[i + 1], &grads)
                    .with_context(|| format!("backward pass failed at layer {i}"))?;
            }
        }
        anyhow::ensure!(rows > 0, "no rows to train on");

        // step every layer's parameters once per epoch
        let n = rows as f32;
        for layer in &mut self.layers {
            layer.perform_epoch_step(learning_rate, n)?;
        }

        Ok(total_loss / n)
    }

    /// Trains this model until its stops improving by the `min_req_improve` threshold by
    /// loss for `max_epoch_no_improve` epochs.
    pub fn train_until_stop_improvement(
        &mut self,
        data: &DataSheet,
        learning_rate: f32,
        min_req_improve: f32,
        max_epoch_no_improve: u32
    ) -> anyhow::Result<()> {
        let start_time = Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch = 0;

        loop {
            let loss = self.train(data, learning_rate)?;
            epoch += 1;

            if loss < best_loss * (1.0 - min_req_improve) {
                best_loss = loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if epoch % 100 == 0 {
                println!("Epoch {epoch}:  Loss {loss:.4}  Accuracy {:.1}%", self.est_accuracy(data)? * 100.0);
            }

            if epochs_without_improvement >= max_epoch_no_improve {
                let end = start_time.elapsed();
                println!("Stopped at epoch {epoch}: loss plateaued at {best_loss:.4}, accuracy {:.2}%, training took {} ms", self.est_accuracy(data)? * 100.0, end.as_millis());
                break;
            }
        }

        Ok(())
    }

    /// Estimates the accuracy of this model as the fraction of rows whose output columns are all
    /// predicted correctly.
    pub fn est_accuracy(&self, data: &DataSheet) -> anyhow::Result<f32> {
        let mut correct = 0;
        let mut rows = 0;
        for row in data.iter() {
            rows += 1;
            let predictions = self.predict(row)?;
            if predictions.iter().all(|(idx, value)| row.get(*idx) == Some(value)) {
                correct += 1;
            }
        }
        anyhow::ensure!(rows > 0, "no rows to estimate accuracy on");

        return Ok(correct as f32 / rows as f32);
    }

    /// The layer that reads data sheet rows.
    fn first_layer(&self) -> anyhow::Result<&dyn Layer> {
        self.layers
            .front()
            .map(|layer| layer.as_ref())
            .context("model has no layers")
    }

    /// The layer that scores and decodes outputs.
    fn last_layer(&self) -> anyhow::Result<&dyn Layer> {
        self.layers
            .back()
            .map(|layer| layer.as_ref())
            .context("model has no layers")
    }
}
