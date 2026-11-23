use std::{collections::LinkedList, time::Instant};

use anyhow::Context;
use getset::Getters;
use ndarray::{Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::{data::{DataSheet, DataStorage}, designer::ModelDesigner, layers::Layer};

/// A trained or trainable stack of layers that predicts the output columns of a data sheet row
/// from its input columns.
#[derive(Getters, Serialize, Deserialize)]
pub struct Model {
    #[getset(get = "pub")]
    layers: LinkedList<Box<dyn Layer>>
}

/// The rows of a data sheet encoded into the numbers a [`Model`] works with, made once by
/// `Model::encode` and reused across training epochs and accuracy checks.
#[derive(Getters, Clone, Debug)]
pub struct EncodedSheet {
    #[getset(get = "pub")]
    inputs: Array2<f32>,
    #[getset(get = "pub")]
    targets: Array2<f32>
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

    /// Encodes the inputs and expected outputs of every row of a data sheet, so training and
    /// accuracy checks can reuse them instead of encoding the rows again every epoch.
    pub fn encode(&self, data: &DataSheet) -> anyhow::Result<EncodedSheet> {
        let rows = data
            .iter()
            .collect::<Vec<_>>();
        anyhow::ensure!(!rows.is_empty(), "no rows to encode");

        Ok(EncodedSheet {
            inputs: self.first_layer()?.encode_inputs(&rows)?,
            targets: self.last_layer()?.encode_targets(&rows)?
        })
    }

    /// Runs a data sheet row through every layer and gives the final layer's outputs, the
    /// probability of every option of every output column.
    pub fn find_probabilities(&self, row: &[DataStorage]) -> anyhow::Result<Box<[f32]>> {
        let inputs = self.first_layer()?.encode_inputs(&[row])?;
        let probabilities = self.run(inputs.view())?;
        return Ok(probabilities.iter().copied().collect());
    }

    /// Predicts the value of every output column for a data sheet row.
    pub fn predict(&self, row: &[DataStorage]) -> anyhow::Result<Vec<(usize, DataStorage)>> {
        let inputs = self.first_layer()?.encode_inputs(&[row])?;
        let probabilities = self.run(inputs.view())?;
        self.last_layer()?
            .decode_outputs(probabilities.view())?
            .into_iter()
            .next()
            .context("model gave no prediction for the row")
    }

    /// Perform a training epoch over every row of a data sheet.  The average loss (miss) is
    /// returned at the end of the epoch, useful to see if training is giving better results.
    pub fn train(&mut self, data: &DataSheet, learning_rate: f32, batch_size: usize) -> anyhow::Result<f32> {
        let encoded = self.encode(data)?;
        self.train_encoded(&encoded, learning_rate, batch_size)
    }

    /// Perform a training epoch over every row of an encoded data sheet.  The rows are shuffled
    /// then split into batches of `batch_size` rows, and the parameters take one step per batch.
    /// The average loss (miss) over the epoch is returned.
    pub fn train_encoded(
        &mut self,
        data: &EncodedSheet,
        learning_rate: f32,
        batch_size: usize
    ) -> anyhow::Result<f32> {
        let rows = data.inputs.nrows();
        anyhow::ensure!(rows > 0, "no rows to train on");
        anyhow::ensure!(batch_size > 0, "batch size must be at least 1");

        // visit the rows in a fresh random order every epoch
        let mut order = (0 .. rows).collect::<Vec<_>>();
        order.shuffle(&mut rand::thread_rng());

        // one step per batch of rows
        let mut total_loss = 0.0;
        for batch in order.chunks(batch_size) {
            let inputs = data.inputs.select(Axis(0), batch);
            let targets = data.targets.select(Axis(0), batch);
            total_loss += self.train_batch(inputs.view(), targets.view(), learning_rate)?;
        }

        Ok(total_loss / rows as f32)
    }

    /// Runs one batch forward and backward and steps every layer's parameters against the
    /// batch's average gradient.  Gives the loss summed over the batch's rows.
    fn train_batch(
        &mut self,
        inputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        learning_rate: f32
    ) -> anyhow::Result<f32> {
        // forward pass, keeping every layer's output; outputs[i] is layer i's output
        let mut outputs: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_inputs = outputs
                .last()
                .map_or(inputs.view(), |output| output.view());
            let next = layer
                .transform(layer_inputs)
                .with_context(|| format!("forward pass failed at layer {i}"))?;
            outputs.push(next);
        }

        // the last layer scores its outputs against the expected values
        let final_outputs = outputs
            .last()
            .context("model has no layers")?;
        let (batch_loss, mut grads) = self
            .last_layer()?
            .output_loss(final_outputs.view(), targets)?;

        // backward pass, each layer handing dLoss/dInput to the one before it
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_inputs = if i == 0 { inputs.view() } else { outputs[i - 1].view() };
            grads = layer
                .back_propagate(layer_inputs, outputs[i].view(), grads.view())
                .with_context(|| format!("backward pass failed at layer {i}"))?;
        }

        // step every layer's parameters once per batch
        let n = inputs.nrows() as f32;
        for layer in &mut self.layers {
            layer.perform_epoch_step(learning_rate, n)?;
        }

        Ok(batch_loss)
    }

    /// Trains this model until its stops improving by the `min_req_improve` threshold by
    /// loss for `max_epoch_no_improve` epochs.
    pub fn train_until_stop_improvement(
        &mut self,
        data: &DataSheet,
        learning_rate: f32,
        batch_size: usize,
        min_req_improve: f32,
        max_epoch_no_improve: u32
    ) -> anyhow::Result<()> {
        let start_time = Instant::now();
        let encoded = self.encode(data)?;
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch = 0;

        loop {
            let loss = self.train_encoded(&encoded, learning_rate, batch_size)?;
            epoch += 1;

            if loss < best_loss * (1.0 - min_req_improve) {
                best_loss = loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if epoch % 100 == 0 {
                println!("Epoch {epoch}:  Loss {loss:.4}  Accuracy {:.1}%", self.est_accuracy_encoded(&encoded)? * 100.0);
            }

            if epochs_without_improvement >= max_epoch_no_improve {
                let end = start_time.elapsed();
                println!("Stopped at epoch {epoch}: loss plateaued at {best_loss:.4}, accuracy {:.2}%, training took {} ms", self.est_accuracy_encoded(&encoded)? * 100.0, end.as_millis());
                break;
            }
        }

        Ok(())
    }

    /// Estimates the accuracy of this model as the fraction of rows whose output columns are all
    /// predicted correctly.
    pub fn est_accuracy(&self, data: &DataSheet) -> anyhow::Result<f32> {
        let encoded = self.encode(data)?;
        self.est_accuracy_encoded(&encoded)
    }

    /// Estimates the accuracy of this model over an encoded data sheet as the fraction of rows
    /// whose output columns are all predicted correctly.
    pub fn est_accuracy_encoded(&self, data: &EncodedSheet) -> anyhow::Result<f32> {
        let rows = data.inputs.nrows();
        anyhow::ensure!(rows > 0, "no rows to estimate accuracy on");

        let probabilities = self.run(data.inputs.view())?;
        let correct = self
            .last_layer()?
            .count_correct(probabilities.view(), data.targets.view())?;

        return Ok(correct as f32 / rows as f32);
    }

    /// Runs a batch of encoded inputs through every layer and gives the last layer's outputs.
    fn run(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        let mut results = inputs.to_owned();
        for (i, layer) in self.layers.iter().enumerate() {
            results = layer
                .transform(results.view())
                .with_context(|| format!("forward pass failed at layer {i}"))?;
        }
        Ok(results)
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
