pub mod input_schema;
pub mod output_schema;
pub mod weights;
pub mod zero_neg;

use crate::data::{DataSheet, DataStorage};

/// Layers are simply something that takes x inputs and turns them into y outputs.
/// These may be anything from an output schema for probability outputs, predict via weights,
/// or an input schema to turn data sheet rows into normalized inputs.
///
/// A model's first layer reads data sheet rows through `encode_row`, and its last layer scores
/// and reads back outputs through `output_loss` and `decode_outputs`.  Only the
/// [`InputSchema`](input_schema::InputSchema) and [`OutputSchema`](output_schema::OutputSchema)
/// layers implement those, every other layer keeps the default that returns an error.
#[typetag::serde(tag = "type")]
pub trait Layer {

    /// The number of outputs this stage will produce, useful for validating
    /// that this stage will produce the expected outputs of the model or
    /// expected inputs of the next stage.
    fn output_size(&self, previous_layer_num_outputs: usize) -> usize;

    /// Prepare this layer for training.  This may be clearing out old data to
    /// avoid influence of previous runs, fitting schemas to the sample data, or
    /// anything else that is needed to ensure clean repeatable training runs.
    fn prepare_train(
        &mut self,
        sample_data: &DataSheet,
        previous_layer_num_outputs: usize
    ) -> anyhow::Result<()>;

    /// Transform some inputs into some outputs with lengths defined by the
    /// above elements.  They MUST always have lengths matching those lengths.
    fn transform(&self, inputs: &[f32]) -> anyhow::Result<Box<[f32]>>;

    /// Used while training to pass outputs back through the model with what
    /// the outputs should have been for training.
    /// Only used in training.
    ///
    ///  Given the `inputs` this layer saw, the `outputs` it
    /// produced, and dLoss/dOutput, adds this row's parameter gradients to the layer's running
    /// totals and returns dLoss/dInput for the previous layer.
    fn back_propagate(
        &mut self,
        inputs: &[f32],
        outputs: &[f32],
        output_gradient: &[f32]
    ) -> anyhow::Result<Box<[f32]>>;

    /// Apply any steps required at the end of each epoch for training.
    /// Only used in training.
    fn perform_epoch_step(&mut self, _learning_rate: f32, _n: f32) -> anyhow::Result<()> { Ok(()) }

    /// Converts a data sheet row into this layer's inputs.  Called on a model's first layer to
    /// start every prediction and training pass.
    fn encode_row(&self, _row: &[DataStorage]) -> anyhow::Result<Box<[f32]>> {
        anyhow::bail!("this layer cannot read data sheet rows, a model must start with an InputSchema layer")
    }

    /// Scores this layer's `outputs` against the expected values in a data sheet `row`, returning
    /// the loss and dLoss/dOutput.  Called on a model's last layer during training.
    fn output_loss(&self, _outputs: &[f32], _row: &[DataStorage]) -> anyhow::Result<(f32, Box<[f32]>)> {
        anyhow::bail!("this layer cannot score outputs, a model must end with an OutputSchema layer")
    }

    /// Reads this layer's `outputs` back into data sheet values, each paired with the index of
    /// the column it predicts.  Called on a model's last layer to make predictions.
    fn decode_outputs(&self, _outputs: &[f32]) -> anyhow::Result<Vec<(usize, DataStorage)>> {
        anyhow::bail!("this layer cannot decode outputs, a model must end with an OutputSchema layer")
    }
}
