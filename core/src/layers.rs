pub mod input_schema;
pub mod output_schema;
pub mod weights;
pub mod zero_neg;

use ndarray::{Array2, ArrayView2};

use crate::data::{DataSheet, DataStorage};

/// Layers are simply something that takes x inputs and turns them into y outputs.
/// These may be anything from an output schema for probability outputs, predict via weights,
/// or an input schema to turn data sheet rows into normalized inputs.
///
/// Every layer works on a batch at a time: a 2D array with one row per data sheet row and one
/// column per input or output of the layer.
///
/// A model's first layer reads data sheet rows through `encode_inputs`, and its last layer
/// encodes, scores, and reads back outputs through `encode_targets`, `output_loss`,
/// `count_correct`, and `decode_outputs`.  Only the [`InputSchema`](input_schema::InputSchema)
/// and [`OutputSchema`](output_schema::OutputSchema) layers implement those, every other layer
/// keeps the default that returns an error.
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

    /// Transform a batch of inputs into a batch of outputs, one output row per input row.  The
    /// number of columns MUST always match the sizes defined by the above elements.
    fn transform(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>>;

    /// Used while training to pass outputs back through the model with what
    /// the outputs should have been for training.
    /// Only used in training.
    ///
    /// Given the batch of `inputs` this layer saw, the `outputs` it produced, and dLoss/dOutput
    /// for each row, adds the batch's parameter gradients to the layer's running totals and
    /// returns dLoss/dInput for each row for the previous layer.
    fn back_propagate(
        &mut self,
        inputs: ArrayView2<f32>,
        outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>>;

    /// Apply any steps required at the end of each training batch, such as stepping parameters
    /// against the batch's `n` rows of gradients.
    /// Only used in training.
    fn perform_epoch_step(&mut self, _learning_rate: f32, _n: f32) -> anyhow::Result<()> { Ok(()) }

    /// Converts data sheet rows into a batch of this layer's inputs, one row per data sheet row.
    /// Called on a model's first layer to start every prediction and training pass.
    fn encode_inputs(&self, _rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        anyhow::bail!("this layer cannot read data sheet rows, a model must start with an InputSchema layer")
    }

    /// Converts the expected output values of data sheet rows into a batch shaped like this
    /// layer's outputs, one row per data sheet row.  Called on a model's last layer to prepare
    /// the targets for `output_loss` and `count_correct`.
    fn encode_targets(&self, _rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        anyhow::bail!("this layer cannot encode targets, a model must end with an OutputSchema layer")
    }

    /// Scores a batch of this layer's `outputs` against `targets` from `encode_targets`,
    /// returning the loss summed over every row and dLoss/dOutput for each row.  Called on a
    /// model's last layer during training.
    fn output_loss(
        &self,
        _outputs: ArrayView2<f32>,
        _targets: ArrayView2<f32>
    ) -> anyhow::Result<(f32, Array2<f32>)> {
        anyhow::bail!("this layer cannot score outputs, a model must end with an OutputSchema layer")
    }

    /// Counts the rows of a batch of this layer's `outputs` whose every predicted value matches
    /// `targets` from `encode_targets`.  Called on a model's last layer to measure accuracy.
    fn count_correct(
        &self,
        _outputs: ArrayView2<f32>,
        _targets: ArrayView2<f32>
    ) -> anyhow::Result<usize> {
        anyhow::bail!("this layer cannot score outputs, a model must end with an OutputSchema layer")
    }

    /// Reads a batch of this layer's `outputs` back into data sheet values, one list per row,
    /// each value paired with the index of the column it predicts.  Called on a model's last
    /// layer to make predictions.
    fn decode_outputs(&self, _outputs: ArrayView2<f32>) -> anyhow::Result<Vec<Vec<(usize, DataStorage)>>> {
        anyhow::bail!("this layer cannot decode outputs, a model must end with an OutputSchema layer")
    }
}
