pub mod input_schema;
pub mod output_schema;
pub mod weights;
pub mod zero_neg;

use std::fmt::Debug;

use ndarray::{Array2, ArrayView2};

use crate::{data::{DataSheet, DataStorage}, model::ColumnScore};

/// A kind of layer that can be created without any settings, listed with `inventory::submit!`
/// next to the layer's `Layer` impl so tools such as the model editor can offer every layer.
///
/// Fields:
/// * name: The layer's typetag name, the `type` it is saved with.
/// * create: Creates the layer with default settings, ready to be added to a model design.
#[derive(Debug)]
pub struct LayerType {
    pub name: &'static str,
    pub create: fn() -> Box<dyn Layer>
}

inventory::collect!(LayerType);

/// Every registered layer type, sorted by name so menus list them in a stable order.
///
/// Types are collected from every `inventory::submit!` of a [`LayerType`] in the program, so
/// layers defined outside this crate are included as long as their crate is linked in.
pub fn layer_types() -> Vec<&'static LayerType> {
    let mut types = inventory::iter::<LayerType>
        .into_iter()
        .collect::<Vec<_>>();
    types.sort_by_key(|layer_type| layer_type.name);
    types
}

/// Layers are simply something that takes x inputs and turns them into y outputs.
/// These may be anything from an output schema for probability outputs, predict via weights,
/// or an input schema to turn data sheet rows into normalized inputs.
///
/// Every layer works on a batch at a time: a 2D array with one row per data sheet row and one
/// column per input or output of the layer.
///
/// Each node of a [`Model`](crate::model::Model) runs one layer.  A model's input nodes read
/// data sheet rows through `encode_inputs`, and its output nodes encode, score, and read back
/// outputs through `encode_targets`, `output_loss`, `evaluate`, and `decode_outputs`.  Only the
/// [`InputSchema`](input_schema::InputSchema) and [`OutputSchema`](output_schema::OutputSchema)
/// layers implement those, every other layer keeps the default that returns an error.
///
/// Layers are `Debug + Send` so tools such as the model editor can hold and move them between
/// threads, and every layer type is listed with [`LayerType`] so tools can offer new ones.
#[typetag::serde(tag = "type")]
pub trait Layer: Debug + Send {

    /// The number of outputs this layer will produce per row, used to size the layers after it
    /// and to check that the widths of a model's layers line up.
    ///
    /// # Parameters
    /// * previous_layer_num_outputs: The number of inputs this layer gets per row, the total
    ///   width of the nodes feeding it.
    fn output_size(&self, previous_layer_num_outputs: usize) -> usize;

    /// Prepares this layer for training.  This may be clearing out old data to avoid influence
    /// of previous runs, fitting schemas to the sample data, sizing parameters, or anything else
    /// that is needed to ensure clean repeatable training runs.  Called once per node when a
    /// model is built, after every node feeding it has been prepared.
    ///
    /// # Parameters
    /// * sample_data: Rows the model is built and trained with.
    /// * previous_layer_num_outputs: The number of inputs this layer gets per row, the total
    ///   width of the nodes feeding it.
    ///
    /// # Errors
    /// * The layer cannot be fitted to `sample_data` or sized to its inputs.
    fn prepare_train(
        &mut self,
        sample_data: &DataSheet,
        previous_layer_num_outputs: usize
    ) -> anyhow::Result<()>;

    /// Transforms a batch of inputs into a batch of outputs, one output row per input row.  The
    /// number of columns MUST always match `output_size`.
    ///
    /// # Parameters
    /// * inputs: The batch, one row per data sheet row and one column per input.
    ///
    /// # Errors
    /// * The layer has not been prepared.
    /// * `inputs` does not have the number of columns the layer was prepared for.
    fn transform(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>>;

    /// Passes gradients backwards through this layer during training.
    ///
    /// Adds the batch's parameter gradients to the layer's running totals and returns
    /// dLoss/dInput for each row, which the model splits between the nodes feeding this one.
    ///
    /// # Parameters
    /// * inputs: The batch this layer saw in the forward pass.
    /// * outputs: The batch this layer produced from `inputs`.
    /// * output_gradient: dLoss/dOutput for each row, summed over every node this one feeds.
    ///
    /// # Errors
    /// * The layer has not been prepared.
    /// * The shapes of `inputs`, `outputs`, and `output_gradient` do not match the layer.
    fn back_propagate(
        &mut self,
        inputs: ArrayView2<f32>,
        outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>>;

    /// Applies any steps required at the end of each training batch, such as stepping parameters
    /// against the batch's gradients.  Only used in training, the default does nothing.
    ///
    /// # Parameters
    /// * learning_rate: How far to step the parameters against their gradients.
    /// * n: The number of rows in the batch, to average the gradient totals over.
    ///
    /// # Errors
    /// * The layer has not been prepared or has no gradients to step with.
    fn perform_epoch_step(&mut self, _learning_rate: f32, _n: f32) -> anyhow::Result<()> { Ok(()) }

    /// Converts data sheet rows into a batch of this layer's inputs, one row per data sheet row.
    /// Called on a model's input nodes to start every prediction and training pass.
    ///
    /// # Parameters
    /// * rows: The data sheet rows to encode, each laid out in column index order.
    ///
    /// # Errors
    /// * The layer cannot read data sheet rows, the default.
    /// * The layer has not been prepared or a row cannot be encoded.
    fn encode_inputs(&self, _rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        anyhow::bail!("this layer cannot read data sheet rows, a model must start with an InputSchema layer")
    }

    /// Converts the expected output values of data sheet rows into a batch shaped like this
    /// layer's outputs, one row per data sheet row.  Called on a model's output nodes to prepare
    /// the targets for `output_loss` and `evaluate`.
    ///
    /// # Parameters
    /// * rows: The data sheet rows to encode, each laid out in column index order.
    ///
    /// # Errors
    /// * The layer cannot encode targets, the default.
    /// * The layer has not been prepared or a row cannot be encoded.
    fn encode_targets(&self, _rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        anyhow::bail!("this layer cannot encode targets, a model must end with an OutputSchema layer")
    }

    /// Scores a batch of this layer's `outputs` against `targets` from `encode_targets`,
    /// returning the loss summed over every row and the gradient for each row that is handed to
    /// this layer's `back_propagate`.  That is usually dLoss/dOutput, but a layer may fold its own
    /// derivative in, as [`OutputSchema`](output_schema::OutputSchema) does.  Called on a model's
    /// output nodes during training.
    ///
    /// # Parameters
    /// * outputs: A batch this layer produced.
    /// * targets: The expected values of the same rows, from `encode_targets`.
    ///
    /// # Errors
    /// * The layer cannot score outputs, the default.
    /// * The layer has not been prepared or the shapes of `outputs` and `targets` do not match.
    fn output_loss(
        &self,
        _outputs: ArrayView2<f32>,
        _targets: ArrayView2<f32>
    ) -> anyhow::Result<(f32, Array2<f32>)> {
        anyhow::bail!("this layer cannot score outputs, a model must end with an OutputSchema layer")
    }

    /// Scores how well a batch of this layer's `outputs` predicts `targets` from
    /// `encode_targets`, one score per output column.  Called on a model's output nodes to
    /// measure accuracy.
    ///
    /// # Parameters
    /// * outputs: A batch this layer produced.
    /// * targets: The expected values of the same rows, from `encode_targets`.
    ///
    /// # Errors
    /// * The layer cannot score outputs, the default.
    /// * The layer has not been prepared or the shapes of `outputs` and `targets` do not match.
    fn evaluate(
        &self,
        _outputs: ArrayView2<f32>,
        _targets: ArrayView2<f32>
    ) -> anyhow::Result<Vec<ColumnScore>> {
        anyhow::bail!("this layer cannot score outputs, a model must end with an OutputSchema layer")
    }

    /// Reads a batch of this layer's `outputs` back into data sheet values, one list per row,
    /// each value paired with the index of the column it predicts.  Called on a model's output
    /// nodes to make predictions.
    ///
    /// # Parameters
    /// * outputs: A batch this layer produced.
    ///
    /// # Errors
    /// * The layer cannot decode outputs, the default.
    /// * The layer has not been prepared or a value cannot be decoded.
    fn decode_outputs(&self, _outputs: ArrayView2<f32>) -> anyhow::Result<Vec<Vec<(usize, DataStorage)>>> {
        anyhow::bail!("this layer cannot decode outputs, a model must end with an OutputSchema layer")
    }
}
