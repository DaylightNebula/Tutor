pub mod zero_neg;
pub mod scalars;
pub mod softmax;
pub mod weights;

/// Layers are simply something that takes x inputs and turns them into y outputs.
/// These may be anything from soft max for probability outputs, predict via weights,
/// or apply scalars to normalize inputs.
#[typetag::serde(tag = "type")]
pub trait Layer {

    /// The number of outputs this stage will produce, useful for validating
    /// that this stage will produce the expected outputs of the model or
    /// expected inputs of the next stage.
    fn output_size(&self, previous_layer_num_outputs: usize) -> usize;

    /// Prepare this layer for training.  This may be clearing out old data to
    /// avoid influence of previous runes, establish ranges for scalars, or
    /// anything else that is needed to ensure clean repeatable training runs.
    fn prepare_train(
        &mut self,
        sample_data: &[(Box<[f32]>, usize)],
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
}
