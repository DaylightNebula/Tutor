use std::fmt::Debug;

pub mod scalars;
pub mod softmax;
pub mod weights;

/// Layers are simply something that takes x inputs and turns them into y outputs.
/// These may be anything from soft max for probability outputs, predict via weights,
/// or apply scalars to normalize inputs.
pub trait Layer: Clone + Debug {

    /// The number of inputs this stage may take.  Transform must be given a
    /// input array of this length.
    fn input_size(&self) -> usize;

    /// The number of outputs this stage will produce, useful for validating
    /// that this stage will produce the expected outputs of the model or
    /// expected inputs of the next stage.
    fn output_size(&self) -> usize;

    /// Transform some inputs into some outputs with lengths defined by the
    /// above elements.  They MUST always have lengths matching those lengths.
    fn transform(&self, inputs: &[f32]) -> Box<[f32]>;
}
