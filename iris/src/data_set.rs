use clap::ValueEnum;
use tutor_core::designer::ModelDesigner;

pub mod iris;
pub mod auto_mpg;
pub mod breast_cancer;

/// The data sets that can be loaded to train and test a model.
///
/// Variants:
/// * Iris: Iris flower measurements classified into 3 species, from `./Iris.csv`.
/// * AutoMpg: Car specs classified into 3 fuel efficiency bands, from `./auto-mpg.csv`.
/// * BreastCancer: Tumor measurements classified as benign or malignant, from
///   `./breast_cancer.csv`.
#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSet {
    Iris,
    AutoMpg,
    BreastCancer
}


impl DataSet {
    /// Loads every row of this data set.
    ///
    /// Each row is its input features paired with the index of its class.
    ///
    /// # Errors
    /// * The data set's csv file cannot be opened or parsed.
    pub fn load_data(&self) -> anyhow::Result<Vec<(Box<[f32]>, usize)>> {
        match self {
            DataSet::Iris => iris::load_data(),
            DataSet::AutoMpg => auto_mpg::load_data(),
            DataSet::BreastCancer => breast_cancer::load_data()
        }
    }

    /// Builds an untrained model shaped for this data set.
    ///
    /// Each data set has its own layer stack, sized from tuning runs, starting with `scalars` to
    /// normalize the inputs and ending with a softmax over the classes.
    ///
    /// # Parameters
    /// * scalars: Input normalization fitted to the training rows, used as the first layer.
    /// * num_inputs: Number of input features in each row.
    /// * num_outputs: Number of classes the model chooses between.
    ///
    /// # Errors
    /// * `scalars` does not take `num_inputs` inputs, so the layers do not line up.
    pub fn design_model(
        &self,
        num_inputs: usize,
        num_outputs: usize
    ) -> anyhow::Result<ModelDesigner> {
        // todo give scalars a build step so we dont need to pass them through here
        match self {
            DataSet::Iris => iris::design_model(num_inputs, num_outputs),
            DataSet::AutoMpg => auto_mpg::design_model(num_inputs, num_outputs),
            DataSet::BreastCancer => breast_cancer::design_model(num_inputs, num_outputs)
        }
    }

    /// The number of classes in this data set, and so the number of outputs a model needs.
    pub fn num_classes(&self) -> usize {
        match self {
            DataSet::Iris => 3,
            DataSet::AutoMpg => auto_mpg::NUM_CLASSES,
            DataSet::BreastCancer => breast_cancer::NUM_CLASSES
        }
    }
}
