use clap::ValueEnum;
use tutor_core::{data::DataSheet, designer::ModelDesigner};

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
    /// Loads every row of this data set into a data sheet, with its features as input columns
    /// and its class as an output column.
    ///
    /// # Errors
    /// * The data set's csv file cannot be opened or parsed.
    pub fn load_data(&self) -> anyhow::Result<DataSheet> {
        match self {
            DataSet::Iris => iris::load_data(),
            DataSet::AutoMpg => auto_mpg::load_data(),
            DataSet::BreastCancer => breast_cancer::load_data()
        }
    }

    /// Builds an untrained model shaped for this data set.
    ///
    /// Each data set has its own layer stack, sized from tuning runs, starting with an
    /// `InputSchema` to encode the inputs and ending with an `OutputSchema` over the classes.
    /// The schemas are fitted to the data when the design is built.
    ///
    /// # Errors
    /// * A layer cannot be added to the design.
    pub fn design_model(&self) -> anyhow::Result<ModelDesigner> {
        let num_outputs = self.num_classes();
        match self {
            DataSet::Iris => iris::design_model(num_outputs),
            DataSet::AutoMpg => auto_mpg::design_model(num_outputs),
            DataSet::BreastCancer => breast_cancer::design_model(num_outputs)
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
