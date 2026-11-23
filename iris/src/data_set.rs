use anyhow::Context;
use clap::ValueEnum;
use tutor_core::{data::DataSheet, designer::ModelDesigner};

pub mod iris;
pub mod auto_mpg;
pub mod breast_cancer;
pub mod census;

/// The data sets that can be loaded to train and test a model.
///
/// Variants:
/// * Iris: Iris flower measurements classified into 3 species, from `./Iris.csv`.
/// * AutoMpg: Car specs classified by car name, from `./auto-mpg.csv`.
/// * BreastCancer: Tumor measurements classified as benign or malignant, from
///   `./breast_cancer.csv`.
/// * Census: Census details of a person classified by whether their income is over $50K, from
///   `./census.csv`.
#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSet {
    Iris,
    AutoMpg,
    BreastCancer,
    Census
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
            DataSet::BreastCancer => breast_cancer::load_data(),
            DataSet::Census => census::load_data()
        }
    }

    /// Builds an untrained model shaped for this data set.
    ///
    /// Each data set has its own layer stack, sized from tuning runs, starting with an
    /// `InputSchema` to encode the inputs and ending with an `OutputSchema` over the classes.
    /// The schemas are fitted to the data when the design is built.  The last layer gives one
    /// score per output option found in `sample_data`, so it matches the `OutputSchema` built
    /// from the same rows.
    ///
    /// # Parameters
    /// * sample_data: Rows the model will be built and trained with, used to count the output
    ///   options.
    ///
    /// # Errors
    /// * The output columns of `sample_data` cannot be fitted to converters.
    /// * A layer cannot be added to the design.
    pub fn design_model(&self, sample_data: &DataSheet) -> anyhow::Result<ModelDesigner> {
        // one score per option of every output column
        let num_outputs = sample_data
            .build_output_schema()
            .context("failed to count the output options")?
            .iter()
            .map(|(_, converter)| converter.num_weights())
            .sum();

        match self {
            DataSet::Iris => iris::design_model(num_outputs),
            DataSet::AutoMpg => auto_mpg::design_model(num_outputs),
            DataSet::BreastCancer => breast_cancer::design_model(num_outputs),
            DataSet::Census => census::design_model(num_outputs)
        }
    }
}
