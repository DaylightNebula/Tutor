use clap::ValueEnum;

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

    /// The number of classes in this data set, and so the number of outputs a model needs.
    pub fn num_classes(&self) -> usize {
        match self {
            DataSet::Iris => 3,
            DataSet::AutoMpg => auto_mpg::NUM_CLASSES,
            DataSet::BreastCancer => breast_cancer::NUM_CLASSES
        }
    }
}
