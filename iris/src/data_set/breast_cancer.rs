use anyhow::Context;
use tutor_core::{designer::ModelDesigner, layers::{scalars::Scalars, softmax::SoftMax, weights::Weights, zero_neg::ZeroNegative}, model::Model};

/// The diagnosis of a tumor in the breast cancer data set.
///
/// Variants:
/// * Benign: Non-cancerous tumor, written as `B` in the csv.
/// * Malignant: Cancerous tumor, written as `M` in the csv.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Diagnosis {
    Benign = 0,
    Malignant = 1
}

/// Number of variants in `Diagnosis`, and so the number of outputs a model needs for this set.
pub const NUM_CLASSES: usize = 2;

/// Number of measurement columns that follow the `id` and `diagnosis` columns.
pub const NUM_FEATURES: usize = 30;

impl Diagnosis {
    /// Parses a diagnosis from its csv code.
    ///
    /// # Parameters
    /// * code: The `diagnosis` column value, either `B` or `M`.
    ///
    /// # Errors
    /// * The code is neither `B` nor `M`.
    pub fn from_code(code: &str) -> anyhow::Result<Self> {
        match code {
            "B" => Ok(Diagnosis::Benign),
            "M" => Ok(Diagnosis::Malignant),
            other => anyhow::bail!("unknown diagnosis code {other:?}")
        }
    }
}

/// Loads the breast cancer data set from `./breast_cancer.csv`.
///
/// Each row becomes its 30 measurements paired with the index of its `Diagnosis`.  The `id`
/// column is dropped.  The rows are read by position rather than through serde, as the csv header
/// has a trailing comma that gives it one more column than the rows.
///
/// # Errors
/// * The csv file cannot be opened or a row cannot be read.
/// * A row has an unknown diagnosis or a measurement that is not a number.
/// * A row has fewer than 30 measurements.
pub(crate) fn load_data() -> anyhow::Result<Vec<(Box<[f32]>, usize)>> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path("./breast_cancer.csv")
        .context("failed to open ./breast_cancer.csv")?;

    let mut rows = Vec::new();
    for (line, record) in csv_reader.records().enumerate() {
        let record = record.with_context(|| format!("failed to read breast_cancer.csv row {line}"))?;

        // column 1 is the diagnosis, columns 2.. are the measurements
        let diagnosis = Diagnosis::from_code(record.get(1).unwrap_or_default())
            .with_context(|| format!("bad diagnosis on breast_cancer.csv row {line}"))?;
        let features = record
            .iter()
            .skip(2)
            .take(NUM_FEATURES)
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("bad measurement on breast_cancer.csv row {line}"))?;
        anyhow::ensure!(
            features.len() == NUM_FEATURES,
            "breast_cancer.csv row {line} has {} measurements, expected {NUM_FEATURES}",
            features.len()
        );

        rows.push((features.into_boxed_slice(), diagnosis as usize));
    }

    Ok(rows)
}

/// Width of the hidden layer in the model built by `create_model`.
const HIDDEN_WIDTH: usize = 32;

/// Builds an untrained model shaped for this data set.
///
/// Uses a single 32 wide ReLU hidden layer.  The diagnoses are close to linearly separable, so a
/// linear model is nearly as good, but a 32 wide hidden layer gave a small consistent gain that
/// stopped growing past that width.  Deeper stacks did not help.
///
/// # Parameters
/// * scalars: Input normalization fitted to the training rows, used as the first layer.
/// * num_inputs: Number of input features in each row.
/// * num_outputs: Number of classes the model chooses between.
///
/// # Errors
/// * `scalars` does not take `num_inputs` inputs, so the layers do not line up.
pub(crate) fn design_model(
    num_inputs: usize,
    num_outputs: usize
) -> anyhow::Result<ModelDesigner> {
    Model::designer(num_inputs, num_outputs)
        .add_layer(Scalars::new(num_inputs))?
        .add_layer(Weights::random(num_inputs, HIDDEN_WIDTH, 1e-2))?
        .add_layer(ZeroNegative::new(HIDDEN_WIDTH))?
        .add_layer(Weights::random(HIDDEN_WIDTH, num_outputs, 1e-2))?
        .add_layer(SoftMax::new(num_outputs))
}
