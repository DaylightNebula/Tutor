use anyhow::Context;

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
