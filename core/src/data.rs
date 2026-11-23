use std::collections::{LinkedList, hash_map::Iter};

use ahash::AHashMap;
use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};

/// All possible data types we can take as inputs or outputs to statistical models.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataType {
    Decimal,
    Integer,
    Enum
}

impl DataType {
    /// Convert this data type and some string data into a storage type, returning an error result on fail.
    pub fn store(&self, str: String) -> anyhow::Result<DataStorage> {
        Ok(match self {
            Self::Decimal => DataStorage::Decimal(str.parse()?),
            Self::Integer => DataStorage::Integer(str.parse()?),
            Self::Enum => DataStorage::Enum(str)
        })
    }

    /// Create an input converter for this data type and some sample data, return an error result on fail.
    /// Sample data used to pull information to reliably convert the input a `DataStorage` to weights.
    pub fn input_converter(&self, sample_data: &[DataStorage]) -> anyhow::Result<DataConverter> {
        anyhow::ensure!(!sample_data.is_empty(), "no sample data to build an input converter from");

        match self {
            Self::Decimal => decimal_converter(sample_data),
            Self::Integer => integer_converter(sample_data),
            Self::Enum => enum_converter(sample_data)
        }
    }
}

/// All possible data types containing there stored data.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum DataStorage {
    Decimal(f32),
    Integer(i32),
    Enum(String)
}

/// Convert some inputs into f32 weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataConverter {
    Decimal { mean: f32, standard_deviation: f32 },
    Integer { mean: f32, standard_deviation: f32 },
    Enum { options: Vec<String> }
}

impl DataConverter {
    pub fn num_weights(&self) -> usize {
        match self {
            Self::Decimal { .. } => 1,
            Self::Integer { .. } => 1,
            Self::Enum { options } => options.len()
        }
    }

    /// Appends the model weights for one value to `out`.
    pub fn encode(&self, value: &DataStorage, out: &mut Vec<f32>) -> anyhow::Result<()> {
        match (self, value) {
            (Self::Decimal { mean, standard_deviation }, DataStorage::Decimal(v)) => {
                out.push((v - mean) / standard_deviation);
            },
            (Self::Integer { mean, standard_deviation }, DataStorage::Integer(v)) => {
                out.push((*v as f32 - mean) / standard_deviation);
            },
            (Self::Enum { options }, DataStorage::Enum(v)) => {
                let position = self.option_index(v)?;
                out.extend((0 .. options.len()).map(|i| if i == position { 1.0 } else { 0.0 }));
            },
            _ => anyhow::bail!("cannot encode {value:?} with a {self:?} converter")
        }

        Ok(())
    }

    /// Reads the model weights of one value, as written by `encode`, back into a value.  Enum
    /// weights give their highest scoring option, and numeric weights are scaled back to the
    /// column's units, with integers rounded to the nearest whole number.
    pub fn decode(&self, encoded: &[f32]) -> anyhow::Result<DataStorage> {
        anyhow::ensure!(
            encoded.len() == self.num_weights(),
            "{self:?} expects {} weights to decode but got {}", self.num_weights(), encoded.len()
        );

        match self {
            Self::Decimal { .. } => Ok(DataStorage::Decimal(self.denormalize(encoded[0])?)),
            Self::Integer { .. } => Ok(DataStorage::Integer(self.denormalize(encoded[0])?.round() as i32)),
            Self::Enum { options } => {
                let best = encoded
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .context("enum converter has no options to decode")?;
                Ok(DataStorage::Enum(options[best].clone()))
            }
        }
    }

    /// Scales a normalized number from `encode` back to the column's units, without rounding.
    /// Enum converters have no units, so they give an error.
    pub fn denormalize(&self, normalized: f32) -> anyhow::Result<f32> {
        match self {
            Self::Decimal { mean, standard_deviation } | Self::Integer { mean, standard_deviation } => {
                Ok(normalized * standard_deviation + mean)
            },
            Self::Enum { .. } => anyhow::bail!("{self:?} is not a numeric converter")
        }
    }

    /// Position of `option` in this enum converter's options.
    pub fn option_index(&self, option: &str) -> anyhow::Result<usize> {
        let Self::Enum { options } = self else { anyhow::bail!("{self:?} is not an enum converter") };
        options
            .iter()
            .position(|o| o == option)
            .with_context(|| format!("{option:?} is not one of the known options {options:?}"))
    }
}

#[derive(Default, Debug, Clone, Getters)]
pub struct DataSheet {
    #[getset(get = "pub")]
    columns: AHashMap<String, usize>,
    #[getset(get = "pub")]
    columns_rev: AHashMap<usize, String>,
    #[getset(get = "pub")]
    schema: Schema,
    data: LinkedList<Vec<DataStorage>>
}

impl DataSheet {

    /// Create a new empty `DataSheet`.
    pub fn new() -> Self { Self::default() }

    /// Get the index of a column with a certain name.
    pub fn column_idx(&self, column: impl Into<String>) -> anyhow::Result<usize> {
        self.columns.get(&column.into()).copied().context("No column with that name")
    }

    /// Get the name of a column from an index number.
    pub fn column_name(&self, idx: usize) -> anyhow::Result<String> {
        self.columns_rev.get(&idx).cloned().context("No column with that index")
    }

    /// Creates an empty sheet with one column per schema entry.
    pub fn from_schema(schema: Schema) -> Self {
        let mut names = schema
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        names.sort();

        // index the columns both ways
        let columns = names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect();
        let columns_rev = names
            .into_iter()
            .enumerate()
            .collect();

        Self { columns, columns_rev, schema, data: LinkedList::new() }
    }

    /// Adds a row to the end of the sheet after checking it against the schema.
    pub fn push(&mut self, data: &[DataStorage]) -> anyhow::Result<()> {
        let types = self.column_types();
        anyhow::ensure!(
            data.len() == types.len(),
            "row has {} values but the schema has {} columns", data.len(), types.len()
        );

        // every value must match its column's type
        for (idx, (value, data_type)) in data.iter().zip(&types).enumerate() {
            anyhow::ensure!(
                storage_matches(*data_type, value),
                "column {} expects a {data_type:?} value but got {value:?}", self.column_name(idx)?
            );
        }

        self.data.push_back(data.to_vec());
        Ok(())
    }

    /// Iterates over every row in the order they were added, each laid out in column index order.
    pub fn iter(&self) -> impl Iterator<Item = &[DataStorage]> {
        self.data
            .iter()
            .map(|row| row.as_slice())
    }

    /// Parses csv text with a header row into a sheet using the given schema.
    pub fn from_csv(schema: Schema, csv_content: &str) -> anyhow::Result<Self> {
        let mut sheet = Self::from_schema(schema);
        let types = sheet.column_types();
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .flexible(true)
            .from_reader(csv_content.as_bytes());

        // find which csv field holds each schema column
        let headers = reader
            .headers()
            .context("failed to read csv header row")?
            .clone();
        let positions = (0 .. types.len())
            .map(|idx| {
                let name = sheet.column_name(idx)?;
                headers
                    .iter()
                    .position(|header| header == name)
                    .with_context(|| format!("csv has no column named {name}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // parse each record into a row in column index order
        for (row_idx, record) in reader.records().enumerate() {
            let record = record.with_context(|| format!("failed to read csv row {}", row_idx + 1))?;
            let row = positions
                .iter()
                .zip(&types)
                .enumerate()
                .map(|(idx, (&position, data_type))| {
                    let field = record
                        .get(position)
                        .with_context(|| format!("csv row {} is missing column {}", row_idx + 1, sheet.column_name(idx).unwrap_or_default()))?;
                    data_type
                        .store(field.to_string())
                        .with_context(|| format!("csv row {}: {field:?} is not a valid {data_type:?}", row_idx + 1))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            sheet.push(&row)?;
        }

        Ok(sheet)
    }

    /// Writes the sheet as csv text with a header row, columns in column index order.
    pub fn to_csv(&self) -> anyhow::Result<String> {
        let mut writer = csv::Writer::from_writer(Vec::new());

        // header row followed by every data row
        let headers = (0 .. self.columns.len())
            .map(|idx| self.column_name(idx))
            .collect::<anyhow::Result<Vec<_>>>()?;
        writer.write_record(&headers)?;
        for row in self.iter() {
            writer.write_record(row.iter().map(storage_to_string))?;
        }

        let bytes = writer
            .into_inner()
            .context("failed to finish writing csv")?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Schema types of every column, in column index order.
    fn column_types(&self) -> Vec<DataType> {
        let mut columns = self.schema
            .iter()
            .collect::<Vec<_>>();
        columns.sort_by(|a, b| a.0.cmp(b.0));
        columns
            .into_iter()
            .map(|(_, (data_type, _))| *data_type)
            .collect()
    }

    /// Builds a converter for every input column, fitted to the rows currently in the sheet.
    ///
    /// Decimal and integer columns learn their mean and standard deviation, and enum columns
    /// learn their options.  Columns are returned in column index order.
    ///
    /// # Errors
    /// * The sheet has no rows to fit the converters to.
    pub fn build_input_schema(&self) -> anyhow::Result<Vec<(String, DataConverter)>> {
        self.build_role_schema(Role::Input)
    }

    /// Builds a converter for every output column, fitted to the rows currently in the sheet.
    ///
    /// Decimal and integer columns learn their mean and standard deviation, and enum columns
    /// learn their options.  Columns are returned in column index order.
    ///
    /// # Errors
    /// * The sheet has no rows to fit the converters to.
    pub fn build_output_schema(&self) -> anyhow::Result<Vec<(String, DataConverter)>> {
        self.build_role_schema(Role::Output)
    }

    /// Fits a converter to every column with the given role, in column index order.
    fn build_role_schema(&self, role: Role) -> anyhow::Result<Vec<(String, DataConverter)>> {
        let mut converters = Vec::new();
        for idx in 0 .. self.columns.len() {
            let name = self.column_name(idx)?;
            let (data_type, column_role) = self.schema.0
                .get(&name)
                .with_context(|| format!("column {name} is not in the schema"))?;
            if *column_role != role { continue; }

            // fit the converter to every value in this column
            let values = self
                .iter()
                .map(|row| row[idx].clone())
                .collect::<Vec<_>>();
            let converter = data_type
                .input_converter(&values)
                .with_context(|| format!("failed to build a converter for column {name}"))?;
            converters.push((name, converter));
        }

        Ok(converters)
    }
}

#[derive(Default, Debug, Clone)]
pub struct Schema(AHashMap<String, (DataType, Role)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Input,
    Output
}

impl Schema {
    pub fn new() -> Self { Self::default() }

    pub fn iter<'a>(&'a self) -> Iter<'a, String, (DataType, Role)> {
        self.0.iter()
    }

    pub fn push(&mut self, name: impl Into<String>, role: Role, data_type: DataType) {
        self.0.insert(name.into(), (data_type, role));
    }
}

/// Whether `value` is stored as the variant that `data_type` describes.
fn storage_matches(data_type: DataType, value: &DataStorage) -> bool {
    matches!(
        (data_type, value),
        (DataType::Decimal, DataStorage::Decimal(_))
            | (DataType::Integer, DataStorage::Integer(_))
            | (DataType::Enum, DataStorage::Enum(_))
    )
}

/// Text form of a stored value as it is written to csv.
fn storage_to_string(value: &DataStorage) -> String {
    match value {
        DataStorage::Decimal(v) => v.to_string(),
        DataStorage::Integer(v) => v.to_string(),
        DataStorage::Enum(v) => v.clone()
    }
}

/// Fits a decimal converter to the mean and standard deviation of the sample.
fn decimal_converter(sample_data: &[DataStorage]) -> anyhow::Result<DataConverter> {
    let values = sample_data
        .iter()
        .enumerate()
        .map(|(i, value)| match value {
            DataStorage::Decimal(v) => Ok(*v),
            _ => anyhow::bail!("sample value {i} is not a decimal")
        })
        .collect::<anyhow::Result<Vec<f32>>>()?;

    let (mean, standard_deviation) = mean_and_deviation(&values);
    Ok(DataConverter::Decimal { mean, standard_deviation })
}

/// Fits an integer converter to the mean and standard deviation of the sample.
fn integer_converter(sample_data: &[DataStorage]) -> anyhow::Result<DataConverter> {
    let values = sample_data
        .iter()
        .enumerate()
        .map(|(i, value)| match value {
            DataStorage::Integer(v) => Ok(*v as f32),
            _ => anyhow::bail!("sample value {i} is not an integer")
        })
        .collect::<anyhow::Result<Vec<f32>>>()?;

    let (mean, standard_deviation) = mean_and_deviation(&values);
    Ok(DataConverter::Integer { mean, standard_deviation })
}

/// Collects the sorted, distinct options found in the sample.
fn enum_converter(sample_data: &[DataStorage]) -> anyhow::Result<DataConverter> {
    let mut options = sample_data
        .iter()
        .enumerate()
        .map(|(i, value)| match value {
            DataStorage::Enum(v) => Ok(v.clone()),
            _ => anyhow::bail!("sample value {i} is not an enum option")
        })
        .collect::<anyhow::Result<Vec<String>>>()?;

    options.sort();
    options.dedup();
    Ok(DataConverter::Enum { options })
}

/// Population mean and standard deviation of `values`, which must not be empty.  A column with no
/// spread gets a standard deviation of 1 so normalizing it never divides by zero.
fn mean_and_deviation(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>() / n;

    let standard_deviation = variance.sqrt();
    if standard_deviation > 0.0 {
        (mean, standard_deviation)
    } else {
        (mean, 1.0)
    }
}
