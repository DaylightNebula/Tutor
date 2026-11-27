use std::collections::{LinkedList, hash_map::Iter};

use ahash::AHashMap;
use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};

/// All possible data types we can take as inputs or outputs to statistical models.
///
/// Variants:
/// * Decimal: A floating point number, stored as an `f32`.
/// * Integer: A whole number, stored as an `i32`.
/// * Enum: One option out of a fixed set of text options.
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DataType {
    #[default]
    Decimal,
    Integer,
    Enum
}

impl DataType {
    /// Converts some text into a stored value of this data type.
    ///
    /// Decimal and integer text is parsed into a number, and enum text is stored as is.
    ///
    /// # Parameters
    /// * str: The text to convert, such as one csv field.
    ///
    /// # Errors
    /// * The text is not a valid number for a decimal or integer data type.
    pub fn store(&self, str: String) -> anyhow::Result<DataStorage> {
        Ok(match self {
            Self::Decimal => DataStorage::Decimal(str.parse()?),
            Self::Integer => DataStorage::Integer(str.parse()?),
            Self::Enum => DataStorage::Enum(str)
        })
    }

    /// Creates a converter that turns values of this data type into model weights, fitted to
    /// some sample data.
    ///
    /// Decimal and integer converters learn the mean and standard deviation of the sample, and
    /// enum converters learn the sorted, distinct options found in it.
    ///
    /// # Parameters
    /// * sample_data: Values of this data type to fit the converter to, usually every value of
    ///   one data sheet column.
    ///
    /// # Errors
    /// * `sample_data` is empty.
    /// * A value in `sample_data` is not stored as this data type.
    pub fn input_converter(&self, sample_data: &[DataStorage]) -> anyhow::Result<DataConverter> {
        anyhow::ensure!(!sample_data.is_empty(), "no sample data to build an input converter from");

        match self {
            Self::Decimal => decimal_converter(sample_data),
            Self::Integer => integer_converter(sample_data),
            Self::Enum => enum_converter(sample_data)
        }
    }
}

/// A single value of one of the [`DataType`]s, as stored in a data sheet row.
///
/// Variants:
/// * Decimal: A floating point number.
/// * Integer: A whole number.
/// * Enum: The text of one enum option.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum DataStorage {
    Decimal(f32),
    Integer(i32),
    Enum(String)
}

/// Converts values of one data sheet column to and from the `f32` weights a model works with.
/// Fitted to a column's sample values by `DataType::input_converter`.
///
/// Variants:
/// * Decimal: Normalizes a decimal to `(value − mean) / standard_deviation`, one weight.
/// * Integer: Normalizes an integer the same way as a decimal, one weight.
/// * Enum: One-hot encodes an option as one weight per entry of `options`, 1 for the matching
///   option and 0 for every other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataConverter {
    Decimal { mean: f32, standard_deviation: f32 },
    Integer { mean: f32, standard_deviation: f32 },
    Enum { options: Vec<String> }
}

impl DataConverter {
    /// The number of weights one value takes up once encoded: 1 for numeric converters and one
    /// per option for enum converters.
    pub fn num_weights(&self) -> usize {
        match self {
            Self::Decimal { .. } => 1,
            Self::Integer { .. } => 1,
            Self::Enum { options } => options.len()
        }
    }

    /// Appends the model weights for one value to `out`.
    ///
    /// Numeric values are normalized with the converter's mean and standard deviation, and enum
    /// values are one-hot encoded.  `num_weights` weights are appended in total.
    ///
    /// # Parameters
    /// * value: The value to encode.
    /// * out: Buffer the weights are appended to.
    ///
    /// # Errors
    /// * `value` is not stored as the data type this converter was made for.
    /// * `value` is an enum option this converter does not know.
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

    /// Reads the model weights of one value, as written by `encode`, back into a value.
    ///
    /// Enum weights give their highest scoring option, and numeric weights are scaled back to the
    /// column's units, with integers rounded to the nearest whole number.
    ///
    /// # Parameters
    /// * encoded: The weights of one value, `num_weights` long.
    ///
    /// # Errors
    /// * `encoded` is not `num_weights` long.
    /// * An enum converter has no options to pick from.
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
    ///
    /// # Parameters
    /// * normalized: A weight as written by `encode` or predicted by a model.
    ///
    /// # Errors
    /// * This is an enum converter, which has no units to scale to.
    pub fn denormalize(&self, normalized: f32) -> anyhow::Result<f32> {
        match self {
            Self::Decimal { mean, standard_deviation } | Self::Integer { mean, standard_deviation } => {
                Ok(normalized * standard_deviation + mean)
            },
            Self::Enum { .. } => anyhow::bail!("{self:?} is not a numeric converter")
        }
    }

    /// Gives the position of an option in this enum converter's options, which is also the
    /// position of its 1 in the option's one-hot weights.
    ///
    /// # Parameters
    /// * option: The text of the option to find.
    ///
    /// # Errors
    /// * This is not an enum converter.
    /// * `option` is not one of this converter's options.
    pub fn option_index(&self, option: &str) -> anyhow::Result<usize> {
        let Self::Enum { options } = self else { anyhow::bail!("{self:?} is not an enum converter") };
        options
            .iter()
            .position(|o| o == option)
            .with_context(|| format!("{option:?} is not one of the known options {options:?}"))
    }
}

/// A table of rows that models are built, trained, and scored with, where every column has a
/// name, a [`DataType`], and a [`Role`] from the sheet's [`Schema`].
///
/// Columns are indexed by their names in sorted order, and every row is laid out in that order.
///
/// Fields:
/// * columns: The index of every column, keyed by column name.
/// * columns_rev: The name of every column, keyed by column index.
/// * schema: The data type and role of every column.
/// * data: Every row in the order they were added, each laid out in column index order.
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

    /// Gets the index of a column with a certain name, its position in every row.
    ///
    /// # Parameters
    /// * column: The name of the column.
    ///
    /// # Errors
    /// * The sheet has no column with that name.
    pub fn column_idx(&self, column: impl Into<String>) -> anyhow::Result<usize> {
        self.columns.get(&column.into()).copied().context("No column with that name")
    }

    /// Gets the name of the column at a position in every row.
    ///
    /// # Parameters
    /// * idx: The column index.
    ///
    /// # Errors
    /// * The sheet has no column at that index.
    pub fn column_name(&self, idx: usize) -> anyhow::Result<String> {
        self.columns_rev.get(&idx).cloned().context("No column with that index")
    }

    /// Creates an empty sheet with one column per schema entry, indexed in name order.
    ///
    /// # Parameters
    /// * schema: The name, data type, and role of every column.
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
    ///
    /// # Parameters
    /// * data: The row's values, one per column in column index order.
    ///
    /// # Errors
    /// * The row does not have one value per column.
    /// * A value is not stored as its column's data type.
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
    ///
    /// Each schema column is read from the csv column with the same header, and any other csv
    /// columns are ignored.  Fields are trimmed of surrounding whitespace.
    ///
    /// # Parameters
    /// * schema: The name, data type, and role of every column to read.
    /// * csv_content: The csv text, starting with a header row.
    ///
    /// # Errors
    /// * The header row cannot be read.
    /// * The csv has no column for one of the schema's columns.
    /// * A row cannot be read or is missing a field.
    /// * A field is not a valid value of its column's data type.
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
    ///
    /// # Errors
    /// * A row cannot be written as csv.
    /// * The written csv is not valid UTF-8.
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
    /// * A column is missing from the schema.
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
    /// * A column is missing from the schema.
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

/// The columns of a [`DataSheet`], each with a [`DataType`] and a [`Role`], keyed by column
/// name.
#[derive(Default, Debug, Clone)]
pub struct Schema(AHashMap<String, (DataType, Role)>);

/// What a model does with a data sheet column.
///
/// Variants:
/// * Input: The model reads the column to make its predictions.
/// * Output: The model predicts the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Input,
    Output
}

impl Schema {
    /// Creates an empty schema with no columns.
    pub fn new() -> Self { Self::default() }

    /// Iterates over every column's name, data type, and role, in no particular order.
    pub fn iter<'a>(&'a self) -> Iter<'a, String, (DataType, Role)> {
        self.0.iter()
    }

    /// Adds a column to the schema, replacing any column with the same name.
    ///
    /// # Parameters
    /// * name: The column's name, matched against csv headers when reading csv.
    /// * role: Whether models read or predict the column.
    /// * data_type: The type of every value in the column.
    pub fn push(&mut self, name: impl Into<String>, role: Role, data_type: DataType) {
        self.0.insert(name.into(), (data_type, role));
    }
}

/// One named column of a [`DesignSchema`].
///
/// Fields:
/// * name: Name of the column, matched against csv headers.
/// * data_type: Type of every value in the column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaEntry {
    pub name: String,
    pub data_type: DataType
}

/// The columns a model design reads and predicts, kept in the order they were listed.  Saved
/// with a design, and turned into a data sheet [`Schema`] with `to_schema`.
///
/// Fields:
/// * inputs: Columns the model reads to make its predictions.
/// * outputs: Columns the model predicts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignSchema {
    pub inputs: Vec<SchemaEntry>,
    pub outputs: Vec<SchemaEntry>
}

impl DesignSchema {
    /// Converts this design schema into a data sheet [`Schema`], giving inputs the
    /// [`Role::Input`] role and outputs the [`Role::Output`] role.  A name used more than once
    /// keeps its last entry, outputs coming after inputs.
    pub fn to_schema(&self) -> Schema {
        let mut schema = Schema::new();
        for entry in &self.inputs {
            schema.push(entry.name.clone(), Role::Input, entry.data_type);
        }
        for entry in &self.outputs {
            schema.push(entry.name.clone(), Role::Output, entry.data_type);
        }
        schema
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
    // population mean and variance
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>() / n;

    // a column with no spread keeps a standard deviation of 1
    let standard_deviation = variance.sqrt();
    if standard_deviation > 0.0 {
        (mean, standard_deviation)
    } else {
        (mean, 1.0)
    }
}
