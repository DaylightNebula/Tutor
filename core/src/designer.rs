use ahash::HashMap;
use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{data::{DataSheet, DesignSchema}, layers::Layer, model::{Model, ModelNode}};

/// The network of a model before it is fitted to any data.  It can be saved as json and built
/// into a [`Model`] once sample data is available.
///
/// Nodes may be added in any order and may read from nodes that are not added yet, the network
/// is only checked when it is built.
///
/// Fields:
/// * nodes: Every node of the network keyed by its ID.  Nodes with no inputs read data sheet rows
///   and should be `InputSchema`s, and nodes no other node reads from are outputs and should be
///   `OutputSchema`s.
/// * last_added: The most recently added node, which `add_layer` chains the next layer after.
/// * schema: The named, typed columns the model reads and predicts.  Empty for designs saved
///   before schemas existed.
#[derive(Getters, Default, Serialize, Deserialize)]
pub struct ModelDesigner {
    #[getset(get = "pub")]
    nodes: HashMap<Uuid, ModelNode>,
    #[serde(default)]
    last_added: Option<Uuid>,
    #[getset(get = "pub")]
    #[serde(default)]
    schema: DesignSchema
}

impl ModelDesigner {
    /// Create a new empty `ModelDesigner`.
    pub fn new() -> Self { Self::default() }

    /// Reads a model design from json written by `serialize`.
    ///
    /// # Parameters
    /// * json: The json text of the design.
    ///
    /// # Errors
    /// * `json` is not a valid model design.
    pub fn deserialize(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Writes this model design as json, to be read back by `deserialize`.
    ///
    /// # Errors
    /// * A layer cannot be written as json.
    pub fn serialize(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// Adds a layer that reads from the most recently added node, or an input node if this is the
    /// first one.  A shorthand for designs that are a single chain of layers.
    ///
    /// # Parameters
    /// * layer: The layer to add.
    ///
    /// # Errors
    /// * Never, kept fallible so chained designs can use `?` throughout.
    pub fn add_layer<L: Layer + 'static>(mut self, layer: L) -> anyhow::Result<Self> {
        let inputs = self
            .last_added
            .into_iter()
            .collect::<Vec<_>>();
        self.add_node(layer, inputs);
        Ok(self)
    }

    /// Adds a layer as a new node with a fresh random ID, reading from `inputs`, and gives the
    /// new node's ID.  The inputs are not checked until the design is built.
    ///
    /// # Parameters
    /// * layer: The layer the node runs.
    /// * inputs: The nodes whose outputs are joined side by side, in this order, to make the
    ///   layer's input.  Empty for an input node.
    pub fn add_node<L: Layer + 'static>(&mut self, layer: L, inputs: Vec<Uuid>) -> Uuid {
        let id = Uuid::new_v4();
        self.nodes.insert(id, ModelNode::new(Box::new(layer), inputs));
        self.last_added = Some(id);
        id
    }

    /// Adds a layer as a new node with a given ID, for callers such as an editor that already
    /// track their own IDs.
    ///
    /// # Parameters
    /// * id: The ID of the new node.
    /// * layer: The layer the node runs.
    /// * inputs: The nodes whose outputs are joined side by side, in this order, to make the
    ///   layer's input.  Empty for an input node.
    ///
    /// # Errors
    /// * A node with `id` already exists.
    pub fn insert_node(&mut self, id: Uuid, layer: Box<dyn Layer>, inputs: Vec<Uuid>) -> anyhow::Result<()> {
        anyhow::ensure!(!self.nodes.contains_key(&id), "a node with ID {id} already exists");
        self.nodes.insert(id, ModelNode::new(layer, inputs));
        self.last_added = Some(id);
        Ok(())
    }

    /// Replaces the columns this design reads and predicts.
    ///
    /// # Parameters
    /// * schema: The new input and output columns.
    pub fn set_schema(&mut self, schema: DesignSchema) {
        self.schema = schema;
    }

    /// Builds this design into a `Model` fitted to some sample data.
    ///
    /// Works out the order the nodes run in, prepares each layer in that order, then runs one
    /// sample row end to end so a badly shaped design fails here instead of mid training.
    ///
    /// # Parameters
    /// * sample_data: Rows the model is fitted to and will be trained with.
    ///
    /// # Errors
    /// * `sample_data` is empty.
    /// * The network is empty, reads from a missing node, or has nodes that depend on a cycle.
    /// * A layer cannot be prepared from the sample data.
    /// * The design does not fit the sample data, such as an input node that is not an
    ///   `InputSchema` or layer widths that do not line up.
    pub fn build(self, sample_data: &DataSheet) -> anyhow::Result<Model> {
        let first_row = sample_data
            .iter()
            .next()
            .context("sample data is empty")?;

        let mut model = Model::new(self.nodes).context("model design is not a valid network")?;
        model.prepare_train(sample_data)?;

        // run one row end to end so a badly shaped design fails here instead of mid training
        model
            .predict(first_row)
            .context("model design does not fit the sample data")?;

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    use crate::{data::{DataType, Role, SchemaEntry}, layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights}};

    /// A schema with input `x` (decimal) and output `label` (enum).
    fn schema() -> DesignSchema {
        DesignSchema {
            inputs: vec![SchemaEntry { name: "x".into(), data_type: DataType::Decimal }],
            outputs: vec![SchemaEntry { name: "label".into(), data_type: DataType::Enum }]
        }
    }

    #[test]
    fn schemas_survive_serialization() -> anyhow::Result<()> {
        let mut designer = ModelDesigner::new();
        designer.set_schema(schema());

        let reread = ModelDesigner::deserialize(&designer.serialize()?)?;

        assert_eq!(reread.schema(), &schema());
        Ok(())
    }

    #[test]
    fn designs_without_a_schema_still_load() -> anyhow::Result<()> {
        let designer = ModelDesigner::deserialize(r#"{"nodes": {}}"#)?;

        assert_eq!(designer.schema(), &DesignSchema::default());
        Ok(())
    }

    #[test]
    fn design_schemas_become_data_sheet_schemas() {
        let converted = schema().to_schema();

        let mut columns = converted
            .iter()
            .map(|(name, (data_type, role))| (name.clone(), *data_type, *role))
            .collect::<Vec<_>>();
        columns.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(columns, [
            ("label".to_string(), DataType::Enum, Role::Output),
            ("x".to_string(), DataType::Decimal, Role::Input)
        ]);
    }

    #[test]
    fn training_stops_after_the_epoch_the_stop_flag_is_set_in() -> anyhow::Result<()> {
        let mut designer = ModelDesigner::new()
            .add_layer(InputSchema::new())?
            .add_layer(Weights::random(2, 1e-2))?
            .add_layer(OutputSchema::new())?;
        designer.set_schema(schema());
        let mut csv = String::from("x,label\n");
        for i in 0 .. 20 {
            csv.push_str(&format!("{i},{}\n", if i > 10 { "high" } else { "low" }));
        }
        let sheet = DataSheet::from_csv(designer.schema().to_schema(), &csv)?;
        let mut model = designer.build(&sheet)?;

        let summary = model.train_with_progress(&sheet, 0.05, 4, 1e-4, 50, 1, &AtomicBool::new(true), |_| {})?;

        assert_eq!(summary.epochs, 1);
        assert!(summary.stopped);
        Ok(())
    }
}
