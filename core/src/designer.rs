use ahash::HashMap;
use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{data::DataSheet, layers::Layer, model::{Model, ModelNode}};

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
#[derive(Getters, Default, Serialize, Deserialize)]
pub struct ModelDesigner {
    #[getset(get = "pub")]
    nodes: HashMap<Uuid, ModelNode>,
    #[serde(default)]
    last_added: Option<Uuid>
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
