use std::collections::LinkedList;

use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};

use crate::{data::DataSheet, layers::Layer, model::Model};

/// The layer stack of a model before it is fitted to any data.  It can be saved as json and built
/// into a [`Model`] once sample data is available.
///
/// Fields:
/// * layers: The layers in the order data flows through them, starting with an `InputSchema` and
///   ending with an `OutputSchema`.
#[derive(Getters, Default, Serialize, Deserialize)]
pub struct ModelDesigner {
    #[getset(get = "pub")]
    layers: LinkedList<Box<dyn Layer>>
}

impl ModelDesigner {
    /// Create a new empty `ModelDesigner`.
    pub fn new() -> Self { Self::default() }

    /// Deserialize a json string to a model design.
    pub fn deserialize(json: &str) -> anyhow::Result<Self> {
        return Ok(serde_json::from_str(json)?);
    }

    /// Serialize this model design to a string.
    pub fn serialize(&self) -> anyhow::Result<String> {
        return Ok(serde_json::to_string(&self)?);
    }

    /// Add a new layer to this model, this adds to the end of the layer stack.
    pub fn add_layer<L: Layer + 'static>(mut self, layer: L) -> anyhow::Result<Self> {
        self.layers.push_back(Box::new(layer));
        return Ok(self);
    }

    /// Builds this design into a `Model` fitted to some sample data.
    pub fn build(mut self, sample_data: &DataSheet) -> anyhow::Result<Model> {
        let first_row = sample_data
            .iter()
            .next()
            .context("sample data is empty")?;

        // prepare each layer with the width produced by the layer before it
        let mut width = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .prepare_train(sample_data, width)
                .with_context(|| format!("failed to prepare layer {i}"))?;
            width = layer.output_size(width);
        }

        // run one row end to end so a badly shaped design fails here instead of mid training
        let model = Model::new(self.layers);
        model
            .predict(first_row)
            .context("model design does not fit the sample data")?;

        Ok(model)
    }
}
