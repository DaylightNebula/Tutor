use std::collections::LinkedList;

use getset::{CopyGetters, Getters};
use serde::{Deserialize, Serialize};

use crate::{layers::Layer, model::Model};

#[derive(Getters, CopyGetters, Serialize, Deserialize)]
pub struct ModelDesigner {
    #[getset(get = "pub")]
    layers: LinkedList<Box<dyn Layer>>,
    #[getset(get_copy = "pub")]
    num_inputs: usize,
    #[getset(get_copy = "pub")]
    num_outputs: usize,
    #[getset(get_copy = "pub")]
    cur_width: usize
}

impl ModelDesigner {
    /// Create a new `ModelBuilder` instance.
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            layers: LinkedList::new(),
            num_inputs, num_outputs,
            cur_width: num_inputs
        }
    }

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
        self.cur_width = layer.output_size(self.cur_width);
        self.layers.push_back(Box::new(layer));
        return Ok(self);
    }

    /// Build this `ModelBuilder` into a `Model`, requires some sample data so that layers
    /// can prepare themselves, useful for doing things like finding values for scalar layers.
    pub fn build(mut self, sample_data: &[(Box<[f32]>, usize)]) -> anyhow::Result<Model> {
        if sample_data.len() < 1 { anyhow::bail!("Sample data is empty") }

        // prepare each layer with the width produced by the layer before it
        let mut width = self.num_inputs;
        for layer in &mut self.layers {
            layer.prepare_train(sample_data, width)?;
            width = layer.output_size(width);
        }

        Ok(Model::new(self.layers, self.num_inputs, self.num_outputs))
    }
}
