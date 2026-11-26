use std::sync::Arc;

use deltos::theme::Theme;
use deltos_node_graphs::{editor::property_field, tiles::{IOButton, NodeGraphTile, Wire, WireRules, Wires}};
use gearbox::{anyhow::{self, Context}, glam::{Vec2, Vec4}, ui::{Node, SDFFont, UiBundle, px}};
use serde_json::Value;
use tutor_core::layers::Layer;
use uuid::Uuid;

pub const OUTPUT: Uuid = Uuid::from_u128(1);
pub const INPUT_SLOTS: usize = 1;
pub const INPUT_LAYER: &str = "InputSchema";
pub const OUTPUT_LAYER: &str = "OutputSchema";
pub const SLOT_ID_OFFSET: u128 = 0x100;

/// The ID of the input slot at `index`.  Slots are joined in index order to make the layer's
/// input, so the order the wires feed a node is kept.
pub fn input_slot(index: usize) -> Uuid {
    Uuid::from_u128(SLOT_ID_OFFSET + index as u128)
}

/// The index of the input slot with `id`, or `None` if `id` isn't an input slot.
pub fn slot_index(id: Uuid) -> Option<usize> {
    id
        .as_u128()
        .checked_sub(SLOT_ID_OFFSET)
        .and_then(|index| usize::try_from(index).ok())
}

/// The typetag name a layer is saved under, such as `Weights`.
///
/// # Parameters
/// * layer: The layer to name.
///
/// # Errors
/// * The layer can't be written as json, or is written without a `type`.
pub fn layer_name(layer: &dyn Layer) -> anyhow::Result<String> {
    let value = serde_json::to_value(layer)?;
    value
        .get("type")
        .and_then(Value::as_str)
        .map(String::from)
        .context("layer was saved without a type")
}

/// Copies a layer by writing it as json and reading it back, since layers can't be cloned.
/// Anything a layer doesn't save, such as running training totals, isn't copied.
///
/// # Parameters
/// * layer: The layer to copy.
///
/// # Errors
/// * The layer can't be written as json or read back.
pub fn copy_layer(layer: &dyn Layer) -> anyhow::Result<Box<dyn Layer>> {
    let value = serde_json::to_value(layer)?;
    Ok(serde_json::from_value(value)?)
}

/// A tile drawing one node of a model design: its layer, the input slots its inputs are joined
/// from, and the wires out of its output.  Output schema tiles end the model, so they have no
/// output.
///
/// Fields:
/// * tile_id: ID of the tile, which is the ID of its node in the design.
/// * name: The layer's typetag name, shown as the tile's title.
/// * layer: The layer the node runs.
/// * position: Where the tile is drawn, in logical pixels from the graph's top left.
/// * slots: Number of input slots shown.
/// * wires: The wires out of the output, each to an input slot of the node it feeds.
#[derive(Debug)]
pub struct LayerTile {
    tile_id: Uuid,
    name: String,
    layer: Box<dyn Layer>,
    position: Vec2,
    slots: usize,
    wires: Vec<Wire>
}

impl LayerTile {
    /// Creates an unconnected tile for a node.
    ///
    /// Input layers get only as many slots as `inputs`, every other layer gets at least
    /// `INPUT_SLOTS`.
    ///
    /// # Parameters
    /// * tile_id: ID of the node the tile draws.
    /// * layer: The layer the node runs.
    /// * position: Where the tile is drawn, in logical pixels from the graph's top left.
    /// * inputs: Number of nodes the node already reads from.
    ///
    /// # Errors
    /// * The layer's name can't be found, see `layer_name`.
    pub fn new(tile_id: Uuid, layer: Box<dyn Layer>, position: Vec2, inputs: usize) -> anyhow::Result<Self> {
        let name = layer_name(&*layer)?;
        let slots = if name == INPUT_LAYER { inputs } else { inputs.max(INPUT_SLOTS) };
        Ok(Self { tile_id, name, layer, position, slots, wires: Vec::new() })
    }

    /// The layer the tile's node runs.
    pub fn layer(&self) -> &dyn Layer { &*self.layer }

    /// The wires out of the tile's output.
    pub fn wires(&self) -> &[Wire] { &self.wires }

    /// The settings of the layer that can be typed in: every number, string, and true or false
    /// value at the top of its saved json, keyed by field name.
    fn editable_fields(&self) -> anyhow::Result<serde_json::Map<String, Value>> {
        let Value::Object(fields) = serde_json::to_value(&*self.layer)? else {
            anyhow::bail!("{} wasn't saved as an object", self.name);
        };
        Ok(fields
            .into_iter()
            .filter(|(name, value)| name != "type" && (value.is_number() || value.is_string() || value.is_boolean()))
            .collect())
    }

    /// The layer's editable settings as (name, value) text pairs, in name order.  A layer that
    /// can't be written has none, so its tile still draws.
    fn property_values(&self) -> Vec<(String, String)> {
        let Ok(fields) = self.editable_fields() else { return Vec::new() };
        fields
            .into_iter()
            .map(|(name, value)| match value {
                Value::String(text) => (name, text),
                other => (name, other.to_string())
            })
            .collect()
    }
}

impl NodeGraphTile for LayerTile {
    fn tile_id(&self) -> Uuid { self.tile_id }
    fn title(&self) -> String { self.name.clone() }
    fn position(&self) -> Vec2 { self.position }

    fn color(&self, theme: &Theme) -> Vec4 {
        match self.name.as_str() {
            INPUT_LAYER => theme.blue,
            OUTPUT_LAYER => theme.green,
            _ => theme.orange
        }
    }

    fn set_position(&mut self, position: Vec2) {
        self.position = position;
    }

    fn connect(&mut self, output_button_id: Uuid, input_tile_id: Uuid, input_button_id: Uuid) -> anyhow::Result<()> {
        anyhow::ensure!(self.name != OUTPUT_LAYER, "{OUTPUT_LAYER} ends the model, so nothing can read from it");
        anyhow::ensure!(output_button_id == OUTPUT, "No output with ID {output_button_id}");
        anyhow::ensure!(input_tile_id != self.tile_id, "A layer can't read from itself");
        anyhow::ensure!(slot_index(input_button_id).is_some(), "No input slot with ID {input_button_id}");
        self.wires.push(Wire::new(input_tile_id, input_button_id));
        Ok(())
    }

    fn disconnect(&mut self, output_button_id: Uuid, input_tile_id: Uuid, input_button_id: Uuid) -> anyhow::Result<()> {
        anyhow::ensure!(output_button_id == OUTPUT, "No output with ID {output_button_id}");
        let wire = Wire::new(input_tile_id, input_button_id);
        let index = self.wires
            .iter()
            .position(|existing| *existing == wire)
            .with_context(|| format!("{} isn't connected to input {input_button_id}", self.name))?;
        self.wires.remove(index);
        Ok(())
    }

    fn inputs(&self, theme: &Theme) -> Vec<IOButton> {
        // a lone slot is just "In", several are numbered in the order they're joined
        let label = |index: usize| match self.slots {
            1 => "In".to_string(),
            _ => format!("In {}", index + 1)
        };
        (0..self.slots)
            .map(|index| IOButton::new(input_slot(index), label(index), theme.blue, WireRules::Single))
            .collect()
    }

    fn outputs(&self, theme: &Theme) -> Vec<(IOButton, Wires)> {
        if self.name == OUTPUT_LAYER { return Vec::new() }
        vec![(IOButton::new(OUTPUT, "Out", theme.purple, WireRules::Multi), Wires::new(self.wires.clone()))]
    }

    fn body(&self, theme: &Theme, font: &Arc<SDFFont>) -> Option<UiBundle> {
        let fields = self
            .property_values()
            .into_iter()
            .map(|(name, value)| property_field(theme, font, self.tile_id, &name, &value))
            .collect::<Vec<_>>();
        if fields.is_empty() { return None }
        Some(UiBundle::new(Node::column().gap(px(6.0))).children(fields))
    }

    fn set_property(&mut self, name: &str, value: &str) -> anyhow::Result<()> {
        // read the text as the same kind of value the field holds
        let current = self
            .editable_fields()?
            .remove(name)
            .with_context(|| format!("{} has no property {name}", self.name))?;
        let parsed = match current {
            Value::String(_) => Value::String(value.into()),
            Value::Bool(_) => Value::Bool(value
                .trim()
                .parse()
                .with_context(|| format!("{name} must be true or false"))?),
            _ => {
                let number: Value = serde_json::from_str(value.trim())
                    .with_context(|| format!("{name} must be a number"))?;
                anyhow::ensure!(number.is_number(), "{name} must be a number");
                number
            }
        };

        // rebuild the layer with the new value, which checks the value fits the field
        let mut saved = serde_json::to_value(&*self.layer)?;
        let Some(fields) = saved.as_object_mut() else {
            anyhow::bail!("{} wasn't saved as an object", self.name);
        };
        fields.insert(name.into(), parsed);
        self.layer = serde_json::from_value(saved)
            .with_context(|| format!("{value} isn't a valid {name}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tutor_core::layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative};

    /// A weights tile with 2 outputs and no weight decay.
    fn weights_tile() -> anyhow::Result<LayerTile> {
        LayerTile::new(Uuid::new_v4(), Box::new(Weights::random(2, 0.0)), Vec2::ZERO, 0)
    }

    /// The value shown for the property `name` of `tile`.
    fn property(tile: &LayerTile, name: &str) -> Option<String> {
        tile
            .property_values()
            .into_iter()
            .find(|(property, _)| property == name)
            .map(|(_, value)| value)
    }

    #[test]
    fn slot_ids_round_trip_and_skip_the_output() {
        assert_eq!(slot_index(input_slot(0)), Some(0));
        assert_eq!(slot_index(input_slot(7)), Some(7));
        assert_eq!(slot_index(OUTPUT), None);
    }

    #[test]
    fn input_layers_get_no_slots_and_others_get_at_least_the_default() -> anyhow::Result<()> {
        let theme = Theme::default();
        let input = LayerTile::new(Uuid::new_v4(), Box::new(InputSchema::new()), Vec2::ZERO, 0)?;
        let wide = LayerTile::new(Uuid::new_v4(), Box::new(Weights::random(1, 0.0)), Vec2::ZERO, 6)?;

        assert_eq!(input.title(), INPUT_LAYER);
        assert!(input.inputs(&theme).is_empty());
        assert_eq!(weights_tile()?.inputs(&theme).len(), INPUT_SLOTS);
        assert_eq!(wide.inputs(&theme).len(), 6);
        Ok(())
    }

    #[test]
    fn single_slots_are_labelled_in_and_several_are_numbered() -> anyhow::Result<()> {
        let theme = Theme::default();
        let wide = LayerTile::new(Uuid::new_v4(), Box::new(Weights::random(1, 0.0)), Vec2::ZERO, 2)?;

        let labels = |tile: &LayerTile| tile
            .inputs(&theme)
            .iter()
            .map(|input| input.name().clone())
            .collect::<Vec<_>>();
        assert_eq!(labels(&weights_tile()?), ["In"]);
        assert_eq!(labels(&wide), ["In 1", "In 2"]);
        Ok(())
    }

    #[test]
    fn output_schemas_take_an_input_but_have_no_output() -> anyhow::Result<()> {
        let theme = Theme::default();
        let mut output = LayerTile::new(Uuid::new_v4(), Box::new(OutputSchema::new()), Vec2::ZERO, 0)?;

        assert_eq!(output.inputs(&theme).len(), 1);
        assert!(output.outputs(&theme).is_empty());
        assert!(output.connect(OUTPUT, Uuid::new_v4(), input_slot(0)).is_err());
        assert!(output.wires().is_empty());
        Ok(())
    }

    #[test]
    fn scalar_settings_are_properties() -> anyhow::Result<()> {
        let tile = weights_tile()?;

        let names = tile
            .property_values()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["depth", "weight_decay"]);
        assert_eq!(property(&tile, "depth").as_deref(), Some("2"));
        Ok(())
    }

    #[test]
    fn only_layers_with_settings_get_a_body() -> anyhow::Result<()> {
        let theme = Theme::default();
        let font = Arc::new(SDFFont::new(include_bytes!("../../../Deltos/core/fonts/BlexMonoNerdFontMono-Regular.ttf"))?);
        let zero_neg = LayerTile::new(Uuid::new_v4(), Box::new(ZeroNegative), Vec2::ZERO, 0)?;

        assert!(weights_tile()?.body(&theme, &font).is_some());
        assert!(zero_neg.body(&theme, &font).is_none());
        Ok(())
    }

    #[test]
    fn valid_values_change_the_layer() -> anyhow::Result<()> {
        let mut tile = weights_tile()?;

        tile.set_property("depth", " 5 ")?;
        tile.set_property("weight_decay", "0.25")?;

        assert_eq!(property(&tile, "depth").as_deref(), Some("5"));
        assert_eq!(property(&tile, "weight_decay").as_deref(), Some("0.25"));
        Ok(())
    }

    #[test]
    fn invalid_values_leave_the_layer_unchanged() -> anyhow::Result<()> {
        let mut tile = weights_tile()?;

        assert!(tile.set_property("depth", "abc").is_err());
        assert!(tile.set_property("depth", "1.5").is_err());
        assert!(tile.set_property("depth", "-1").is_err());
        assert!(tile.set_property("missing", "1").is_err());

        assert_eq!(property(&tile, "depth").as_deref(), Some("2"));
        Ok(())
    }

    #[test]
    fn wires_only_run_from_the_output_to_input_slots() -> anyhow::Result<()> {
        let mut tile = weights_tile()?;
        let other = Uuid::new_v4();

        assert!(tile.connect(input_slot(0), other, input_slot(0)).is_err());
        assert!(tile.connect(OUTPUT, other, OUTPUT).is_err());
        assert!(tile.connect(OUTPUT, tile.tile_id(), input_slot(0)).is_err());
        tile.connect(OUTPUT, other, input_slot(0))?;

        assert_eq!(tile.wires(), [Wire::new(other, input_slot(0))]);
        Ok(())
    }
}
