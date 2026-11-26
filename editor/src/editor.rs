use std::collections::{BTreeMap, HashMap};

use deltos_node_graphs::{editor::NodeGraphEditor, tiles::NodeGraphTile};
use gearbox::{anyhow::{self, Context}, glam::Vec2};
use serde_json::Value;
use tutor_core::{designer::ModelDesigner, layers::layer_types, model::ModelNode};
use uuid::Uuid;

use crate::tiles::{LayerTile, OUTPUT, copy_layer, input_slot, slot_index};

/// Key of the tile positions in a saved design.  `ModelDesigner` ignores keys it doesn't know,
/// so designs saved by the editor still load anywhere else.
pub const POSITIONS_KEY: &str = "editor_positions";

/// Where the first automatically placed tile goes, in logical pixels from the graph's top left.
const LAYOUT_ORIGIN: Vec2 = Vec2::new(40.0, 40.0);

/// Space between automatically placed columns of tiles, one column per depth in the network.
const LAYOUT_COLUMN_SPACING: f32 = 280.0;

/// Space between automatically placed tiles in the same column.
const LAYOUT_ROW_SPACING: f32 = 220.0;

/// Node graph editor for model design files (`.design`), the json written by
/// `ModelDesigner::serialize`.  Each node of the design is a tile, and each wire into a tile's
/// input slot is one of the node's inputs, joined in slot order.  Tile positions are saved in the
/// design under `POSITIONS_KEY`, and tiles without a saved position are laid out by depth.
///
/// Fields:
/// * tiles: A tile per node of the design.
#[derive(Debug, Default)]
pub struct ModelEditor {
    tiles: Vec<LayerTile>
}

impl ModelEditor {
    /// The nodes feeding the tile with `tile_id`, in the order of the input slots their wires end
    /// at.
    fn inputs_of(&self, tile_id: Uuid) -> Vec<Uuid> {
        let mut inputs = self.tiles
            .iter()
            .flat_map(|tile| tile
                .wires()
                .iter()
                .filter(|wire| wire.target_tile() == tile_id)
                .filter_map(|wire| Some((slot_index(wire.target_input())?, tile.tile_id()))))
            .collect::<Vec<_>>();
        inputs.sort();
        inputs
            .into_iter()
            .map(|(_, input)| input)
            .collect()
    }
}

impl NodeGraphEditor for ModelEditor {
    type Tile = LayerTile;

    fn name(&self) -> String { "Model Design".into() }

    fn file_types(&self) -> Vec<String> { vec!["tutordesign".into()] }

    fn open(content: &str) -> anyhow::Result<Self> {
        if content.trim().is_empty() { return Ok(Self::default()) }

        // read the design, then the positions saved next to it
        let designer = ModelDesigner::deserialize(content).context("not a model design")?;
        let saved: Value = serde_json::from_str(content)?;
        let positions: HashMap<Uuid, [f32; 2]> = match saved.get(POSITIONS_KEY) {
            Some(positions) => serde_json::from_value(positions.clone()).context("tile positions are invalid")?,
            None => HashMap::new()
        };

        // a tile per node, placed where it was saved or laid out by depth
        let mut nodes = designer
            .nodes()
            .iter()
            .collect::<Vec<_>>();
        nodes.sort_by_key(|(id, _)| **id);
        let layout = auto_layout(&nodes);
        let mut tiles = Vec::with_capacity(nodes.len());
        for (id, node) in &nodes {
            let position = positions
                .get(id)
                .map(|[x, y]| Vec2::new(*x, *y))
                .or_else(|| layout.get(id).copied())
                .unwrap_or(LAYOUT_ORIGIN);
            tiles.push(LayerTile::new(**id, copy_layer(&**node.layer())?, position, node.inputs().len())?);
        }

        // wire each node's inputs into its slots, in order
        let mut editor = Self { tiles };
        for (id, node) in &nodes {
            for (slot, input) in node.inputs().iter().enumerate() {
                editor.tiles
                    .iter_mut()
                    .find(|tile| tile.tile_id() == *input)
                    .with_context(|| format!("node {id} reads from missing node {input}"))?
                    .connect(OUTPUT, **id, input_slot(slot))?;
            }
        }
        Ok(editor)
    }

    fn save(&self) -> anyhow::Result<String> {
        // build the design from the tiles and their wires
        let mut designer = ModelDesigner::new();
        for tile in &self.tiles {
            designer.insert_node(tile.tile_id(), copy_layer(tile.layer())?, self.inputs_of(tile.tile_id()))?;
        }

        // add the positions, writing through a json value so keys always come out in the same order
        let mut saved: Value = serde_json::from_str(&designer.serialize()?)?;
        let positions = self.tiles
            .iter()
            .map(|tile| (tile.tile_id(), [tile.position().x, tile.position().y]))
            .collect::<BTreeMap<_, _>>();
        saved
            .as_object_mut()
            .context("design wasn't saved as an object")?
            .insert(POSITIONS_KEY.into(), serde_json::to_value(positions)?);
        Ok(serde_json::to_string_pretty(&saved)?)
    }

    fn new_tile_options(&self) -> Vec<String> {
        layer_types()
            .iter()
            .map(|layer_type| layer_type.name.to_string())
            .collect()
    }

    fn new_tile(&mut self, name: String, position: Vec2) {
        let Some(layer_type) = layer_types()
            .into_iter()
            .find(|layer_type| layer_type.name == name) else { return };
        match LayerTile::new(Uuid::new_v4(), (layer_type.create)(), position, 0) {
            Ok(tile) => self.tiles.push(tile),
            Err(err) => eprintln!("Failed to add a {name} tile: {err:#}")
        }
    }

    fn remove_tile(&mut self, tile_id: Uuid) -> anyhow::Result<()> {
        let index = self.tiles
            .iter()
            .position(|tile| tile.tile_id() == tile_id)
            .with_context(|| format!("No tile {tile_id} in the graph"))?;
        self.tiles.remove(index);
        Ok(())
    }

    fn tiles(&self) -> &[LayerTile] { &self.tiles }

    fn tiles_mut(&mut self) -> &mut [LayerTile] { &mut self.tiles }
}

/// Places every node in a column by its depth, the longest chain of nodes feeding it, so data
/// flows left to right.  Nodes in a column are stacked in the order given.  Nodes on a cycle or
/// reading from missing nodes are placed as if those inputs weren't there.
fn auto_layout(nodes: &[(&Uuid, &ModelNode)]) -> HashMap<Uuid, Vec2> {
    let inputs: HashMap<Uuid, &Vec<Uuid>> = nodes
        .iter()
        .map(|(id, node)| (**id, node.inputs()))
        .collect();

    // work out each node's depth, walking its inputs first
    let mut depths: HashMap<Uuid, usize> = HashMap::new();
    for (id, _) in nodes {
        depth_of(**id, &inputs, &mut depths, &mut Vec::new());
    }

    // stack each depth's nodes in a column
    let mut rows: HashMap<usize, usize> = HashMap::new();
    nodes
        .iter()
        .map(|(id, _)| {
            let depth = depths
                .get(*id)
                .copied()
                .unwrap_or(0);
            let row = rows
                .entry(depth)
                .or_default();
            let position = LAYOUT_ORIGIN + Vec2::new(depth as f32 * LAYOUT_COLUMN_SPACING, *row as f32 * LAYOUT_ROW_SPACING);
            *row += 1;
            (**id, position)
        })
        .collect()
}

/// The depth of `id`, remembered in `depths`.  `visiting` holds the nodes being walked, so an
/// input on a cycle back to one of them counts as depth 0 instead of recursing forever.
fn depth_of(id: Uuid, inputs: &HashMap<Uuid, &Vec<Uuid>>, depths: &mut HashMap<Uuid, usize>, visiting: &mut Vec<Uuid>) -> usize {
    if let Some(depth) = depths.get(&id) { return *depth }
    let Some(node_inputs) = inputs.get(&id) else { return 0 };
    if visiting.contains(&id) { return 0 }

    visiting.push(id);
    let depth = node_inputs
        .iter()
        .filter(|input| inputs.contains_key(input))
        .map(|input| depth_of(*input, inputs, depths, visiting) + 1)
        .max()
        .unwrap_or(0);
    visiting.pop();
    depths.insert(id, depth);
    depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use tutor_core::layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative};

    /// A design where two input nodes feed a weights node in the order `[second, first]`,
    /// followed by an activation and an output.  Gives the design and the node IDs in the order
    /// first, second, weights, activation, output.
    fn design() -> (ModelDesigner, [Uuid; 5]) {
        let mut designer = ModelDesigner::new();
        let first = designer.add_node(InputSchema::new(), Vec::new());
        let second = designer.add_node(InputSchema::new(), Vec::new());
        let weights = designer.add_node(Weights::random(3, 0.01), vec![second, first]);
        let activation = designer.add_node(ZeroNegative, vec![weights]);
        let output = designer.add_node(OutputSchema::new(), vec![activation]);
        (designer, [first, second, weights, activation, output])
    }

    /// The tile with `id` in `editor`.
    fn tile(editor: &ModelEditor, id: Uuid) -> anyhow::Result<&LayerTile> {
        editor
            .tiles()
            .iter()
            .find(|tile| tile.tile_id() == id)
            .with_context(|| format!("no tile {id}"))
    }

    #[test]
    fn empty_files_open_as_empty_designs() -> anyhow::Result<()> {
        assert!(ModelEditor::open("")?.tiles().is_empty());
        assert!(ModelEditor::open("  \n")?.tiles().is_empty());
        Ok(())
    }

    #[test]
    fn files_that_arent_designs_fail_to_open() {
        assert!(ModelEditor::open("not json").is_err());
        assert!(ModelEditor::open(r#"{"nodes": {"x": 1}}"#).is_err());
    }

    #[test]
    fn saving_keeps_every_node_and_the_order_of_its_inputs() -> anyhow::Result<()> {
        let (designer, [first, second, weights, activation, output]) = design();

        let editor = ModelEditor::open(&designer.serialize()?)?;
        let saved = ModelDesigner::deserialize(&editor.save()?)?;

        let inputs = |id: Uuid| saved
            .nodes()
            .get(&id)
            .map(|node| node.inputs().clone());
        assert_eq!(saved.nodes().len(), 5);
        assert_eq!(inputs(first), Some(Vec::new()));
        assert_eq!(inputs(weights), Some(vec![second, first]));
        assert_eq!(inputs(activation), Some(vec![weights]));
        assert_eq!(inputs(output), Some(vec![activation]));
        Ok(())
    }

    #[test]
    fn saved_designs_still_build_into_models() -> anyhow::Result<()> {
        let (designer, ids) = design();

        let editor = ModelEditor::open(&designer.serialize()?)?;
        let saved = ModelDesigner::deserialize(&editor.save()?)?;

        // building needs data, so check the network itself is a valid one instead
        let model = tutor_core::model::Model::new(saved
            .nodes()
            .iter()
            .map(|(id, node)| Ok((*id, ModelNode::new(copy_layer(&**node.layer())?, node.inputs().clone()))))
            .collect::<anyhow::Result<_>>()?)?;
        assert_eq!(model.order().len(), ids.len());
        Ok(())
    }

    #[test]
    fn positions_are_saved_and_reopened() -> anyhow::Result<()> {
        let (designer, [_, _, weights, _, _]) = design();
        let mut editor = ModelEditor::open(&designer.serialize()?)?;
        editor
            .tiles_mut()
            .iter_mut()
            .find(|tile| tile.tile_id() == weights)
            .context("no weights tile")?
            .set_position(Vec2::new(12.5, 600.0));

        let reopened = ModelEditor::open(&editor.save()?)?;

        assert_eq!(tile(&reopened, weights)?.position(), Vec2::new(12.5, 600.0));
        Ok(())
    }

    #[test]
    fn unsaved_positions_are_laid_out_by_depth() -> anyhow::Result<()> {
        let (designer, [first, second, weights, activation, output]) = design();

        let editor = ModelEditor::open(&designer.serialize()?)?;

        let x = |id: Uuid| tile(&editor, id).map(|tile| tile.position().x);
        assert_eq!(x(first)?, x(second)?);
        assert_ne!(tile(&editor, first)?.position().y, tile(&editor, second)?.position().y);
        assert!(x(first)? < x(weights)?);
        assert!(x(weights)? < x(activation)?);
        assert!(x(activation)? < x(output)?);
        Ok(())
    }

    #[test]
    fn saving_twice_gives_the_same_text() -> anyhow::Result<()> {
        let (designer, _) = design();
        let editor = ModelEditor::open(&designer.serialize()?)?;

        assert_eq!(editor.save()?, editor.save()?);
        Ok(())
    }

    #[test]
    fn every_layer_type_can_be_added() -> anyhow::Result<()> {
        let mut editor = ModelEditor::default();

        for name in editor.new_tile_options() {
            editor.new_tile(name, Vec2::ZERO);
        }

        let titles = editor
            .tiles()
            .iter()
            .map(NodeGraphTile::title)
            .collect::<Vec<_>>();
        assert_eq!(titles, ["InputSchema", "OutputSchema", "Weights", "ZeroNegative"]);
        assert_eq!(ModelDesigner::deserialize(&editor.save()?)?.nodes().len(), 4);
        Ok(())
    }

    #[test]
    fn removed_tiles_are_left_out_of_the_design() -> anyhow::Result<()> {
        let (designer, [_, second, _, _, _]) = design();
        let mut editor = ModelEditor::open(&designer.serialize()?)?;

        editor.remove_tile(second)?;

        let saved = ModelDesigner::deserialize(&editor.save()?)?;
        assert!(!saved.nodes().contains_key(&second));
        assert!(editor.remove_tile(second).is_err());
        Ok(())
    }
}
