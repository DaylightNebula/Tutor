mod editor;
mod schema;
mod tiles;
mod training;

use deltos::{DeltosEditor, DeltosPlugin, editors::text::SimpleTextDeltosEditorPlugin};
use deltos_node_graphs::{DeltosNodeGraphPlugin, editor::NodeGraphFileEditor};
use gearbox::{App, Plugin, anarchy::ResMut, anyhow};

use crate::{editor::ModelEditor, schema::SchemaOverlayPlugin, training::TrainingPlugin};

/// Registers the model design editor with the Deltos editor, so `.design` files open as node
/// graphs.
struct ModelEditorPlugin;

impl Plugin for ModelEditorPlugin {
    fn build(self, app: App) -> App {
        app .add_plugin(DeltosNodeGraphPlugin)
            .add_plugin(SchemaOverlayPlugin)
            .add_plugin(TrainingPlugin)
            .on_startup(|mut editor: ResMut<DeltosEditor>| {
                let graph_editor = NodeGraphFileEditor::new(ModelEditor::default());
                editor.add_editor(Box::new(graph_editor), false);
            })
    }
}

/// Opens the editor on the local file system, which only exists natively.  On the web the
/// editor starts without sources.
fn main() -> anyhow::Result<()> {
    let plugin = DeltosPlugin::default();

    #[cfg(not(target_arch = "wasm32"))]
    let plugin = plugin.with_source(deltos::sources::local::LocalFileSource);

    App::new()
        .add_plugin(plugin)
        .add_plugin(SimpleTextDeltosEditorPlugin)
        .add_plugin(ModelEditorPlugin)
        .run()
}
