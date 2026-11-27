use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anarchy::{Component, ComponentMeta, EntityID, Event, Query, Res, ResMut, World, next_entity_id};
use deltos::{DeltosEditor, editors::text::OpenFile, notify::NotifyLevel, sources::TaskResult};
use gearbox::{
    Plugin,
    anyhow::{self, Context},
    glam::{Vec2, Vec4},
    ui::{
        Activate, AlignItems, ComputedNode, Display, JustifyContent, Node, PositionType, Rect, Submit, TextInput, Tooltip,
        UiBundle, UiInput, Val, px, themed::{self, Label, Themed}
    }
};
use tutor_core::data::{DataType, DesignSchema, SchemaEntry};

use crate::training;

/// A design's schema, shared between its `ModelEditor` and the SCHEMA overlay that edits it.
pub type SharedSchema = Arc<Mutex<DesignSchema>>;

/// The data types offered for a schema entry, with the text the type drop down shows for each.
const DATA_TYPES: [(DataType, &str); 3] = [
    (DataType::Decimal, "Decimal"),
    (DataType::Integer, "Integer"),
    (DataType::Enum, "Enum")
];

/// Width of a schema entry's name field, in logical pixels.
const NAME_WIDTH: f32 = 140.0;

/// Height of a schema entry's name field, matching the themed widgets' row height.
const NAME_HEIGHT: f32 = 24.0;

/// Width of a type drop down's button and list, in logical pixels, so rows line up.
const TYPE_WIDTH: f32 = 90.0;

/// Width of the import drop down's list, in logical pixels.
const IMPORT_WIDTH: f32 = 140.0;

/// Space between the overlay and the top right corner of the graph, in logical pixels.
const OVERLAY_MARGIN: f32 = 8.0;

/// Open drop down lists draw over everything but Gearbox's menus and tooltips.
const DROPDOWN_Z: i32 = i32::MAX - 2;

/// Adds the systems that open and close the SCHEMA panel and its drop downs, and apply their
/// edits to the design's schema.  Edits to files opened read only are ignored.
pub struct SchemaOverlayPlugin;

impl Plugin for SchemaOverlayPlugin {
    fn build(self, app: gearbox::App) -> gearbox::App {
        app .on_render_update(schema_buttons)
            .on_render_update(schema_names)
            .on_render_update(run_dropdowns)
    }
}

/// Which list of a schema an entry is in.
///
/// Variants:
/// * Inputs: The columns the model reads.
/// * Outputs: The columns the model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Inputs,
    Outputs
}

impl Side {
    /// The list entries move to from this one.
    fn other(self) -> Self {
        match self {
            Self::Inputs => Self::Outputs,
            Self::Outputs => Self::Inputs
        }
    }
}

/// What choosing an item of a drop down does.
///
/// Variants:
/// * SetType: Sets the data type of the entry at `index` of `side`.
/// * ImportCsv: Asks for a csv file and adds its columns to the inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DropdownAction {
    SetType { side: Side, index: usize, data_type: DataType },
    ImportCsv
}

/// Marks the root of a SCHEMA overlay.
///
/// Fields:
/// * schema: The schema the overlay edits, shared with its `ModelEditor`.
#[derive(Debug, Clone, Component)]
struct SchemaOverlay {
    schema: SharedSchema
}

/// Marks the SCHEMA button, which opens and closes the `SchemaPanel` of its overlay.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaToggle;

/// Marks the panel holding the input and output columns.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaPanel;

/// Marks every panel opened from the overlay's button bar, so opening one closes the others.
#[derive(Debug, Clone, Copy, Component)]
pub(crate) struct OverlayPanel;

/// Marks the node holding the entry rows of one side, rebuilt when entries are added, moved, or
/// removed.
///
/// Fields:
/// * side: The list whose entries are shown.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaList {
    side: Side
}

/// Marks an entry's name field.
///
/// Fields:
/// * side: The list the entry is in.
/// * index: Position of the entry in its list.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaName {
    side: Side,
    index: usize
}

/// Marks an entry's button that moves it to the other list.
///
/// Fields:
/// * side: The list the entry is in.
/// * index: Position of the entry in its list.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaMove {
    side: Side,
    index: usize
}

/// Marks an entry's remove button.
///
/// Fields:
/// * side: The list the entry is in.
/// * index: Position of the entry in its list.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaRemove {
    side: Side,
    index: usize
}

/// Marks a list's add button.
///
/// Fields:
/// * side: The list a new entry is added to.
#[derive(Debug, Clone, Copy, Component)]
struct SchemaAdd {
    side: Side
}

/// Marks a button that opens and closes a drop down list below it.
///
/// Fields:
/// * list: The `DropdownList` the button opens.
#[derive(Debug, Clone, Copy, Component)]
struct DropdownButton {
    list: EntityID
}

/// A drop down list of `DropdownItem`s, shown below its button while open.  It is a fixed
/// position node, so the panel it's in doesn't cut it off.
///
/// Fields:
/// * anchor: The `DropdownButton` the list opens below.
/// * open: Whether the list is shown.
#[derive(Debug, Clone, Copy, Component)]
struct DropdownList {
    anchor: EntityID,
    open: bool
}

/// Marks an item of a `DropdownList`.
///
/// Fields:
/// * action: What choosing the item does.
#[derive(Debug, Clone, Copy, Component)]
struct DropdownItem {
    action: DropdownAction
}

/// Builds the overlay drawn in the top right of a design's graph: a bar with the SCHEMA and TRAIN
/// buttons over their closed panels.  TRAIN and its panel are left out of web builds.  The schema panel has an import drop down in its top right,
/// the input columns on the left, and the output columns on the right.  The training panel is
/// built by `training::train_panel`.
///
/// # Parameters
/// * schema: The schema the overlay shows and edits.
pub fn overlay(schema: &SharedSchema) -> UiBundle {
    let current = lock(schema).clone();

    // the import drop down, in the top right of the panel
    let header = themed::row()
        .style(|node| node
            .width(Val::Percent(1.0))
            .justify(JustifyContent::End))
        .child(dropdown("Import", None, IMPORT_WIDTH, vec![("From CSV".into(), DropdownAction::ImportCsv)]));

    // the panel, closed until SCHEMA is clicked
    let panel = themed::frame(None)
        .style(|node| node.display(Display::None))
        .with(SchemaPanel)
        .with(OverlayPanel)
        .child(header)
        .child(themed::row()
            .style(|node| node.align(AlignItems::Start))
            .child(side_column(Side::Inputs, &current.inputs))
            .child(themed::vseparator())
            .child(side_column(Side::Outputs, &current.outputs)));

    // the button bar, with the panels under it, all against the right edge.  Training is left
    // out on the web for now, where it would run on the page's only thread and freeze it
    let train = !cfg!(target_arch = "wasm32");
    let mut buttons = themed::row().child(themed::button("SCHEMA").with(SchemaToggle));
    if train {
        buttons = buttons.child(training::train_button());
    }
    let mut overlay = UiBundle::new(Node::column()
        .gap(px(4.0))
        .align(AlignItems::End)
        .absolute(Rect::new_top_right(px(OVERLAY_MARGIN), px(OVERLAY_MARGIN))))
        .with(SchemaOverlay { schema: schema.clone() })
        .child(buttons)
        .child(panel);
    if train {
        overlay = overlay.child(training::train_panel());
    }
    overlay
}

/// Builds one side of the panel: its title, a row per entry, then an add button.
fn side_column(side: Side, entries: &[SchemaEntry]) -> UiBundle {
    let title = match side {
        Side::Inputs => "Inputs",
        Side::Outputs => "Outputs"
    };
    themed::column()
        .child(themed::strong(title))
        .child(themed::column()
            .with(SchemaList { side })
            .children(entry_rows(side, entries)))
        .child(themed::button("+ Add").with(SchemaAdd { side }))
}

/// Builds a row per entry: its name field, data type drop down, a button moving it to the other
/// list, and a remove button.
fn entry_rows(side: Side, entries: &[SchemaEntry]) -> Vec<UiBundle> {
    let (move_text, move_hint) = match side {
        Side::Inputs => ("\u{2192}", "Move to outputs"),
        Side::Outputs => ("\u{2190}", "Move to inputs")
    };
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let name = UiBundle::new(Node::new()
                .shrink(0.0)
                .size(px(NAME_WIDTH), px(NAME_HEIGHT))
                .padding(Rect::new(px(0.0), px(0.0), px(6.0), px(6.0)))
                .border(1.0, Vec4::ZERO))
                .with(TextInput::new(entry.name.clone()).hint("name"))
                .interactive()
                .with(Themed::TextInput)
                .with(SchemaName { side, index });
            let types = DATA_TYPES
                .iter()
                .map(|(data_type, label)| (label.to_string(), DropdownAction::SetType { side, index, data_type: *data_type }))
                .collect();
            let data_type = dropdown(type_label(entry.data_type), Some(TYPE_WIDTH), TYPE_WIDTH, types);
            let move_button = themed::button(move_text)
                .with(Tooltip(move_hint.into()))
                .with(SchemaMove { side, index });
            let remove = themed::button("x").with(SchemaRemove { side, index });
            themed::row()
                .child(name)
                .child(data_type)
                .child(move_button)
                .child(remove)
        })
        .collect()
}

/// Builds a button showing `label` that opens a list of `items` below it, each a label and what
/// choosing it does.  The button is `button_width` logical pixels wide, or fits its label if
/// `None`, and the list is `list_width` wide.
fn dropdown(label: &str, button_width: Option<f32>, list_width: f32, items: Vec<(String, DropdownAction)>) -> UiBundle {
    let (button_id, list_id) = (next_entity_id(), next_entity_id());

    // the list, hidden and placed by `run_dropdowns`
    let items = items
        .into_iter()
        .map(|(text, action)| themed::menu_item(text).with(DropdownItem { action }));
    let list = themed::frame(Some(list_width))
        .style(|node| node
            .fixed(Rect::new_top_left(px(0.0), px(0.0)))
            .z_index(DROPDOWN_Z)
            .display(Display::None))
        .with_id(list_id)
        .with(DropdownList { anchor: button_id, open: false })
        .children(items);

    // the button, which the list sits beside so the button's label is its only one
    let button = themed::button(label)
        .style(|node| match button_width {
            Some(width) => node.width(px(width)),
            None => node
        })
        .with_id(button_id)
        .with(DropdownButton { list: list_id });
    UiBundle::new(Node::new().shrink(0.0))
        .child(button)
        .child(list)
}

/// Opens and closes the panel, and adds, moves, and removes entries, when their buttons are
/// clicked.
fn schema_buttons(world: World, activated: Event<Activate>) {
    for Activate { entity } in activated.read() {
        // open or close the panel
        if has::<SchemaToggle>(&world, entity) {
            toggle_panel::<SchemaPanel>(&world, entity);
            continue;
        }

        // change the entries, then rebuild the rows of the sides that changed
        let sides = if let Some(SchemaAdd { side }) = component(&world, entity) {
            edit(&world, entity, |schema| add_entry(schema, side));
            vec![side]
        } else if let Some(SchemaRemove { side, index }) = component(&world, entity) {
            edit(&world, entity, |schema| remove_entry(schema, side, index));
            vec![side]
        } else if let Some(SchemaMove { side, index }) = component(&world, entity) {
            edit(&world, entity, |schema| move_entry(schema, side, index));
            vec![side, side.other()]
        } else {
            continue
        };

        // find the overlay first, since rebuilding a side despawns the clicked button with it
        let Some(overlay) = overlay_of(&world, entity) else { continue };
        for side in sides {
            rebuild_list(&world, overlay, side);
        }
    }
}

/// Renames entries when their name fields are submitted.
fn schema_names(world: World, submitted: Event<Submit>) {
    for Submit { entity, text } in submitted.read() {
        let Some(SchemaName { side, index }) = component(&world, entity) else { continue };
        edit(&world, entity, |schema| rename_entry(schema, side, index, &text));
    }
}

/// Runs every drop down: its button opens or closes its list, choosing an item runs the item's
/// action, and pressing anywhere else closes the open lists.  Open lists are placed below their
/// buttons, kept on screen.
fn run_dropdowns(world: World, input: Res<UiInput>, activated: Event<Activate>, mut editor: ResMut<DeltosEditor>) {
    let mut lists = Query::<&DropdownList>::new(&world);
    let mut all = Vec::new();
    while let Ok(Some((id, list))) = lists.next() {
        all.push((id, *list));
    }
    drop(lists);

    // choosing an item of an open list runs it, then closes the list
    let chosen = all
        .iter()
        .filter(|(_, list)| list.open)
        .flat_map(|(id, _)| world
            .hierarchy()
            .descendants(*id)
            .collect::<Vec<_>>())
        .find(|item| input.activated(*item));
    if let Some(item) = chosen
        && let Some(DropdownItem { action }) = component(&world, item) {
        run_action(&world, item, action, &mut editor);
    }
    if chosen.is_some() {
        for (id, _) in &all {
            set_open(&world, *id, false);
        }
    }

    // pressing outside an open list and its button closes it
    if input.pressed {
        let under_pointer = |id: EntityID| computed(&world, id).is_some_and(|node| node.contains(input.cursor));
        for (id, list) in &all {
            if list.open && !under_pointer(*id) && !under_pointer(list.anchor) {
                set_open(&world, *id, false);
            }
        }
    }

    // a drop down button opens its list, closing the others, or closes it if it was open
    for Activate { entity } in activated.read() {
        let Some(DropdownButton { list }) = component(&world, entity) else { continue };
        let was_open = component::<DropdownList>(&world, list).is_some_and(|list| list.open);
        for (id, _) in &all {
            set_open(&world, *id, false);
        }
        set_open(&world, list, !was_open);
    }

    // show the open lists below their buttons, using last frame's sizes
    let scale = input.scale.max(f32::EPSILON);
    for (id, _) in &all {
        let Some(list) = component::<DropdownList>(&world, *id) else { continue };
        let at = computed(&world, list.anchor).map(|anchor| {
            let size = computed(&world, *id).map_or(Vec2::ZERO, |node| node.size);
            let below = anchor.position + Vec2::new(0.0, anchor.size.y);
            below
                .min(input.screen - size)
                .max(Vec2::ZERO) / scale
        });
        let Ok(Some(mut node)) = world.handle(*id).get::<&mut Node>() else { continue };
        let display = if list.open && at.is_some() { Display::Flex } else { Display::None };
        if *node.style().display() != display {
            node.style_mut().set_display(display);
        }
        if let Some(at) = at.filter(|_| list.open) {
            let edges = Rect::new_top_left(px(at.y), px(at.x));
            if !matches!(node.style().position_type(), PositionType::Fixed(now) if *now == edges) {
                node.style_mut().set_position_type(PositionType::Fixed(edges));
            }
        }
    }
}

/// Runs a drop down item's action.  `item` is the chosen item, used to find its overlay.
fn run_action(world: &World, item: EntityID, action: DropdownAction, editor: &mut DeltosEditor) {
    match action {
        DropdownAction::SetType { side, index, data_type } => {
            edit(world, item, |schema| set_entry_type(schema, side, index, data_type));
            relabel_dropdown(world, item, type_label(data_type));
        }
        DropdownAction::ImportCsv => import_csv(world, item, editor)
    }
}

/// Asks for a csv file in the background, then adds its columns to the inputs of the overlay
/// holding `entity`.  Failures are shown as warnings, and closing the file picker does nothing.
fn import_csv(world: &World, entity: EntityID, editor: &DeltosEditor) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    editor.tasks.spawn(Box::pin(async move {
        let picked = pick_csv_columns().await;
        let apply: TaskResult = Box::new(move |world: &World, editor: &mut DeltosEditor| match picked {
            Ok(Some(columns)) => {
                edit(world, overlay, |schema| import_columns(schema, &columns));
                rebuild_list(world, overlay, Side::Inputs);
            }
            Ok(None) => {}
            Err(err) => editor.notify(NotifyLevel::Warning, format!("Couldn't import the schema: {err:#}"))
        });
        apply
    }));
}

/// Asks the user for a csv file and reads its column names, or gives `None` if no file was
/// picked.
async fn pick_csv_columns() -> anyhow::Result<Option<Vec<String>>> {
    let Some(file) = rfd::AsyncFileDialog::new()
        .set_title("Import schema from CSV")
        .add_filter("CSV", &["csv"])
        .pick_file()
        .await else { return Ok(None) };
    let bytes = file.read().await;
    csv_columns(&bytes)
        .with_context(|| format!("couldn't read the columns of {}", file.file_name()))
        .map(Some)
}

/// The column names in the header of a csv file, trimmed, skipping blank names.
fn csv_columns(csv: &[u8]) -> anyhow::Result<Vec<String>> {
    let mut reader = csv::Reader::from_reader(csv);
    let columns = reader
        .headers()
        .context("the csv header can't be read")?
        .iter()
        .map(|name| name
            .trim()
            .to_string())
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    anyhow::ensure!(!columns.is_empty(), "the csv has no column names");
    Ok(columns)
}

/// Shows or hides the panel marked with `P` in the overlay holding `entity`.  Showing it hides
/// the overlay's other panels.
pub(crate) fn toggle_panel<P: ComponentMeta + Sync>(world: &World, entity: EntityID) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    let panels: Vec<EntityID> = world
        .hierarchy()
        .descendants(overlay)
        .filter(|id| has::<OverlayPanel>(world, *id))
        .collect();
    let Some(panel) = panels
        .iter()
        .copied()
        .find(|id| has::<P>(world, *id)) else { return };

    // flip the panel, then close the others if it opened
    let opened = match world.handle(panel).get::<&mut Node>() {
        Ok(Some(mut node)) => {
            let opened = *node.style().display() == Display::None;
            node.style_mut().set_display(if opened { Display::Flex } else { Display::None });
            opened
        }
        _ => return
    };
    if !opened { return }
    for other in panels.into_iter().filter(|id| *id != panel) {
        if let Ok(Some(mut node)) = world.handle(other).get::<&mut Node>() {
            node.style_mut().set_display(Display::None);
        }
    }
}

/// Sets the text of the button of the drop down list holding `item` to `label`.
fn relabel_dropdown(world: &World, item: EntityID, label: &str) {
    let Some(list) = world
        .hierarchy()
        .ancestors(item)
        .find_map(|id| component::<DropdownList>(world, id)) else { return };
    let labels: Vec<EntityID> = world
        .hierarchy()
        .descendants(list.anchor)
        .collect();
    for id in labels {
        if let Ok(Some(mut text)) = world.handle(id).get::<&mut Label>() {
            text.text = label.to_string();
        }
    }
}

/// Opens or closes the drop down list `list`.
fn set_open(world: &World, list: EntityID, open: bool) {
    if let Ok(Some(mut list)) = world.handle(list).get::<&mut DropdownList>() {
        list.open = open;
    }
}

/// Applies `change` to the schema of the overlay holding `entity`, and marks its file as having
/// unsaved changes when `change` made one.  Nothing changes when the file opened read only.
fn edit(world: &World, entity: EntityID, change: impl FnOnce(&mut DesignSchema) -> bool) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    let Some(SchemaOverlay { schema }) = component(world, overlay) else { return };

    // find the file the overlay's graph was opened from, skipping files that can't be changed
    let file = world
        .hierarchy()
        .ancestors(overlay)
        .find(|ancestor| has::<OpenFile>(world, *ancestor));
    let read_only = file
        .and_then(|file| component::<OpenFile>(world, file))
        .is_some_and(|file| file.read_only);
    if read_only { return }

    // change the schema, then mark the file dirty so it can be saved
    if change(&mut lock(&schema))
        && let Some(file) = file
        && let Ok(Some(mut file)) = world.handle(file).get::<&mut OpenFile>() {
        file.dirty = true;
    }
}

/// Replaces the rows of `side` in the overlay holding `entity` with rows built from its schema.
fn rebuild_list(world: &World, entity: EntityID, side: Side) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    let Some(SchemaOverlay { schema }) = component(world, overlay) else { return };
    let Some(list) = world
        .hierarchy()
        .descendants(overlay)
        .find(|id| component::<SchemaList>(world, *id).is_some_and(|list| list.side == side)) else { return };

    // remove the old rows, then add new ones from the schema's current entries
    let old: Vec<EntityID> = world
        .hierarchy()
        .children(list)
        .map(|children| children
            .iter()
            .copied()
            .collect())
        .unwrap_or_default();
    for id in old {
        world.handle(id).despawn_recursive();
    }
    let rows = entry_rows(side, entries(&lock(&schema), side));
    for row in rows {
        world.spawn_with_parent(Some(list), row.into_entity_builder());
    }
}

/// The root of the overlay `entity` is in, if any.
pub(crate) fn overlay_of(world: &World, entity: EntityID) -> Option<EntityID> {
    std::iter::once(entity)
        .chain(world
            .hierarchy()
            .ancestors(entity))
        .find(|id| has::<SchemaOverlay>(world, *id))
}

/// Whether `entity` has a `T`.
pub(crate) fn has<T: ComponentMeta + Sync>(world: &World, entity: EntityID) -> bool {
    world
        .handle(entity)
        .get::<&T>()
        .ok()
        .flatten()
        .is_some()
}

/// A copy of `entity`'s `T`, if it has one.
pub(crate) fn component<T: ComponentMeta + Clone + Sync>(world: &World, entity: EntityID) -> Option<T> {
    world
        .handle(entity)
        .get::<&T>()
        .ok()
        .flatten()
        .map(|found| found.clone())
}

/// Where `entity`'s node was laid out last frame, in window pixels.
fn computed(world: &World, entity: EntityID) -> Option<ComputedNode> {
    world
        .handle(entity)
        .get::<&ComputedNode>()
        .ok()
        .flatten()
        .map(|node| *node)
}

/// Locks `schema`, still giving the schema if a thread panicked while holding it.
pub fn lock(schema: &SharedSchema) -> MutexGuard<'_, DesignSchema> {
    schema
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The entries of `side` in `schema`.
fn entries(schema: &DesignSchema, side: Side) -> &[SchemaEntry] {
    match side {
        Side::Inputs => &schema.inputs,
        Side::Outputs => &schema.outputs
    }
}

/// The entries of `side` in `schema`, mutably.
fn entries_mut(schema: &mut DesignSchema, side: Side) -> &mut Vec<SchemaEntry> {
    match side {
        Side::Inputs => &mut schema.inputs,
        Side::Outputs => &mut schema.outputs
    }
}

/// The text the type drop down shows for `data_type`.
fn type_label(data_type: DataType) -> &'static str {
    DATA_TYPES
        .iter()
        .find(|(option, _)| *option == data_type)
        .map_or("?", |(_, label)| *label)
}

/// Adds an unnamed entry of `DataType::default()` to the end of `side`.  Always changes the schema.
fn add_entry(schema: &mut DesignSchema, side: Side) -> bool {
    entries_mut(schema, side).push(SchemaEntry { name: String::new(), data_type: DataType::default() });
    true
}

/// Removes the entry at `index` of `side`, giving whether there was one.
fn remove_entry(schema: &mut DesignSchema, side: Side, index: usize) -> bool {
    let entries = entries_mut(schema, side);
    if index >= entries.len() { return false }
    entries.remove(index);
    true
}

/// Moves the entry at `index` of `side` to the end of the other side, giving whether there was
/// one.
fn move_entry(schema: &mut DesignSchema, side: Side, index: usize) -> bool {
    let entries = entries_mut(schema, side);
    if index >= entries.len() { return false }
    let entry = entries.remove(index);
    entries_mut(schema, side.other()).push(entry);
    true
}

/// Renames the entry at `index` of `side` to `name`, giving whether the name changed.
fn rename_entry(schema: &mut DesignSchema, side: Side, index: usize, name: &str) -> bool {
    match entries_mut(schema, side).get_mut(index) {
        Some(entry) if entry.name != name => {
            entry.name = name.to_string();
            true
        }
        _ => false
    }
}

/// Sets the data type of the entry at `index` of `side`, giving whether the type changed.
fn set_entry_type(schema: &mut DesignSchema, side: Side, index: usize, data_type: DataType) -> bool {
    match entries_mut(schema, side).get_mut(index) {
        Some(entry) if entry.data_type != data_type => {
            entry.data_type = data_type;
            true
        }
        _ => false
    }
}

/// Adds each of `columns` to the end of the inputs as `DataType::default()`, skipping names already
/// in the schema on either side.  Gives whether any were added.
fn import_columns(schema: &mut DesignSchema, columns: &[String]) -> bool {
    let mut changed = false;
    for column in columns {
        let known = schema.inputs
            .iter()
            .chain(&schema.outputs)
            .any(|entry| entry.name == *column);
        if known { continue }
        schema.inputs.push(SchemaEntry { name: column.clone(), data_type: DataType::default() });
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A schema with inputs `a` (decimal) and `b` (enum), and output `y` (integer).
    fn schema() -> DesignSchema {
        let entry = |name: &str, data_type| SchemaEntry { name: name.into(), data_type };
        DesignSchema {
            inputs: vec![entry("a", DataType::Decimal), entry("b", DataType::Enum)],
            outputs: vec![entry("y", DataType::Integer)]
        }
    }

    /// The names of the entries of `side` in `schema`.
    fn names(schema: &DesignSchema, side: Side) -> Vec<&str> {
        entries(schema, side)
            .iter()
            .map(|entry| entry.name.as_str())
            .collect()
    }

    #[test]
    fn entries_are_added_to_the_end_of_their_side() {
        let mut schema = schema();

        assert!(add_entry(&mut schema, Side::Outputs));

        assert_eq!(schema.inputs.len(), 2);
        assert_eq!(schema.outputs.len(), 2);
        assert_eq!(schema.outputs[1], SchemaEntry { name: String::new(), data_type: DataType::default() });
    }

    #[test]
    fn entries_are_removed_by_index() {
        let mut schema = schema();

        assert!(remove_entry(&mut schema, Side::Inputs, 0));
        assert!(!remove_entry(&mut schema, Side::Inputs, 5));

        assert_eq!(names(&schema, Side::Inputs), ["b"]);
    }

    #[test]
    fn entries_move_to_the_end_of_the_other_side_keeping_their_type() {
        let mut schema = schema();

        assert!(move_entry(&mut schema, Side::Inputs, 1));
        assert!(move_entry(&mut schema, Side::Outputs, 0));
        assert!(!move_entry(&mut schema, Side::Outputs, 9));

        assert_eq!(names(&schema, Side::Inputs), ["a", "y"]);
        assert_eq!(names(&schema, Side::Outputs), ["b"]);
        assert_eq!(schema.outputs[0].data_type, DataType::Enum);
        assert_eq!(schema.inputs[1].data_type, DataType::Integer);
    }

    #[test]
    fn renaming_reports_only_real_changes() {
        let mut schema = schema();

        assert!(rename_entry(&mut schema, Side::Outputs, 0, "z"));
        assert!(!rename_entry(&mut schema, Side::Outputs, 0, "z"));
        assert!(!rename_entry(&mut schema, Side::Outputs, 3, "q"));

        assert_eq!(schema.outputs[0].name, "z");
    }

    #[test]
    fn retyping_reports_only_real_changes() {
        let mut schema = schema();

        assert!(set_entry_type(&mut schema, Side::Inputs, 1, DataType::Integer));
        assert!(!set_entry_type(&mut schema, Side::Inputs, 1, DataType::Integer));

        assert_eq!(schema.inputs[1].data_type, DataType::Integer);
    }

    #[test]
    fn every_data_type_has_a_label() {
        for (data_type, label) in DATA_TYPES {
            assert_eq!(type_label(data_type), label);
        }
    }

    #[test]
    fn csv_headers_become_column_names() -> anyhow::Result<()> {
        let columns = csv_columns(b"sepal_length, sepal_width ,,species\n5.1,3.5,x,setosa\n")?;

        assert_eq!(columns, ["sepal_length", "sepal_width", "species"]);
        Ok(())
    }

    #[test]
    fn csvs_without_column_names_are_refused() {
        assert!(csv_columns(b"").is_err());
        assert!(csv_columns(b" , \n1,2\n").is_err());
    }

    #[test]
    fn imported_columns_are_added_to_the_inputs_with_the_default_data_type() {
        let mut schema = schema();
        let columns = ["b", "c", "y", "d"].map(String::from);

        assert!(import_columns(&mut schema, &columns));
        assert!(!import_columns(&mut schema, &columns));

        assert_eq!(names(&schema, Side::Inputs), ["a", "b", "c", "d"]);
        assert_eq!(names(&schema, Side::Outputs), ["y"]);
        assert_eq!(schema.inputs[1].data_type, DataType::Enum);
        assert!(schema.inputs[2..]
            .iter()
            .all(|entry| entry.data_type == DataType::default()));
    }
}
