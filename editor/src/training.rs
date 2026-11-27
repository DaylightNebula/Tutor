use std::{path::Path, sync::{Arc, Mutex, MutexGuard, PoisonError, atomic::{AtomicBool, Ordering}}, time::Duration};

use anarchy::{Component, EntityID, Event, Query, ResMut, Resource, World};
use deltos::{DeltosEditor, editors::text::OpenFile, notify::NotifyLevel, sources::TaskResult};
use deltos_node_graphs::GraphRoot;
use gearbox::{
    Plugin,
    anyhow::{self, Context},
    glam::Vec4,
    ui::{Activate, AlignItems, ContextMenu, Display, JustifyContent, Node, Rect, Tone, UiBundle, Val, px, themed::{self, Label}}
};
use tutor_core::{data::DataSheet, designer::ModelDesigner, model::{Evaluation, TrainingProgress, TrainingSummary}};

use crate::schema::{OverlayPanel, component, has, overlay_of, toggle_panel};

/// How far each training step moves the parameters, the same as the iris trainer uses.
const LEARNING_RATE: f32 = 0.05;

/// Rows per training step.
const BATCH_SIZE: usize = 32;

/// The fraction the loss must drop by for an epoch to count as improving.
const MIN_IMPROVEMENT: f32 = 1e-4;

/// Epochs in a row without improvement before training stops.
const PATIENCE: u32 = 50;

/// Epochs between progress lines in the training log.
const REPORT_EVERY: u32 = 10;

/// The most lines the training log shows, the newest last.
const LOG_LINES: usize = 8;

/// Width of the training panel, in logical pixels.
const PANEL_WIDTH: f32 = 380.0;

/// Height of the training log, in logical pixels.
const LOG_HEIGHT: f32 = 140.0;

/// Width of the results popup, in logical pixels.
const POPUP_WIDTH: f32 = 420.0;

/// Width of the names in the results popup, so the values line up.
const POPUP_LEGEND_WIDTH: f32 = 120.0;

/// The results popup draws over the graph and overlay, but under drop downs and menus.
const POPUP_Z: i32 = i32::MAX - 3;

/// The text of the start button while no run is going.
const START_TEXT: &str = "Start training";

/// The text of the start button while a run is going.
const RUNNING_TEXT: &str = "Training...";

/// Adds the systems that start training runs, show their progress and results, and save trained
/// models, plus the session's list of trained models.
pub struct TrainingPlugin;

impl Plugin for TrainingPlugin {
    fn build(self, app: gearbox::App) -> gearbox::App {
        app .add_resource(TrainedModels::default())
            .on_render_update(training_buttons)
            .on_render_update(poll_training)
    }
}

/// A model trained this session, with what it was trained on and how it scored.
///
/// Fields:
/// * number: Counts the session's models from 1, to tell them apart.
/// * design: Name of the design file the model was built from.
/// * data: Name of the csv file the model was trained and tested on.
/// * train_rows: How many rows the model was trained on.
/// * test_rows: How many rows the model was tested on.
/// * summary: How training went, scored on the training rows.
/// * test: The scores on the test rows, or `None` if there were none.
/// * bytes: The trained model, saved with postcard as `.tutor` files are.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainedModel {
    pub number: usize,
    pub design: String,
    pub data: String,
    pub train_rows: usize,
    pub test_rows: usize,
    pub summary: TrainingSummary,
    pub test: Option<Evaluation>,
    pub bytes: Vec<u8>
}

/// Every model trained this session, oldest first, shown in each TRAIN panel.
///
/// Fields:
/// * models: The trained models.
#[derive(Debug, Default, Resource)]
pub struct TrainedModels {
    models: Vec<Arc<TrainedModel>>
}

/// What a TRAIN panel's training run is doing, shared between the panel and the task running it.
///
/// Fields:
/// * running: Whether a run is going, so only one runs at a time.
/// * stop: Set by the stop button to stop the current run after its current epoch.  Each run
///   gets a fresh flag, so stopping one never stops the next.
/// * lines: The run's log, newest last.
/// * changed: Whether `lines` or `running` changed since the panel last showed them.
/// * finished: A model the run finished, waiting to be added to `TrainedModels` and shown.
#[derive(Debug, Default)]
struct RunState {
    running: bool,
    stop: Arc<AtomicBool>,
    lines: Vec<String>,
    changed: bool,
    finished: Option<TrainedModel>
}

/// A training run's state, shared with the task running it.
type SharedRun = Arc<Mutex<RunState>>;

/// Marks the TRAIN button, which opens and closes the `TrainPanel` of its overlay.
#[derive(Debug, Clone, Copy, Component)]
struct TrainToggle;

/// Marks the training panel.
///
/// Fields:
/// * run: The state of the panel's training run.
#[derive(Debug, Clone, Component)]
struct TrainPanel {
    run: SharedRun
}

/// Marks the button starting a training run.
#[derive(Debug, Clone, Copy, Component)]
struct StartTraining;

/// Marks the button stopping the running training run, only shown while one runs.
#[derive(Debug, Clone, Copy, Component)]
struct StopTraining;

/// Marks the text box showing the training log.
#[derive(Debug, Clone, Copy, Component)]
struct TrainLog;

/// Marks the list of the session's trained models.
///
/// Fields:
/// * shown: How many models the list shows, so it is rebuilt when more are trained.
#[derive(Debug, Clone, Copy, Component)]
struct TrainHistory {
    shown: usize
}

/// Marks a button in the trained model list that shows its model's results.
///
/// Fields:
/// * model: The model whose results are shown.
#[derive(Debug, Clone, Component)]
struct ShowResults {
    model: Arc<TrainedModel>
}

/// Marks a results popup, removed with all it holds when closed.
#[derive(Debug, Clone, Copy, Component)]
struct ResultsPopup;

/// Marks a results popup's close button.
#[derive(Debug, Clone, Copy, Component)]
struct ClosePopup;

/// Marks a results popup's SAVE button.
///
/// Fields:
/// * model: The model the button saves.
#[derive(Debug, Clone, Component)]
struct SaveModel {
    model: Arc<TrainedModel>
}

/// Builds the TRAIN button for the overlay's button bar.
pub fn train_button() -> UiBundle {
    themed::button("TRAIN").with(TrainToggle)
}

/// Builds the closed training panel: start and stop buttons, the log of the current run, and the
/// session's trained models.
pub fn train_panel() -> UiBundle {
    let stop = themed::button_tone("Stop", Tone::Danger)
        .style(|node| node.display(Display::None))
        .with(StopTraining);
    let header = themed::row()
        .style(|node| node
            .width(Val::Percent(1.0))
            .justify(JustifyContent::SpaceBetween))
        .child(themed::strong("Training"))
        .child(themed::row()
            .child(stop)
            .child(themed::button_tone(START_TEXT, Tone::Primary).with(StartTraining)));
    let history = themed::column()
        .with(TrainHistory { shown: 0 })
        .child(themed::caption("No models trained yet.", Vec4::new(0.6, 0.6, 0.6, 1.0)));

    themed::frame(Some(PANEL_WIDTH))
        .style(|node| node.display(Display::None))
        .with(OverlayPanel)
        .with(TrainPanel { run: SharedRun::default() })
        .child(header)
        .child(themed::text_box("Pick a csv file to train this design on.", LOG_HEIGHT).with(TrainLog))
        .child(themed::separator())
        .child(themed::strong("Past models"))
        .child(history)
}

/// Opens the training panel, starts and stops training runs, shows past models' results, and
/// closes and saves from results popups, when their buttons are clicked.
fn training_buttons(world: World, activated: Event<Activate>, mut editor: ResMut<DeltosEditor>) {
    for Activate { entity } in activated.read() {
        if has::<TrainToggle>(&world, entity) {
            toggle_panel::<TrainPanel>(&world, entity);
        } else if has::<StartTraining>(&world, entity) {
            start_training(&world, entity, &mut editor);
        } else if has::<StopTraining>(&world, entity) {
            stop_training(&world, entity);
        } else if let Some(ShowResults { model }) = component(&world, entity) {
            show_results(&world, entity, model);
        } else if has::<ClosePopup>(&world, entity) {
            close_popup(&world, entity);
        } else if let Some(SaveModel { model }) = component(&world, entity) {
            save_model(model, &editor);
        }
    }
}

/// Starts a training run of the design drawn around `entity`: asks for a csv file, then trains
/// and tests on it in the background, logging progress to the training panel.  Does nothing
/// while the panel's last run is still going.
fn start_training(world: &World, entity: EntityID, editor: &mut DeltosEditor) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    let Some(TrainPanel { run }) = world
        .hierarchy()
        .ancestors(entity)
        .find_map(|id| component::<TrainPanel>(world, id)) else { return };
    if lock(&run).running { return }

    // save the design as drawn, including its schema, to train a copy of it
    let Some(GraphRoot { graph, .. }) = world
        .hierarchy()
        .ancestors(overlay)
        .find_map(|id| component::<GraphRoot>(world, id)) else { return };
    let design = match graph.save() {
        Ok(design) => design,
        Err(err) => return editor.notify(NotifyLevel::Error, format!("Couldn't read the design to train it: {err:#}"))
    };
    let design_name = world
        .hierarchy()
        .ancestors(overlay)
        .find_map(|id| component::<OpenFile>(world, id))
        .map_or_else(|| "design".to_string(), |file| file_name(&file.path));

    // run it in the background, the run's state passing progress back to the panel
    let stop = start_run(&run);
    editor.tasks.spawn(Box::pin(async move {
        let result = pick_and_train(&design, &design_name, &run, &stop).await;
        let apply: TaskResult = Box::new(move |_: &World, editor: &mut DeltosEditor| {
            let mut state = lock(&run);
            match result {
                Ok(Some(model)) => state.finished = Some(model),
                Ok(None) => push_line(&mut state, "No file picked, training cancelled.".into()),
                Err(err) => {
                    push_line(&mut state, format!("Training failed: {err:#}"));
                    editor.notify(NotifyLevel::Error, format!("Training failed: {err:#}"));
                }
            }
            state.running = false;
            state.changed = true;
        });
        apply
    }));
}

/// Asks the panel of `entity`'s running training run to stop after its current epoch.
fn stop_training(world: &World, entity: EntityID) {
    let Some(TrainPanel { run }) = world
        .hierarchy()
        .ancestors(entity)
        .find_map(|id| component::<TrainPanel>(world, id)) else { return };
    let mut state = lock(&run);
    if !state.running || state.stop.swap(true, Ordering::Relaxed) { return }
    push_line(&mut state, "Stopping after this epoch...".into());
}

/// Asks the user for a csv file, then trains the design on it until it plateaus or `stop` is
/// set, or gives `None` if no file was picked.
async fn pick_and_train(design: &str, design_name: &str, run: &SharedRun, stop: &AtomicBool) -> anyhow::Result<Option<TrainedModel>> {
    let Some(file) = rfd::AsyncFileDialog::new()
        .set_title("Train on CSV")
        .add_filter("CSV", &["csv"])
        .pick_file()
        .await else { return Ok(None) };
    let data_name = file.file_name();
    log(run, format!("Reading {data_name}"));
    let csv = String::from_utf8(file.read().await).with_context(|| format!("{data_name} isn't utf-8 text"))?;
    train(design, design_name, &data_name, &csv, stop, |line| log(run, line)).map(Some)
}

/// Builds a design and trains it on csv rows until its loss stops improving or `stop` is set,
/// then tests it.  A stopped model keeps what it learned and is tested like any other.
///
/// Rows alternate between training and testing, as the iris trainer splits them.  Each step and
/// every `REPORT_EVERY` epochs of progress is passed to `log` as a line of text.
///
/// # Parameters
/// * design: The design's json, as `ModelDesigner::serialize` writes it.
/// * design_name: Name of the design's file, kept with the model.
/// * data_name: Name of the csv file, kept with the model.
/// * csv: The csv text, starting with a header row.
/// * stop: Set from any thread to stop training after the current epoch.
/// * log: Called with each line of progress.
///
/// # Errors
/// * The design can't be read, or its schema has no inputs, no outputs, or unnamed columns.
/// * The csv doesn't fit the schema, see `DataSheet::from_csv`.
/// * There are too few rows to train on, or the design can't be built or trained on them.
/// * The trained model can't be saved.
pub fn train(
    design: &str,
    design_name: &str,
    data_name: &str,
    csv: &str,
    stop: &AtomicBool,
    mut log: impl FnMut(String)
) -> anyhow::Result<TrainedModel> {
    let designer = ModelDesigner::deserialize(design).context("the design can't be read")?;
    let schema = designer
        .schema()
        .clone();
    anyhow::ensure!(!schema.inputs.is_empty(), "the schema has no inputs, add some under SCHEMA");
    anyhow::ensure!(!schema.outputs.is_empty(), "the schema has no outputs, add some under SCHEMA");
    anyhow::ensure!(
        schema.inputs
            .iter()
            .chain(&schema.outputs)
            .all(|entry| !entry.name.trim().is_empty()),
        "every schema column needs a name"
    );

    // read the rows, then split them alternately into training and test rows
    let sheet = DataSheet::from_csv(schema.to_schema(), csv).with_context(|| format!("{data_name} doesn't fit the schema"))?;
    let mut train_rows = DataSheet::from_schema(schema.to_schema());
    let mut test_rows = DataSheet::from_schema(schema.to_schema());
    for (i, row) in sheet.iter().enumerate() {
        if i % 2 == 0 { train_rows.push(row)?; } else { test_rows.push(row)?; }
    }
    let (train_count, test_count) = (train_rows.iter().count(), test_rows.iter().count());
    anyhow::ensure!(train_count > 0, "{data_name} has no rows to train on");
    log(format!("Training on {train_count} rows, testing on {test_count}"));

    // build and train the model
    let mut model = designer
        .build(&train_rows)
        .context("the design can't be built for this data")?;
    let summary = model.train_with_progress(
        &train_rows,
        LEARNING_RATE,
        BATCH_SIZE,
        MIN_IMPROVEMENT,
        PATIENCE,
        REPORT_EVERY,
        stop,
        |progress| log(progress_line(progress))
    )?;
    let reason = if summary.stopped { "Stopped early" } else { "Stopped" };
    log(format!("{reason} at epoch {} after {}", summary.epochs, seconds(summary.duration)));

    // test it on the rows it didn't see, then keep it saved
    let test = match test_count {
        0 => None,
        _ => Some(model.evaluate(&test_rows)?)
    };
    let bytes = postcard::to_allocvec(&model).context("the trained model can't be saved")?;
    Ok(TrainedModel {
        number: 0,
        design: design_name.to_string(),
        data: data_name.to_string(),
        train_rows: train_count,
        test_rows: test_count,
        summary,
        test,
        bytes
    })
}

/// Shows each training panel's latest log and start button, adds finished models to the session
/// and shows their results, and rebuilds the lists of trained models when more are added.
fn poll_training(world: World, mut models: ResMut<TrainedModels>) {
    let mut panels = Query::<&TrainPanel>::new(&world);
    let mut found = Vec::new();
    while let Ok(Some((id, panel))) = panels.next() {
        found.push((id, panel.run.clone()));
    }
    drop(panels);

    for (panel, run) in found {
        // take a finished model into the session and show its results
        let finished = lock(&run).finished.take();
        if let Some(mut model) = finished {
            model.number = models.models.len() + 1;
            let model = Arc::new(model);
            models.models.push(model.clone());
            log(&run, format!("Done, saved as model #{}", model.number));
            show_results(&world, panel, model);
        }

        // show the log and whether a run is going
        let (changed, text, running) = {
            let mut state = lock(&run);
            let changed = std::mem::take(&mut state.changed);
            (changed, state.lines.join("\n"), state.running)
        };
        if !changed { continue }
        for id in world.hierarchy().descendants(panel).collect::<Vec<_>>() {
            if has::<TrainLog>(&world, id) {
                set_label(&world, id, &text);
            } else if has::<StartTraining>(&world, id) {
                set_label(&world, id, if running { RUNNING_TEXT } else { START_TEXT });
            } else if has::<StopTraining>(&world, id)
                && let Ok(Some(mut node)) = world.handle(id).get::<&mut Node>() {
                node.style_mut().set_display(if running { Display::Flex } else { Display::None });
            }
        }
    }

    // rebuild the lists that don't show every trained model
    let mut lists = Query::<&TrainHistory>::new(&world);
    let mut stale = Vec::new();
    while let Ok(Some((id, list))) = lists.next() {
        if list.shown != models.models.len() { stale.push(id); }
    }
    drop(lists);
    for list in stale {
        rebuild_history(&world, list, &models.models);
    }
}

/// Replaces the rows of the trained model list `list` with a button per model, newest first.
fn rebuild_history(world: &World, list: EntityID, models: &[Arc<TrainedModel>]) {
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
    for model in models.iter().rev() {
        let row = themed::button(history_label(model)).with(ShowResults { model: model.clone() });
        world.spawn_with_parent(Some(list), row.into_entity_builder());
    }
    if let Ok(Some(mut history)) = world.handle(list).get::<&mut TrainHistory>() {
        history.shown = models.len();
    }
}

/// Shows a popup with `model`'s results in the middle of the window, over the overlay holding
/// `entity`.
fn show_results(world: &World, entity: EntityID, model: Arc<TrainedModel>) {
    let Some(overlay) = overlay_of(world, entity) else { return };
    world.spawn_with_parent(Some(overlay), results_popup(model).into_entity_builder());
}

/// Builds a results popup: a dimmed backdrop over the whole window, holding a box in its middle
/// with the training results over the close and SAVE buttons.
fn results_popup(model: Arc<TrainedModel>) -> UiBundle {
    let rows = result_lines(&model)
        .into_iter()
        .map(|(name, value)| themed::row()
            .style(|node| node.align(AlignItems::Start))
            .child(themed::legend(name, POPUP_LEGEND_WIDTH))
            .child(UiBundle::new(Node::new()
                .grow(1.0)
                .shrink(1.0))
                .with(Label::new(value).wrapped())));
    let buttons = themed::row()
        .style(|node| node
            .width(Val::Percent(1.0))
            .justify(JustifyContent::End))
        .child(themed::button("Close").with(ClosePopup))
        .child(themed::button_tone("SAVE", Tone::Primary).with(SaveModel { model: model.clone() }));
    let body = themed::frame(Some(POPUP_WIDTH))
        .child(themed::heading(format!("Model #{} trained", model.number)))
        .child(themed::separator())
        .children(rows)
        .child(themed::separator())
        .child(buttons);

    // the backdrop takes clicks and right clicks, so the graph under it isn't changed
    UiBundle::new(Node::new()
        .fixed(Rect::new(px(0.0), px(0.0), px(0.0), px(0.0)))
        .z_index(POPUP_Z)
        .justify(JustifyContent::Center)
        .align(AlignItems::Center)
        .background(Vec4::new(0.0, 0.0, 0.0, 0.45)))
        .interactive()
        .with(ContextMenu { disabled: true })
        .with(ResultsPopup)
        .child(body)
}

/// Removes the results popup holding `entity`.
fn close_popup(world: &World, entity: EntityID) {
    let popup = world
        .hierarchy()
        .ancestors(entity)
        .find(|id| has::<ResultsPopup>(world, *id));
    if let Some(popup) = popup {
        world.handle(popup).despawn_recursive();
    }
}

/// Asks where to save `model` as a `.tutor` file in the background, then writes it there.
fn save_model(model: Arc<TrainedModel>, editor: &DeltosEditor) {
    editor.tasks.spawn(Box::pin(async move {
        let result = pick_and_save(&model).await;
        let apply: TaskResult = Box::new(move |_: &World, editor: &mut DeltosEditor| match result {
            Ok(Some(name)) => editor.notify(NotifyLevel::Info, format!("Saved model #{} as {name}", model.number)),
            Ok(None) => {}
            Err(err) => editor.notify(NotifyLevel::Error, format!("Couldn't save model #{}: {err:#}", model.number))
        });
        apply
    }));
}

/// Asks where to save `model`, then writes it there, giving the file's name or `None` if no
/// file was picked.
async fn pick_and_save(model: &TrainedModel) -> anyhow::Result<Option<String>> {
    let Some(file) = rfd::AsyncFileDialog::new()
        .set_title("Save model")
        .add_filter("Tutor model", &["tutor"])
        .set_file_name(model_file_name(model))
        .save_file()
        .await else { return Ok(None) };
    file.write(&model.bytes)
        .await
        .with_context(|| format!("couldn't write {}", file.file_name()))?;
    Ok(Some(file.file_name()))
}

/// The name a model is offered to be saved as: its design's name and number, such as
/// `iris-2.tutor`.
fn model_file_name(model: &TrainedModel) -> String {
    let stem = Path::new(&model.design)
        .file_stem()
        .map_or_else(|| "model".to_string(), |stem| stem
            .to_string_lossy()
            .into_owned());
    format!("{stem}-{}.tutor", model.number)
}

/// The names and values a results popup lists for `model`.
fn result_lines(model: &TrainedModel) -> Vec<(String, String)> {
    let test = model.test
        .as_ref()
        .map_or_else(|| "No test rows".to_string(), Evaluation::to_string);
    vec![
        ("Design".into(), model.design.clone()),
        ("Data".into(), model.data.clone()),
        ("Rows".into(), format!("{} training, {} testing", model.train_rows, model.test_rows)),
        ("Training time".into(), seconds(model.summary.duration)),
        ("Epochs".into(), match model.summary.stopped {
            true => format!("{} (stopped early)", model.summary.epochs),
            false => model.summary.epochs.to_string()
        }),
        ("Final loss".into(), format!("{:.4}", model.summary.best_loss)),
        ("Training scores".into(), model.summary.evaluation.to_string()),
        ("Test scores".into(), test),
        ("Model size".into(), format!("{} bytes", model.bytes.len()))
    ]
}

/// The text of `model`'s button in the trained model list: its number, data, and scores, the
/// test scores if it was tested.
fn history_label(model: &TrainedModel) -> String {
    let scores = model.test
        .as_ref()
        .unwrap_or(&model.summary.evaluation);
    format!("#{}  {}  {}", model.number, model.data, scores)
}

/// A training log line for a progress report.
fn progress_line(progress: &TrainingProgress) -> String {
    format!(
        "Epoch {}  loss {:.4}  {}  ({})",
        progress.epoch, progress.loss, progress.evaluation, seconds(progress.elapsed)
    )
}

/// `duration` in seconds to two decimals, such as `1.25 s`.
fn seconds(duration: Duration) -> String {
    format!("{:.2} s", duration.as_secs_f32())
}

/// The last part of `path`, or all of it if it has none.
fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Clears `run`'s log and marks it running with a fresh stop flag, which it gives for the run's
/// training loop to check.
fn start_run(run: &SharedRun) -> Arc<AtomicBool> {
    let mut state = lock(run);
    state.running = true;
    state.stop = Arc::new(AtomicBool::new(false));
    state.lines.clear();
    state.finished = None;
    push_line(&mut state, "Choose a csv file to train on...".into());
    state.stop.clone()
}

/// Adds `line` to `run`'s log.
fn log(run: &SharedRun, line: String) {
    push_line(&mut lock(run), line);
}

/// Adds `line` to the end of `state`'s log, dropping the oldest lines past `LOG_LINES`.
fn push_line(state: &mut RunState, line: String) {
    state.lines.push(line);
    let extra = state.lines
        .len()
        .saturating_sub(LOG_LINES);
    state.lines.drain(.. extra);
    state.changed = true;
}

/// Sets the text of every label in and below `entity` to `text`.
fn set_label(world: &World, entity: EntityID, text: &str) {
    let ids: Vec<EntityID> = std::iter::once(entity)
        .chain(world
            .hierarchy()
            .descendants(entity))
        .collect();
    for id in ids {
        if let Ok(Some(mut label)) = world.handle(id).get::<&mut Label>() {
            label.text = text.to_string();
        }
    }
}

/// Locks `run`, still giving the state if a thread panicked while holding it.
fn lock(run: &SharedRun) -> MutexGuard<'_, RunState> {
    run.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tutor_core::{
        data::{DataType, DesignSchema, SchemaEntry},
        layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights}
    };

    /// A design reading `x` and `y` to predict the enum `label`, with `schema` swapped in if given.
    fn design(schema: Option<DesignSchema>) -> anyhow::Result<String> {
        let entry = |name: &str, data_type| SchemaEntry { name: name.into(), data_type };
        let mut designer = ModelDesigner::new()
            .add_layer(InputSchema::new())?
            .add_layer(Weights::random(2, 1e-2))?
            .add_layer(OutputSchema::new())?;
        designer.set_schema(schema.unwrap_or_else(|| DesignSchema {
            inputs: vec![entry("x", DataType::Decimal), entry("y", DataType::Decimal)],
            outputs: vec![entry("label", DataType::Enum)]
        }));
        designer.serialize()
    }

    /// Csv rows where `label` is `high` when `x` is large and `low` otherwise.
    fn csv() -> String {
        let mut csv = String::from("x,y,label\n");
        for i in 0 .. 40 {
            let x = i as f32 / 4.0;
            let label = if x > 5.0 { "high" } else { "low" };
            csv.push_str(&format!("{x},{},{label}\n", (i % 7) as f32));
        }
        csv
    }

    #[test]
    fn designs_train_and_test_on_alternate_rows() -> anyhow::Result<()> {
        let mut lines = Vec::new();

        let model = train(&design(None)?, "test.tutordesign", "data.csv", &csv(), &AtomicBool::new(false), |line| lines.push(line))?;

        assert_eq!((model.train_rows, model.test_rows), (20, 20));
        assert!(model.summary.epochs > 0);
        assert!(model.test.is_some());
        assert!(!model.bytes.is_empty());
        assert_eq!(lines.first().map(String::as_str), Some("Training on 20 rows, testing on 20"));
        assert!(lines.last().is_some_and(|line| line.starts_with("Stopped at epoch")));
        Ok(())
    }

    #[test]
    fn setting_the_stop_flag_stops_after_the_current_epoch() -> anyhow::Result<()> {
        let mut lines = Vec::new();

        let model = train(&design(None)?, "d", "c", &csv(), &AtomicBool::new(true), |line| lines.push(line))?;

        assert_eq!(model.summary.epochs, 1);
        assert!(model.summary.stopped);
        assert!(model.test.is_some());
        assert!(lines.last().is_some_and(|line| line.starts_with("Stopped early at epoch 1")));
        assert!(result_lines(&model).contains(&("Epochs".into(), "1 (stopped early)".into())));
        Ok(())
    }

    #[test]
    fn stopping_only_flags_a_running_run_and_each_run_gets_a_fresh_flag() {
        let run = SharedRun::default();
        let stop = start_run(&run);

        stop.store(true, Ordering::Relaxed);
        let next = start_run(&run);

        assert!(!next.load(Ordering::Relaxed));
        assert!(stop.load(Ordering::Relaxed));
    }

    #[test]
    fn trained_models_load_back_from_their_bytes() -> anyhow::Result<()> {
        let model = train(&design(None)?, "test.tutordesign", "data.csv", &csv(), &AtomicBool::new(false), |_| {})?;

        let loaded: tutor_core::model::Model = postcard::from_bytes(&model.bytes)?;

        assert_eq!(loaded.nodes().len(), 3);
        Ok(())
    }

    #[test]
    fn designs_without_a_full_schema_are_refused() -> anyhow::Result<()> {
        let entry = |name: &str| SchemaEntry { name: name.into(), data_type: DataType::Decimal };
        let no_outputs = DesignSchema { inputs: vec![entry("x")], outputs: Vec::new() };
        let unnamed = DesignSchema { inputs: vec![entry(" ")], outputs: vec![entry("label")] };

        assert!(train(&design(Some(DesignSchema::default()))?, "d", "c", &csv(), &AtomicBool::new(false), |_| {}).is_err());
        assert!(train(&design(Some(no_outputs))?, "d", "c", &csv(), &AtomicBool::new(false), |_| {}).is_err());
        assert!(train(&design(Some(unnamed))?, "d", "c", &csv(), &AtomicBool::new(false), |_| {}).is_err());
        Ok(())
    }

    #[test]
    fn csvs_missing_schema_columns_are_refused() -> anyhow::Result<()> {
        assert!(train(&design(None)?, "d", "c", "x,label\n1,low\n2,high\n", &AtomicBool::new(false), |_| {}).is_err());
        Ok(())
    }

    #[test]
    fn the_log_keeps_only_the_newest_lines() {
        let mut state = RunState::default();

        for i in 0 .. LOG_LINES + 3 {
            push_line(&mut state, i.to_string());
        }

        assert_eq!(state.lines.len(), LOG_LINES);
        assert_eq!(state.lines.first().map(String::as_str), Some("3"));
        assert!(state.changed);
    }

    #[test]
    fn models_are_offered_to_be_saved_by_design_and_number() -> anyhow::Result<()> {
        let mut model = train(&design(None)?, "iris.tutordesign", "data.csv", &csv(), &AtomicBool::new(false), |_| {})?;
        model.number = 2;

        assert_eq!(model_file_name(&model), "iris-2.tutor");
        assert!(history_label(&model).starts_with("#2  data.csv  "));
        assert_eq!(result_lines(&model).len(), 9);
        Ok(())
    }
}
