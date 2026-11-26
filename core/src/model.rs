use std::{collections::VecDeque, fmt, time::Instant};

use ahash::{HashMap, HashMapExt};
use anyhow::Context;
use getset::Getters;
use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix2, s};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{data::{DataSheet, DataStorage}, designer::ModelDesigner, layers::Layer};

/// A trained or trainable network of layers that predicts the output columns of a data sheet row
/// from its input columns.
///
/// Input nodes (nodes with no inputs) are filled from the data sheet rows, then every other node
/// runs once all of its inputs are filled, until every output node (a node no other node reads
/// from) is filled.  The order the nodes run in is worked out once when the model is made or
/// loaded, so a network that could never fill its outputs fails then instead of mid run.
///
/// Fields:
/// * nodes: Every node of the network, keyed by its ID.
/// * order: The node IDs in an order where every node comes after all of its inputs.  Worked out
///   from `nodes` and never saved.
/// * outputs: The IDs of the output nodes, sorted so their results are always joined in the same
///   order.  Worked out from `nodes` and never saved.
#[derive(Getters, Serialize, Deserialize)]
#[serde(try_from = "SavedModel")]
pub struct Model {
    #[getset(get = "pub")]
    nodes: HashMap<Uuid, ModelNode>,
    #[serde(skip)]
    #[getset(get = "pub")]
    order: Vec<Uuid>,
    #[serde(skip)]
    #[getset(get = "pub")]
    outputs: Vec<Uuid>
}

/// The saved form of a [`Model`], its nodes without the run order that is worked out on load.
///
/// Fields:
/// * nodes: Every node of the network, keyed by its ID.
#[derive(Deserialize)]
struct SavedModel {
    nodes: HashMap<Uuid, ModelNode>
}

impl TryFrom<SavedModel> for Model {
    type Error = anyhow::Error;

    /// Works out the run order of a loaded model's nodes.
    fn try_from(saved: SavedModel) -> anyhow::Result<Self> {
        Model::new(saved.nodes)
    }
}

/// A node inside a model's network, a layer and the nodes whose outputs feed it.
///
/// Fields:
/// * layer: The layer this node runs.
/// * inputs: The nodes whose outputs are joined side by side, in this order, to make this layer's
///   input.  Empty for an input node, which reads data sheet rows instead.
#[derive(Getters, Serialize, Deserialize)]
pub struct ModelNode {
    #[getset(get = "pub")]
    layer: Box<dyn Layer>,
    #[getset(get = "pub")]
    inputs: Vec<Uuid>
}

impl ModelNode {
    /// Creates a node that runs `layer` on the joined outputs of `inputs`.
    ///
    /// # Parameters
    /// * layer: The layer this node runs.
    /// * inputs: The nodes feeding this one, joined in this order.  Empty for an input node.
    pub fn new(layer: Box<dyn Layer>, inputs: Vec<Uuid>) -> Self {
        Self { layer, inputs }
    }
}

/// The rows of a data sheet encoded into the numbers a [`Model`] works with, made once by
/// `Model::encode` and reused across training epochs and accuracy checks.
///
/// Fields:
/// * rows: Number of data sheet rows encoded.
/// * inputs: The encoded rows for every input node, keyed by node ID.
/// * targets: The encoded expected outputs for every output node, keyed by node ID.
#[derive(Getters, Clone, Debug)]
pub struct EncodedSheet {
    #[getset(get = "pub")]
    rows: usize,
    #[getset(get = "pub")]
    inputs: HashMap<Uuid, Array2<f32>>,
    #[getset(get = "pub")]
    targets: HashMap<Uuid, Array2<f32>>
}

impl EncodedSheet {
    /// Picks out the given rows of every input and target, in the given order.
    fn select(&self, rows: &[usize]) -> Self {
        let pick = |map: &HashMap<Uuid, Array2<f32>>| map
            .iter()
            .map(|(id, array)| (*id, array.select(Axis(0), rows)))
            .collect();

        Self { rows: rows.len(), inputs: pick(&self.inputs), targets: pick(&self.targets) }
    }
}

/// How well a model predicts one output column: the fraction of rows predicted correctly for an
/// enum column, or the mean absolute error in the column's units for a decimal or integer column.
///
/// Variants:
/// * Accuracy: The `column`'s name and the `accuracy`, from 0 to 1, of an enum column.
/// * MeanAbsoluteError: The `column`'s name and the average `error`, in the column's units, of a
///   decimal or integer column.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnScore {
    Accuracy { column: String, accuracy: f32 },
    MeanAbsoluteError { column: String, error: f32 }
}

impl fmt::Display for ColumnScore {
    /// Writes the column's name and score, accuracy as a percentage.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accuracy { column, accuracy } => write!(f, "{column} accuracy {:.2}%", accuracy * 100.0),
            Self::MeanAbsoluteError { column, error } => write!(f, "{column} mean absolute error {error:.3}")
        }
    }
}

/// The scores of every output column of a model over a set of rows, from `Model::evaluate`.
///
/// Fields:
/// * columns: The score of every output column, output node by output node.
#[derive(Getters, Clone, Debug, PartialEq)]
pub struct Evaluation {
    #[getset(get = "pub")]
    columns: Vec<ColumnScore>
}

impl fmt::Display for Evaluation {
    /// Writes every column's score, separated by commas.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, score) in self.columns.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{score}")?;
        }
        Ok(())
    }
}

/// Every node's output for one batch, keyed by node ID.
type NodeOutputs = HashMap<Uuid, Array2<f32>>;

impl Model {

    /// Creates a model from a network of nodes and works out the order they run in.
    ///
    /// The nodes are ordered so every node comes after all of its inputs, starting from the
    /// nodes with no inputs.  Nodes that are never reached that way sit on or after a cycle and
    /// could never be filled, so the network is rejected.
    ///
    /// # Parameters
    /// * nodes: Every node of the network, keyed by its ID.
    ///
    /// # Errors
    /// * `nodes` is empty.
    /// * A node reads from an ID that is not in `nodes`.
    /// * Some nodes could not be filled because they depend on a cycle.
    pub fn new(nodes: HashMap<Uuid, ModelNode>) -> anyhow::Result<Self> {
        let (order, outputs) = run_order(&nodes)?;
        Ok(Self { nodes, order, outputs })
    }

    /// Creates a new empty `ModelDesigner`, the way to lay out a model's network before
    /// building it.
    pub fn designer() -> ModelDesigner {
        ModelDesigner::new()
    }

    /// Prepares every node's layer for training on some sample data, in run order so each layer
    /// is sized from the total width of its inputs.
    ///
    /// # Parameters
    /// * sample_data: Rows the model is fitted to and will be trained with.
    ///
    /// # Errors
    /// * A layer cannot be prepared from the sample data.
    pub(crate) fn prepare_train(&mut self, sample_data: &DataSheet) -> anyhow::Result<()> {
        let mut widths: HashMap<Uuid, usize> = HashMap::with_capacity(self.nodes.len());
        for id in &self.order {
            let node = self
                .nodes
                .get_mut(id)
                .with_context(|| format!("node {id} is missing"))?;
            let width = node
                .inputs
                .iter()
                .map(|input| widths
                    .get(input)
                    .copied()
                    .with_context(|| format!("node {input} was not prepared before node {id}")))
                .sum::<anyhow::Result<usize>>()?;

            node.layer
                .prepare_train(sample_data, width)
                .with_context(|| format!("failed to prepare node {id}"))?;
            widths.insert(*id, node.layer.output_size(width));
        }
        Ok(())
    }

    /// Encodes the inputs and expected outputs of every row of a data sheet, so training and
    /// accuracy checks can reuse them instead of encoding the rows again every epoch.
    ///
    /// Every input node encodes the rows' inputs, and every output node encodes the rows'
    /// expected outputs.
    ///
    /// # Parameters
    /// * data: The rows to encode.
    ///
    /// # Errors
    /// * `data` has no rows.
    /// * An input node is not an `InputSchema` or an output node is not an `OutputSchema`.
    /// * A row cannot be encoded.
    pub fn encode(&self, data: &DataSheet) -> anyhow::Result<EncodedSheet> {
        let rows = data
            .iter()
            .collect::<Vec<_>>();
        anyhow::ensure!(!rows.is_empty(), "no rows to encode");

        // expected outputs for every output node
        let mut targets = HashMap::with_capacity(self.outputs.len());
        for id in &self.outputs {
            let encoded = self
                .node(id)?
                .layer
                .encode_targets(&rows)
                .with_context(|| format!("failed to encode targets for node {id}"))?;
            targets.insert(*id, encoded);
        }

        Ok(EncodedSheet { rows: rows.len(), inputs: self.encode_inputs(&rows)?, targets })
    }

    /// Runs a data sheet row through the network and gives the output nodes' outputs joined in
    /// order: the probability of every option of every enum output column and the normalized
    /// value of every numeric one.
    ///
    /// # Parameters
    /// * row: The data sheet row to run, laid out in column index order.
    ///
    /// # Errors
    /// * The row cannot be encoded.
    /// * A node fails to run.
    pub fn find_probabilities(&self, row: &[DataStorage]) -> anyhow::Result<Box<[f32]>> {
        let outputs = self.forward(&self.encode_inputs(&[row])?)?;

        let mut probabilities = Vec::new();
        for id in &self.outputs {
            probabilities.extend(output_of(&outputs, id)?.iter().copied());
        }
        Ok(probabilities.into_boxed_slice())
    }

    /// Predicts the value of every output column for a data sheet row, gathered from every output
    /// node.  Each prediction is paired with the index of the column it predicts.
    ///
    /// # Parameters
    /// * row: The data sheet row to predict for, laid out in column index order.
    ///
    /// # Errors
    /// * The row cannot be encoded.
    /// * A node fails to run.
    /// * An output node cannot decode its outputs.
    pub fn predict(&self, row: &[DataStorage]) -> anyhow::Result<Vec<(usize, DataStorage)>> {
        let outputs = self.forward(&self.encode_inputs(&[row])?)?;

        let mut predictions = Vec::new();
        for id in &self.outputs {
            let decoded = self
                .node(id)?
                .layer
                .decode_outputs(output_of(&outputs, id)?.view())?
                .into_iter()
                .next()
                .with_context(|| format!("output node {id} gave no prediction for the row"))?;
            predictions.extend(decoded);
        }
        Ok(predictions)
    }

    /// Performs a training epoch over every row of a data sheet.  The average loss (miss) is
    /// returned at the end of the epoch, useful to see if training is giving better results.
    ///
    /// The rows are encoded first, so `train_encoded` is faster when training for many epochs.
    ///
    /// # Parameters
    /// * data: The rows to train on.
    /// * learning_rate: How far each step moves the parameters against their gradients.
    /// * batch_size: The number of rows per parameter step.
    ///
    /// # Errors
    /// * The rows cannot be encoded, see `encode`.
    /// * Training fails, see `train_encoded`.
    pub fn train(&mut self, data: &DataSheet, learning_rate: f32, batch_size: usize) -> anyhow::Result<f32> {
        let encoded = self.encode(data)?;
        self.train_encoded(&encoded, learning_rate, batch_size)
    }

    /// Perform a training epoch over every row of an encoded data sheet.  The rows are shuffled
    /// then split into batches of `batch_size` rows, and the parameters take one step per batch.
    /// The average loss (miss) over the epoch is returned.
    ///
    /// # Parameters
    /// * data: The encoded rows to train on, from `encode`.
    /// * learning_rate: How far each step moves the parameters against their gradients.
    /// * batch_size: The number of rows per parameter step.
    ///
    /// # Errors
    /// * `data` has no rows or `batch_size` is 0.
    /// * A node fails to run forwards or backwards, or to step its parameters.
    pub fn train_encoded(
        &mut self,
        data: &EncodedSheet,
        learning_rate: f32,
        batch_size: usize
    ) -> anyhow::Result<f32> {
        let rows = data.rows;
        anyhow::ensure!(rows > 0, "no rows to train on");
        anyhow::ensure!(batch_size > 0, "batch size must be at least 1");

        // visit the rows in a fresh random order every epoch
        let mut order = (0 .. rows).collect::<Vec<_>>();
        order.shuffle(&mut rand::thread_rng());

        // one step per batch of rows
        let mut total_loss = 0.0;
        for batch in order.chunks(batch_size) {
            total_loss += self.train_batch(&data.select(batch), learning_rate)?;
        }

        Ok(total_loss / rows as f32)
    }

    /// Runs one batch forward and backward and steps every layer's parameters against the
    /// batch's average gradient.  Gives the loss summed over the batch's rows and output nodes.
    fn train_batch(&mut self, batch: &EncodedSheet, learning_rate: f32) -> anyhow::Result<f32> {
        let outputs = self.forward(&batch.inputs)?;

        // every output node scores its outputs against the expected values
        let mut batch_loss = 0.0;
        let mut grads: NodeOutputs = HashMap::with_capacity(self.nodes.len());
        for id in &self.outputs {
            let targets = batch
                .targets
                .get(id)
                .with_context(|| format!("no targets for output node {id}"))?;
            let (loss, grad) = self
                .node(id)?
                .layer
                .output_loss(output_of(&outputs, id)?.view(), targets.view())
                .with_context(|| format!("failed to score output node {id}"))?;
            batch_loss += loss;
            grads.insert(*id, grad);
        }

        // backward pass in reverse run order, so a node has every gradient from the nodes it
        // feeds before it hands its own input gradient back to its inputs
        for id in self.order.iter().rev() {
            let grad = grads
                .remove(id)
                .with_context(|| format!("node {id} got no gradient"))?;
            let input_grad = {
                let layer_input = self.layer_input(id, &batch.inputs, &outputs)?;
                self.nodes
                    .get_mut(id)
                    .with_context(|| format!("node {id} is missing"))?
                    .layer
                    .back_propagate(layer_input.view(), output_of(&outputs, id)?.view(), grad.view())
                    .with_context(|| format!("backward pass failed at node {id}"))?
            };

            // split the input gradient into each input's columns, summing where an input feeds
            // more than one node
            let mut offset = 0;
            for input in &self.node(id)?.inputs {
                let width = output_of(&outputs, input)?.ncols();
                let part = input_grad.slice(s![.., offset .. offset + width]);
                match grads.get_mut(input) {
                    Some(total) => *total += &part,
                    None => { grads.insert(*input, part.to_owned()); }
                }
                offset += width;
            }
        }

        // step every layer's parameters once per batch
        let n = batch.rows as f32;
        for (id, node) in &mut self.nodes {
            node.layer
                .perform_epoch_step(learning_rate, n)
                .with_context(|| format!("failed to step node {id}"))?;
        }

        Ok(batch_loss)
    }

    /// Trains this model until its loss stops improving.
    ///
    /// Runs epochs until `max_epoch_no_improve` epochs in a row fail to lower the best loss by at
    /// least the `min_req_improve` fraction.  Progress is printed every 100 epochs, and the final
    /// loss, scores, and training time are printed when training stops.
    ///
    /// # Parameters
    /// * data: The rows to train on.
    /// * learning_rate: How far each step moves the parameters against their gradients.
    /// * batch_size: The number of rows per parameter step.
    /// * min_req_improve: The fraction the loss must drop by for an epoch to count as improving.
    /// * max_epoch_no_improve: How many epochs in a row may fail to improve before training stops.
    ///
    /// # Errors
    /// * The rows cannot be encoded, see `encode`.
    /// * An epoch fails, see `train_encoded`.
    /// * The model cannot be scored, see `evaluate_encoded`.
    pub fn train_until_stop_improvement(
        &mut self,
        data: &DataSheet,
        learning_rate: f32,
        batch_size: usize,
        min_req_improve: f32,
        max_epoch_no_improve: u32
    ) -> anyhow::Result<()> {
        let start_time = Instant::now();
        let encoded = self.encode(data)?;
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch = 0;

        loop {
            let loss = self.train_encoded(&encoded, learning_rate, batch_size)?;
            epoch += 1;

            // only a big enough drop resets the count of epochs without improvement
            if loss < best_loss * (1.0 - min_req_improve) {
                best_loss = loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            // report progress every 100 epochs
            if epoch % 100 == 0 {
                println!("Epoch {epoch}:  Loss {loss:.4}  {}", self.evaluate_encoded(&encoded)?);
            }

            // stop and report once the loss has plateaued
            if epochs_without_improvement >= max_epoch_no_improve {
                let end = start_time.elapsed();
                println!("Stopped at epoch {epoch}: loss plateaued at {best_loss:.4}, {}, training took {} ms", self.evaluate_encoded(&encoded)?, end.as_millis());
                break;
            }
        }

        Ok(())
    }

    /// Scores how well this model predicts each output column of a data sheet's rows, see
    /// `ColumnScore`.
    ///
    /// # Parameters
    /// * data: The rows to score against.
    ///
    /// # Errors
    /// * The rows cannot be encoded, see `encode`.
    /// * The model cannot be scored, see `evaluate_encoded`.
    pub fn evaluate(&self, data: &DataSheet) -> anyhow::Result<Evaluation> {
        let encoded = self.encode(data)?;
        self.evaluate_encoded(&encoded)
    }

    /// Scores how well this model predicts each output column of an encoded data sheet's rows,
    /// see `ColumnScore`.  Columns are listed output node by output node.
    ///
    /// # Parameters
    /// * data: The encoded rows to score against, from `encode`.
    ///
    /// # Errors
    /// * `data` has no rows or is missing the targets of an output node.
    /// * A node fails to run.
    /// * An output node cannot score its outputs.
    pub fn evaluate_encoded(&self, data: &EncodedSheet) -> anyhow::Result<Evaluation> {
        anyhow::ensure!(data.rows > 0, "no rows to evaluate");

        let outputs = self.forward(&data.inputs)?;
        let mut columns = Vec::new();
        for id in &self.outputs {
            let targets = data
                .targets
                .get(id)
                .with_context(|| format!("no targets for output node {id}"))?;
            let scores = self
                .node(id)?
                .layer
                .evaluate(output_of(&outputs, id)?.view(), targets.view())
                .with_context(|| format!("failed to score output node {id}"))?;
            columns.extend(scores);
        }

        Ok(Evaluation { columns })
    }

    /// Encodes data sheet rows for every input node.
    fn encode_inputs(&self, rows: &[&[DataStorage]]) -> anyhow::Result<NodeOutputs> {
        let mut inputs = HashMap::new();
        for (id, node) in self.nodes.iter().filter(|(_, node)| node.inputs.is_empty()) {
            let encoded = node
                .layer
                .encode_inputs(rows)
                .with_context(|| format!("failed to encode inputs for input node {id}"))?;
            inputs.insert(*id, encoded);
        }
        Ok(inputs)
    }

    /// Runs a batch of encoded inputs through the network in run order and gives every node's
    /// output.
    fn forward(&self, inputs: &NodeOutputs) -> anyhow::Result<NodeOutputs> {
        let mut outputs: NodeOutputs = HashMap::with_capacity(self.nodes.len());
        for id in &self.order {
            let output = self
                .node(id)?
                .layer
                .transform(self.layer_input(id, inputs, &outputs)?.view())
                .with_context(|| format!("forward pass failed at node {id}"))?;
            outputs.insert(*id, output);
        }
        Ok(outputs)
    }

    /// The input a node's layer sees: its encoded rows for an input node, otherwise its inputs'
    /// outputs joined side by side.  A single input is borrowed rather than copied.
    fn layer_input<'a>(
        &self,
        id: &Uuid,
        inputs: &'a NodeOutputs,
        outputs: &'a NodeOutputs
    ) -> anyhow::Result<CowArray<'a, f32, Ix2>> {
        let node = self.node(id)?;
        match node.inputs.as_slice() {
            [] => Ok(inputs
                .get(id)
                .with_context(|| format!("no encoded rows for input node {id}"))?
                .view()
                .into()),
            [input] => Ok(output_of(outputs, input)?.view().into()),
            many => {
                let views = many
                    .iter()
                    .map(|input| output_of(outputs, input).map(Array2::view))
                    .collect::<anyhow::Result<Vec<ArrayView2<f32>>>>()?;
                let joined = ndarray::concatenate(Axis(1), &views)
                    .with_context(|| format!("inputs of node {id} have different row counts"))?;
                Ok(joined.into())
            }
        }
    }

    /// The node with the given ID.
    fn node(&self, id: &Uuid) -> anyhow::Result<&ModelNode> {
        self.nodes
            .get(id)
            .with_context(|| format!("node {id} is missing"))
    }
}

/// A node's output from a batch's outputs.
fn output_of<'a>(outputs: &'a NodeOutputs, id: &Uuid) -> anyhow::Result<&'a Array2<f32>> {
    outputs
        .get(id)
        .with_context(|| format!("node {id} was not filled before it was needed"))
}

/// Orders a network's nodes so every node comes after all of its inputs, and finds its output
/// nodes.  Nodes are taken in ID order whenever several are ready, so the same network always
/// runs in the same order.
fn run_order(nodes: &HashMap<Uuid, ModelNode>) -> anyhow::Result<(Vec<Uuid>, Vec<Uuid>)> {
    anyhow::ensure!(!nodes.is_empty(), "model has no nodes");

    // count each node's unfilled inputs and list the nodes each one feeds
    let mut waiting: HashMap<Uuid, usize> = HashMap::with_capacity(nodes.len());
    let mut feeds: HashMap<Uuid, Vec<Uuid>> = HashMap::with_capacity(nodes.len());
    for (id, node) in nodes {
        waiting.insert(*id, node.inputs.len());
        for input in &node.inputs {
            anyhow::ensure!(nodes.contains_key(input), "node {id} reads from missing node {input}");
            feeds
                .entry(*input)
                .or_default()
                .push(*id);
        }
    }

    // input nodes are ready straight away
    let mut ready = waiting
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    ready.sort();
    let mut ready = VecDeque::from(ready);

    // fill each ready node, readying the nodes it feeds once all their inputs are filled
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(id) = ready.pop_front() {
        order.push(id);

        let mut next = feeds
            .get(&id)
            .cloned()
            .unwrap_or_default();
        next.sort();
        next.dedup();
        for fed in next {
            let count = waiting
                .get_mut(&fed)
                .with_context(|| format!("node {fed} is missing"))?;
            let reads = nodes
                .get(&fed)
                .with_context(|| format!("node {fed} is missing"))?
                .inputs
                .iter()
                .filter(|input| **input == id)
                .count();
            *count -= reads;
            if *count == 0 { ready.push_back(fed); }
        }
    }

    // anything left was never readied, so it sits on or after a cycle
    if order.len() < nodes.len() {
        let mut unfilled = waiting
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        unfilled.sort();
        anyhow::bail!("nodes {unfilled:?} could not be filled, they depend on a cycle");
    }

    // output nodes feed no other node
    let mut outputs = nodes
        .keys()
        .filter(|id| !feeds.contains_key(id))
        .copied()
        .collect::<Vec<_>>();
    outputs.sort();

    Ok((order, outputs))
}
