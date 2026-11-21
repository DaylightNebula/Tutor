#!/usr/bin/env bash
# Trains a model for every data set in the `DataSet` enum and saves each one to
# `models/<data-set>-1.tutor`, overwriting any existing file of the same name.
#
# Usage: ./train_all.sh   (from any directory)
set -euo pipefail

# run from the iris crate so the csv data files resolve
cd "$(dirname "$0")"
mkdir -p models

# build once up front so each training run starts immediately
cargo build -q --release

# read the data set names from the cli help so new enum variants are picked up automatically
data_sets=$(
    cargo run -q --release -- --help \
        | grep -- '--data-set' \
        | sed -E 's/.*possible values: ([^]]*)\].*/\1/' \
        | tr -d ','
)
if [ -z "$data_sets" ]; then
    echo "could not read data set names from --help" >&2
    exit 1
fi

# train and save each data set, continuing past failures and reporting them at the end
failed=()
for data_set in $data_sets; do
    echo "=== training $data_set ==="
    if ! cargo run -q --release -- --mode train --data-set "$data_set" --model "models/$data_set-1.tutor"; then
        failed+=("$data_set")
    fi
done

if [ ${#failed[@]} -ne 0 ]; then
    echo "failed data sets: ${failed[*]}" >&2
    exit 1
fi
echo "saved models for: $data_sets"
