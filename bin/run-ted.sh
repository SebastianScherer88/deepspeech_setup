#!/bin/sh
set -xe
if [ ! -f DeepSpeech_no_kenlm.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

python -u DeepSpeech_no_kenlm.py \
  --train_files data/ted/ted-train.csv \
  --dev_files data/ted/ted-dev_clean.csv \
  --test_files data/ldc93s1/ldc93s1.csv \
  --test False \
  --train_batch_size 10 \
  --dev_batch_size 10 \
  --test_batch_size 10 \
  --n_hidden 2048 \
  --epoch 20 \
  --learning_rate 0.0001 \
  --checkpoint_dir "./checkpoints" \
  --checkpoint_secs 1800 \
  --display_step 1 \
  --validation_step 1 \
  --report_count 0 \
  --early_stop False \
  --log_placement True \
  --export_dir "./models" \
  "$@"
