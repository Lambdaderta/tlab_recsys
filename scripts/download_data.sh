#!/bin/bash
# Downloads the Yambda-50M flat files used here (~0.8 GB) from Hugging Face into data/yambda/.
set -e
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/datasets/yandex/yambda/resolve/main
mkdir -p data/yambda/flat/50m
for f in listens likes dislikes multi_event; do
  curl -L --fail -o data/yambda/flat/50m/$f.parquet "$BASE/flat/50m/$f.parquet"
done
curl -L --fail -o data/yambda/artist_item_mapping.parquet "$BASE/artist_item_mapping.parquet"
ls -la data/yambda data/yambda/flat/50m
