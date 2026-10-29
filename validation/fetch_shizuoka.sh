#!/bin/bash
set -euo pipefail
# Download the four Shizuoka tiles of the second validation dataset (VIRTUAL SHIZUOKA, Shizuoka Prefecture,
# CC BY 4.0 / ODbL; public bucket s3://virtual-shizuoka, no account needed), then run make_shizuoka.py.
#   ./fetch_shizuoka.sh [dir]        (default ~/forest_data/shizuoka/raw; about 3 GB of LAS)
DIR=${1:-$HOME/forest_data/shizuoka/raw}
B=https://virtual-shizuoka.s3.ap-northeast-1.amazonaws.com
mkdir -p "$DIR/grid"
for k in 2021/LP/%s/08/MD/42/08MD4295 2021/LP/%s/08/OC/09/08OC0957 2025/LP/%s/08/NC/07/08NC0725 2025/LP/%s/08/MC/79/08MC7931; do
  m=$(basename "$k")
  curl -sSf -o "$DIR/$m.zip" "$B/$(printf "$k" Original).zip"
  (cd "$DIR" && unzip -o -q "$m.zip" && rm "$m.zip")
  curl -sSf -o "$DIR/grid/$m.zip" "$B/$(printf "$k" Grid).zip"
  (cd "$DIR/grid" && unzip -o -q "$m.zip" && rm "$m.zip")
  echo "$m"
done
