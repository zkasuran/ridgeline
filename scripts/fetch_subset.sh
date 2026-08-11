#!/usr/bin/env bash
# Fetch a small idempotent dev subset of Dataset059 (seg-derived-recto-surfaces),
# ~250 MiB across all three scrolls. Re-running skips files already present.
# Pairs: imagesTr/<stem>_0000.tif  and  labelsTr/<stem>.tif
set -eu
BASE=https://dl.ash2txt.org/datasets/seg-derived-recto-surfaces
OUT=${1:-subset}
mkdir -p "$OUT/imagesTr" "$OUT/labelsTr"
for stem in \
  s1_z10240_y2560_x2560 s1_z10240_y2560_x3200 s1_z10240_y2880_x2560 \
  s1_z10240_y2880_x3200 s1_z10240_y2880_x3520 s1_z10240_y3200_x2560 \
  s1_z10240_y3200_x3520 s1_z10240_y3200_x3840 \
  s4_z1024_y1024_x1024 s4_z1024_y1280_x1024 s4_z1024_y1792_x1280 \
  s4_z1024_y1792_x2304 s4_z1024_y1792_x2560 s4_z1024_y2048_x1280 \
  s4_z1024_y2048_x1536 s4_z10240_y1024_x2048 \
  s5_z5500_y3990_x5890 s5_z5500_y4180_x3420 s5_z5500_y4370_x3990 \
  s5_z5690_y1710_x3800 s5_z5690_y2090_x4180 s5_z6000_y2850_x2470 \
  s5_z6190_y2090_x4180 s5_z6190_y2660_x3230 ; do
  img="$OUT/imagesTr/${stem}_0000.tif"
  lbl="$OUT/labelsTr/${stem}.tif"
  [ -s "$img" ] || curl -sf -o "$img" "$BASE/imagesTr/${stem}_0000.tif" || echo "miss img $stem"
  [ -s "$lbl" ] || curl -sf -o "$lbl" "$BASE/labelsTr/${stem}.tif"      || echo "miss lbl $stem"
done
echo "subset in $OUT ($(find "$OUT" -type f | wc -l) files)"
