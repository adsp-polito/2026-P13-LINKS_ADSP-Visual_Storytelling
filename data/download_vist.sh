#!/usr/bin/env bash
# Download VIST into data/raw/vist/.  TEMPLATE – owner: Parsa. Verify URLs before running.
#
# Target layout (expected by src/p13/data/vist.py):
#   data/raw/vist/sis/{train,val,test}.story-in-sequence.json
#   data/raw/vist/dii/{train,val,test}.description-in-isolation.json
#   data/raw/vist/dis/{train,val,test}.description-in-sequence.json
#   data/raw/vist/images/{train,val,test}/<photo_flickr_id>.<ext>
set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw/vist}"
mkdir -p "$RAW_DIR"/{sis,dii,dis,images}

echo "1. Download the SIS / DII / DIS annotation archives from https://visionandlanguage.net/VIST/dataset.html"
echo "   and extract the JSON files into $RAW_DIR/{sis,dii,dis}/"
echo "2. Download the image archives (train is split into several parts; ~250 GB total)"
echo "   and extract into $RAW_DIR/images/{train,val,test}/"
echo "   Tip: start with val + test only (much smaller) for zero-shot experiments."
echo
echo "TODO(Parsa): replace these echo lines with the actual wget/curl + tar commands."
