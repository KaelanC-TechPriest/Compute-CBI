#!/usr/bin/env bash
# =============================================================================
# Please note, this script was written, in its entirety, with an LLM.
# =============================================================================
# Build full-extent mosaics of CBI rasters with a single gdalwarp call per
# kind: gdalwarp accepts multiple source files directly, computes their
# union extent/resolution/SRS itself, and merges them (later files win on
# overlap). Raw (*_CBI.tif) and bias-corrected (*_CBI_bc.tif / *_bc.tif)
# inputs are mosaicked separately.
set -euo pipefail

PROG=$(basename "$0")

usage() {
  cat <<EOF
Usage: $PROG [options] <tif...>
       $PROG [options] -i <dir>

Options:
  -i DIR    read *.tif/*.tiff recursively from DIR (instead of positional args)
  -o DIR    output directory (default: .)
  -p NAME   output filename prefix (default: mosaic)
  -k KIND   which kind to build: raw | bc | both (default: both)
  -r RES    output pixel size, in target-SRS units (default: from inputs)
  -s SRS    output SRS, e.g. EPSG:5070 (default: from the first input)
  -n VALUE  nodata value, numeric or "nan" (default: nan)
  -h        show this help

Outputs (whichever kinds have inputs):
  DIR/NAME_raw.tif
  DIR/NAME_bc.tif
EOF
}

OUT_DIR="."
PREFIX="mosaic"
KIND="both"
RES=""
SRS=""
NODATA="nan"
IN_DIR=""

while getopts ":i:o:p:k:r:s:n:h" opt; do
  case "$opt" in
    i) IN_DIR=$OPTARG ;;
    o) OUT_DIR=$OPTARG ;;
    p) PREFIX=$OPTARG ;;
    k) KIND=$OPTARG ;;
    r) RES=$OPTARG ;;
    s) SRS=$OPTARG ;;
    n) NODATA=$OPTARG ;;
    h) usage; exit 0 ;;
    \?) echo "error: unknown option -$OPTARG" >&2; usage >&2; exit 2 ;;
    :) echo "error: -$OPTARG requires an argument" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

case "$KIND" in
  raw|bc|both) ;;
  *) echo "error: -k must be raw, bc, or both" >&2; exit 2 ;;
esac

command -v gdalwarp >/dev/null || { echo "error: gdalwarp not found in PATH" >&2; exit 1; }

FILES=()
if [[ -n "$IN_DIR" ]]; then
  [[ -d "$IN_DIR" ]] || { echo "error: not a directory: $IN_DIR" >&2; exit 1; }
  while IFS= read -r -d '' f; do
    FILES+=("$f")
  done < <(find "$IN_DIR" -type f \( -iname '*.tif' -o -iname '*.tiff' \) -print0 | sort -z)
fi
FILES+=("$@")

[[ ${#FILES[@]} -gt 0 ]] || { echo "error: no input GeoTIFFs" >&2; usage >&2; exit 1; }

# A file is "bc" if its name (minus extension, case-folded) ends in _bc;
# everything else is "raw". Covers *_CBI_bc.tif, *_cbi_bc.tiff, *_bc.tif, ...
RAW_FILES=()
BC_FILES=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "error: not a file: $f" >&2; exit 1; }
  stem=$(basename "$f")
  stem=${stem%.*}
  if [[ "${stem,,}" == *_bc ]]; then
    BC_FILES+=("$f")
  else
    RAW_FILES+=("$f")
  fi
done

mkdir -p "$OUT_DIR"

build_mosaic() {
  local out=$1 tag=$2
  shift 2
  local inputs=("$@")

  if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "skip: no $tag inputs"
    return 0
  fi

  # gdalwarp, given several sources and no pre-existing destination, creates
  # the output itself: extent is the union of all sources, resolution/SRS
  # come from the first source unless overridden, uncovered area is filled
  # with -dstnodata (not zero -- CBI can legitimately be 0), and later
  # sources overwrite earlier ones wherever they have valid data.
  local extra_args=()
  [[ -n "$SRS" ]] && extra_args+=(-t_srs "$SRS")
  [[ -n "$RES" ]] && extra_args+=(-tr "$RES" "$RES")

  echo "==> $(basename "$out")  (${#inputs[@]} file(s))"
  gdalwarp -q -overwrite -r near -ot Float32 \
    -srcnodata "$NODATA" -dstnodata "$NODATA" \
    "${extra_args[@]}" \
    -co TILED=YES -co COMPRESS=ZSTD -co ZSTD_LEVEL=1 -co BIGTIFF=IF_SAFER \
    "${inputs[@]}" "$out"
}

[[ "$KIND" == "both" || "$KIND" == "raw" ]] && build_mosaic "$OUT_DIR/${PREFIX}_raw.tif" raw "${RAW_FILES[@]}"
[[ "$KIND" == "both" || "$KIND" == "bc" ]] && build_mosaic "$OUT_DIR/${PREFIX}_bc.tif" bc "${BC_FILES[@]}"

echo "done"
