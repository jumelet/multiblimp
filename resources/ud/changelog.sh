#!/usr/bin/env bash
# Apply our local changes to a UD release folder.
# Usage: bash changelog.sh <ud-treebanks-folder>   (e.g. bash changelog.sh ud-treebanks-v2.19)
set -euo pipefail

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <ud-treebanks-folder>" >&2
    exit 1
fi
UD_DIR="${1%/}"

# Rename a treebank folder, unless that was already done (a second `mv` would move it into the target folder)
rename_treebank() {
    if [ -d "$UD_DIR/$1" ] && [ ! -e "$UD_DIR/$2" ]; then
        mv "$UD_DIR/$1" "$UD_DIR/$2"
    else
        echo "Skipping rename $1 -> $2 (source missing or target exists)" >&2
    fi
}

# Rename UD_Norwegian-Bokmaal -> UD_Norwegian_Bokmål-NDT
rename_treebank UD_Norwegian-Bokmaal UD_Norwegian_Bokmål-NDT

# Rename UD_Norwegian-Nynorsk -> UD_Norwegian_Nynorsk-NDT
rename_treebank UD_Norwegian-Nynorsk UD_Norwegian_Nynorsk-NDT

# Remove colons from Gheg
GHEG="$UD_DIR/UD_Gheg-GPS/aln_gps-ud-test.conllu"
sed -r "s/([a-z]*):([a-z]*)/\1\2/g" "$GHEG" > "$GHEG.tmp"; mv "$GHEG.tmp" "$GHEG"

# Split UD_Romanian-ArT -> UD_Aromanian-ArT
rename_treebank UD_Romanian-ArT UD_Aromanian-ArT

# Split UD_Old_Georgian-MidGLauRo -> UD_Middle_Georgian-MidGLauRo (12th and 17th c. texts; since UD v2.19)
rename_treebank UD_Old_Georgian-MidGLauRo UD_Middle_Georgian-MidGLauRo

# Split the Old Norse part of UD_Icelandic-IcePaHC (texts dated up to and including 1350, the year that starts each sent_id) into UD_Old_Norse-IcePaHC
mkdir -p "$UD_DIR/UD_Old_Norse-IcePaHC"
cp "$UD_DIR/UD_Icelandic-IcePaHC/README.md" "$UD_DIR/UD_Icelandic-IcePaHC/LICENSE.txt" "$UD_DIR/UD_Old_Norse-IcePaHC/"
for split in train dev test; do awk 'BEGIN { RS = ""; ORS = "\n\n" } match($0, /# sent_id = [0-9]+/) && substr($0, RSTART + 12, RLENGTH - 12) + 0 <= 1350' "$UD_DIR/UD_Icelandic-IcePaHC/is_icepahc-ud-$split.conllu" > "$UD_DIR/UD_Old_Norse-IcePaHC/non_icepahc-ud-$split.conllu"; done
