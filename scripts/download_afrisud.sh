# Download the AfriSUD treebanks (not part of the UD release) into a UD release folder.
# Usage: bash download_afrisud.sh [ud-treebanks-folder]   (default: resources/ud/ud-treebanks-v2.19, relative to the current directory)
UD_DIR="${1:-resources/ud/ud-treebanks-v2.19}"
if [ ! -d "$UD_DIR" ]; then
  echo "Not a directory: $UD_DIR" >&2
  exit 1
fi
cd "$UD_DIR" || exit 1
urls=(
  "https://gmd-universal.grew.fr/tgz/UD_Hausa-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_isiXhosa-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Kinyarwanda-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Naija-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Runyankore-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Swahili-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Wolof-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Yoruba-AfriSUD@conv.tgz"
)
for url in "${urls[@]}"; do
  curl -O "$url"
done
shopt -s nullglob
for f in *.tgz *.tar; do
  dir="${f%.tgz}"
  dir="${dir%.tar}"
  dir="${dir%@*}"
  mkdir -p "$dir"
  tmp="$(mktemp -d)"
  tar -xvzf "$f" -C "$tmp" 2>/dev/null || tar -xvf "$f" -C "$tmp"
  if [ "$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ] && [ "$(find "$tmp" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 0 ]; then
    inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d)"
    mv "$inner"/* "$dir"/
  else
    mv "$tmp"/* "$dir"/
  fi
  rm -rf "$tmp"
  rm "$f"
done
