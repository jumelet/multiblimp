# Rename UD_Norwegian-Bokmaal -> UD_Norwegian_Bokmål-NDT 
mv ud-treebanks-v2.18/UD_Norwegian-Bokmaal ud-treebanks-v2.18/UD_Norwegian_Bokmaal-NDT 

# Rename UD_Norwegian-Nynorsk -> UD_Norwegian_Nynorsk-NDT
mv ud-treebanks-v2.18/UD_Norwegian-Nynorsk ud-treebanks-v2.18/UD_Norwegian_Nynorsk-NDT 

# Remove colons from Gheg
sed -r "s/([a-z]*):([a-z]*)/\1\2/g" ud-treebanks-v2.18/UD_Gheg-GPS/aln_gps-ud-test.conllu > tmp.conllu; mv tmp.conllu ud-treebanks-v2.18/UD_Gheg-GPS/aln_gps-ud-test.conllu

# Split UD_Romanian-ArT -> UD_Aromanian-ArT
mv ud-treebanks-v2.18/UD_Romanian-ArT ud-treebanks-v2.18/UD_Aromanian-ArT

# Split the Old Norse part of UD_Icelandic-IcePaHC (texts dated up to and including 1350, the year that starts each sent_id) into UD_Old_Norse-IcePaHC
mkdir -p ud-treebanks-v2.18/UD_Old_Norse-IcePaHC
cp ud-treebanks-v2.18/UD_Icelandic-IcePaHC/README.md ud-treebanks-v2.18/UD_Icelandic-IcePaHC/LICENSE.txt ud-treebanks-v2.18/UD_Old_Norse-IcePaHC/
for split in train dev test; do awk 'BEGIN { RS = ""; ORS = "\n\n" } match($0, /# sent_id = [0-9]+/) && substr($0, RSTART + 12, RLENGTH - 12) + 0 <= 1350' ud-treebanks-v2.18/UD_Icelandic-IcePaHC/is_icepahc-ud-$split.conllu > ud-treebanks-v2.18/UD_Old_Norse-IcePaHC/non_icepahc-ud-$split.conllu; done
