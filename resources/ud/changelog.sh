# Rename UD_Norwegian-Bokmaal -> UD_Norwegian_Bokmål-NDT 
mv ud-treebanks-v2.16/UD_Norwegian-Bokmaal ud-treebanks-v2.16/UD_Norwegian_Bokmål-NDT 

# Rename UD_Norwegian-Nynorsk -> UD_Norwegian_Nynorsk-NDT
mv ud-treebanks-v2.16/UD_Norwegian-Nynorsk ud-treebanks-v2.16/UD_Norwegian_Nynorsk-NDT 

# Remove colons from Gheg
sed -r "s/([a-z]*):([a-z]*)/\1\2/g" ud-treebanks-v2.16/UD_Gheg-GPS/aln_gps-ud-test.conllu > tmp.conllu; mv tmp.conllu ud-treebanks-v2.16/UD_Gheg-GPS/aln_gps-ud-test.conllu
