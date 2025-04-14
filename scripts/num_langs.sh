find ../final_pairs/*/*/*.tsv | sed -E 's|.*/([^/]+)/.*|\1|' | sort | uniq | wc -l
find ../final_pairs/*/*/*.tsv | wc -l
find ../final_pairs/*/*/*.tsv | xargs wc -l | tail -1

find ../final_pairs/*/*/*[~+].tsv | sed -E 's|.*/([^/]+)/.*|\1|' | sort | uniq | wc -l
find ../final_pairs/*/*/*[~+].tsv | wc -l
find ../final_pairs/*/*/*[~+].tsv | xargs wc -l | tail -1
