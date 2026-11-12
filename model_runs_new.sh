#!/bin/bash
# Word order evaluation runs, one job per model; submit from scripts/word_order.
# Walltime: fitted on Apertus 8B (0:48) and Gemma 3 27B (1:36) on an H100, i.e.
# ~28 min overhead + ~2.5 min per billion parameters, with a 1.5x margin,
# rounded up to half hours (at least 1h).
# Models that do not fit on one H100 get --gpus=N; eval_model.py then splits the
# model over the GPUs automatically.
cd "$(dirname "$0")/scripts/word_order"

# HF dataset to score and where to save the scores; override from the command line, e.g.
#   DATASET=jumelet/multiblimp-v2 OUTPUT_DIR=$HOME/model_results bash model_runs_new.sh
# The default is the pairs that are new in UD v2.19. Scores are saved as {model}.parquet,
# so they go to a separate folder to keep them apart from the scores on the full dataset.
DATASET="${DATASET:-jumelet/multiblimp-new}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/model_results_new}"

# Gemma 4 (10 models): E2B, E4B, 12B, 26B-A4B (MoE) and 31B, each in base and instruction-tuned versions.
sbatch -t 1:00:00 job.sh --model google/gemma-4-E2B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-4-E2B-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-4-E4B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-4-E4B-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-4-12B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-4-12B-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model google/gemma-4-26B-A4B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model google/gemma-4-26B-A4B-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model google/gemma-4-31B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model google/gemma-4-31B-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Gemma 3 (10 models): 270M, 1B, 4B, 12B and 27B, each in base and instruction-tuned versions.
sbatch -t 1:00:00 job.sh --model google/gemma-3-270m --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-3-270m-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-3-1b-pt --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-3-1b-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-3-4b-pt --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model google/gemma-3-4b-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-3-12b-pt --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model google/gemma-3-12b-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:30:00 job.sh --model google/gemma-3-27b-it --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:30:00 job.sh --model google/gemma-3-27b-pt --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Llama 3 (4 models): 8B and 70B, each in base and instruction-tuned versions.
sbatch -t 1:30:00 job.sh --model meta-llama/Meta-Llama-3-8B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model meta-llama/Meta-Llama-3-8B-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# 141 GB in bf16, split over 2 GPUs (94 GB each)
sbatch -t 5:30:00 --gpus=2 job.sh --model meta-llama/Meta-Llama-3-70B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 5:30:00 --gpus=2 job.sh --model meta-llama/Meta-Llama-3-70B-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Llama 4 (4 models): Scout (109B MoE) and Maverick (400B MoE), both with 17B active parameters, each in base and instruction-tuned versions.
# 218 GB in bf16, split over 3 GPUs (94 GB each)
sbatch -t 8:00:00 --gpus=3 job.sh --model meta-llama/Llama-4-Scout-17B-16E --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 8:00:00 --gpus=3 job.sh --model meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# 800 GB in bf16, does not fit on a single node (4x H100)
# sbatch job.sh --model meta-llama/Llama-4-Maverick-17B-128E --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# sbatch job.sh --model meta-llama/Llama-4-Maverick-17B-128E-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Aya + Tiny Aya (7 models): Aya Expanse 8B and 32B, plus Tiny Aya 3.35B in its base version and four instruction-tuned variants (Global, Earth, Fire, Water).
sbatch -t 1:30:00 job.sh --model CohereLabs/aya-expanse-8b --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model CohereLabs/aya-expanse-32b --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model CohereLabs/tiny-aya-base --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model CohereLabs/tiny-aya-global --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model CohereLabs/tiny-aya-earth --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model CohereLabs/tiny-aya-fire --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model CohereLabs/tiny-aya-water --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# OLMo 3 (4 models): 7B and 32B, base + post-trained.
sbatch -t 1:30:00 job.sh --model allenai/Olmo-3-1025-7B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model allenai/Olmo-3-7B-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model allenai/Olmo-3-1125-32B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# there is no OLMo 3 (3.0) 32B Instruct, only 32B Think; this is the 3.1 Instruct
sbatch -t 3:00:00 job.sh --model allenai/Olmo-3.1-32B-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Qwen 3.6 (2 models): 27B and 35B-A3B (MoE), both post-trained.
sbatch -t 3:30:00 job.sh --model Qwen/Qwen3.6-27B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 3:00:00 job.sh --model Qwen/Qwen3.6-35B-A3B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# EuroLLM (6 models): 1.7B, 9B and 22B, each in base and instruction-tuned versions.
sbatch -t 1:00:00 job.sh --model utter-project/EuroLLM-1.7B --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model utter-project/EuroLLM-1.7B-Instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model utter-project/EuroLLM-9B-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model utter-project/EuroLLM-9B-Instruct-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:30:00 job.sh --model utter-project/EuroLLM-22B-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:30:00 job.sh --model utter-project/EuroLLM-22B-Instruct-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Salamandra (6 models): 2B, 7B, 40B, base + post-trained.
sbatch -t 1:00:00 job.sh --model BSC-LT/salamandra-2b --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model BSC-LT/salamandra-2b-instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model BSC-LT/salamandra-7b --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model BSC-LT/salamandra-7b-instruct --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# Salamandra 40B: not found

# Apertus (4 models): 8B and 70B, each in base and instruction-tuned versions.
sbatch -t 1:30:00 job.sh --model swiss-ai/Apertus-8B-2509 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model swiss-ai/Apertus-8B-Instruct-2509 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# 141 GB in bf16, split over 2 GPUs (94 GB each)
sbatch -t 5:30:00 --gpus=2 job.sh --model swiss-ai/Apertus-70B-2509 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 5:30:00 --gpus=2 job.sh --model swiss-ai/Apertus-70B-Instruct-2509 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"

# Mistral 3 (8 models): Ministral 3 in 3B, 8B and 14B, and Mistral Large 3 (675B MoE), each in base and instruction-tuned versions.
# the Instruct repos without -BF16 store their weights in FP8
sbatch -t 1:00:00 job.sh --model mistralai/Ministral-3-3B-Base-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:00:00 job.sh --model mistralai/Ministral-3-3B-Instruct-2512-BF16 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model mistralai/Ministral-3-8B-Base-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 1:30:00 job.sh --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:00:00 job.sh --model mistralai/Ministral-3-14B-Base-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
sbatch -t 2:00:00 job.sh --model mistralai/Ministral-3-14B-Instruct-2512-BF16 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# 675B, does not fit on a single node (4x H100)
# sbatch job.sh --model mistralai/Mistral-Large-3-675B-Base-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
# sbatch job.sh --model mistralai/Mistral-Large-3-675B-Instruct-2512 --dataset "$DATASET" --output_dir "$OUTPUT_DIR"
