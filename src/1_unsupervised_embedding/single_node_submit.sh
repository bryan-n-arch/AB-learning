#!/bin/bash
#SBATCH -p GPU-small
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:v100-32:4
#SBATCH --ntasks-per-node=20
#SBATCH --job-name=ab_all

cd /ocean/projects/mcb180070p/bnaiden/source/model_genespace_embeddings

source /jet/home/bnaiden/.bashrc

conda activate pytorch_1_8

# AB learning
python -m torch.distributed.launch --master_port 123453 --nproc_per_node=4 --nnodes=1 src/1_unsupervised_embedding/train_genespace.py \
--genespace_input_file "data/genespace/narms_2017_and_2022_genespace.dir.zarr" \
--prebatch_input_file "data/pre_batching/ab_all_species.dir.zarr" \
--narms_metadata_2017_table_input_file "data/phenotypes_and_metadata/narms_metadata_2017.csv" \
--narms_metadata_2022_table_input_file "data/phenotypes_and_metadata/narms_metadata_2022.csv" \
--narms_phenotypes_2017_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2017.csv" \
--narms_phenotypes_2022_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2022.csv" \
--included_phenotypes "mic_amoxicillin-clavulanic_acid;mic_azithromycin;mic_cefoxitin;mic_ceftiofur;mic_ceftriaxone;mic_ciprofloxacin;mic_gentamicin;mic_kanamycin;mic_nalidixic_acid;mic_sulfisoxazole;mic_tetracycline;mic_trimethoprim-sulfamethoxazole" \
--included_species "campylobacter_coli;campylobacter_jejuni;escherichia_coli;salmonella_enterica_2017;salmonella_enterica_2022;shigella_boydii;shigella_dysenteriae;shigella_flexneri;shigella_sonnei;shigella_unknown;vibrio_alginolyticus;vibrio_cincinnatiensis;vibrio_fluvialis;vibrio_furnissii;vibrio_mimicus;vibrio_navarrensis;vibrio_other;vibrio_parahaemolyticus;vibrio_vulnificus" \
--embedding_dimension 128 \
--input_length 10 \
--num_layers 4 \
--num_heads 4 \
--proj_dropout_prob 0.3 \
--attn_dropout_prob 0.0 \
--num_warmup_steps 5000 \
--weight_decay 0.10 \
--log_file_name "viz/all_species_ab.html" \
--embeddings_file_name "data/embeddings/all_species_ab.dir.zarr" \
--model_file_name "models/all_species_ab.pt"
