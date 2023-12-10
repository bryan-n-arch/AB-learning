#!/bin/bash
#SBATCH -p bigmem
#SBATCH -t 96:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --mem=720GB
#SBATCH --array=0-1

# World parameters
WORLD_SIZE=2
RANK=$SLURM_ARRAY_TASK_ID

# Input/Output parameters
INCLUDED_SPECIES="salmonella_enterica_2017"

NUMBER_SAMPLES_TO_PREBATCH=20000000
CHUNK_SIZE=100000

TRANSFORMER_INPUT_LENGTH=10

# File parameters
GENESPACE_INPUT_FILE="data/genespace/narms_2017_and_2022_genespace.dir.zarr"

PREBATCH_SPECIES_NAMES=`echo "${INCLUDED_SPECIES}" | tr ";" .`
PREBATCH_OUTPUT_FILE="data/pre_batching/ab_${PREBATCH_SPECIES_NAMES}.dir.zarr"

NARMS_METADATA_2017_TABLE_INPUT_FILE="data/phenotypes_and_metadata/narms_metadata_2017.csv"
NARMS_METADATA_2022_TABLE_INPUT_FILE="data/phenotypes_and_metadata/narms_metadata_2022.csv"
NARMS_PHENOTYPES_2017_TABLE_INPUT_FILE="data/phenotypes_and_metadata/narms_phenotypes_2017.csv"
NARMS_PHENOTYPES_2022_TABLE_INPUT_FILE="data/phenotypes_and_metadata/narms_phenotypes_2022.csv"

/usr/bin/time -v python src/0_prebatching/make_ab_prebatches.py \
--genespace_input_file $GENESPACE_INPUT_FILE \
--prebatch_output_file $PREBATCH_OUTPUT_FILE \
--narms_metadata_2017_table_input_file $NARMS_METADATA_2017_TABLE_INPUT_FILE \
--narms_metadata_2022_table_input_file $NARMS_METADATA_2022_TABLE_INPUT_FILE \
--narms_phenotypes_2017_table_input_file $NARMS_PHENOTYPES_2017_TABLE_INPUT_FILE \
--narms_phenotypes_2022_table_input_file $NARMS_PHENOTYPES_2022_TABLE_INPUT_FILE \
--number_samples_to_prebatch $NUMBER_SAMPLES_TO_PREBATCH \
--transformer_input_length $TRANSFORMER_INPUT_LENGTH \
--included_species "salmonella_enterica_2017" \
--chunk_size $CHUNK_SIZE \
--world_size $WORLD_SIZE \
--local_rank $RANK
