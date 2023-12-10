'''A module for creating prebatched samples for the transformer model. 

Prebatching creates Zarr files with randomly created sentence structure for the transformer to sample from. 
This is done to avoid the overhead of dynamically creating samples during training.

Example usage for one species (salmonella_enterica_2017):
    # python src/0_prebatching/make_ab_prebatches.py \
    # --genespace_input_file "data/genespace/narms_2017_and_2022_genespace.dir.zarr" \
    # --prebatch_output_file "data/pre_batching/ab_salmonella_enterica_2017.dir.zarr" \
    # --narms_metadata_2017_table_input_file "data/phenotypes_and_metadata/narms_metadata_2017.csv" \
    # --narms_metadata_2022_table_input_file "data/phenotypes_and_metadata/narms_metadata_2022.csv" \
    # --narms_phenotypes_2017_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2017.csv" \
    # --narms_phenotypes_2022_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2022.csv" \
    # --number_samples_to_prebatch 20000000 \
    # --transformer_input_length 10 \
    # --included_species "salmonella_enterica_2017" \
    # --chunk_size 100000 \
    # --world_size 2 \
    # --local_rank 0
'''

import datetime
import os
import pathlib
import time
from typing import Tuple

import click
import numcodecs
import numpy as np
import tqdm
import zarr

from fast_differential_space import randint, shuffle_matrix_per_row, generate_differential_abspace
import helper

def initialize_output_database(  # pylint: disable=too-many-arguments
    prebatch_output_file : str,
    chunk_size : int,
    num_chunks : int,
    transformer_input_length : int,
    transformer_output_classifier_length : int,
    included_species : str
) -> Tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]:
    '''Initializes the output Zarr database.
    
    Args:
        prebatch_output_file (str): The output file location.
        chunk_size (int): The number of samples to generate per chunk.
        num_chunks (int): The total number of chunks to generate.
        transformer_input_length (int): The total number of inputs that a transformer will use.
        transformer_output_classifier_length (int): The total number of outputs that a transformer will generate.
        included_species (str): The species to include, delimited by semi-colon.
    
    Returns:
        Tuple[zarr.core.Array, zarr.core.Array, zarr.core.Array]: The subset sizes, shuffled accessions, and differential genespace arrays.
    '''

    z_store_out         = zarr.DirectoryStore(prebatch_output_file)
    z_root_out          = zarr.group(store=z_store_out)
    compressor          = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=2)

    z_subset_sizes = z_root_out.zeros(
        'subset_sizes',
        shape=( num_chunks, chunk_size, 2 ),
        chunks=( 1, -1, -1 ),
        dtype=np.int32,
        compressor=compressor
    )

    z_shuffled_accessions = z_root_out.zeros(
        'shuffled_accessions',
        shape=( num_chunks, chunk_size, transformer_input_length ),
        chunks=( 1, -1, -1 ),
        dtype=np.int32,
        compressor=compressor
    )

    z_differential_genespace = z_root_out.zeros(
        'differential_genespace',
        shape=( num_chunks, chunk_size, transformer_output_classifier_length ),
        chunks=( 1, -1, -1 ),
        dtype=np.bool_,
        compressor=compressor
    )

    z_root_out.attrs['creation_date']				= datetime.datetime.now().strftime('%b_%d_%Y__%H:%M:%S')
    z_root_out.attrs['chunk_size']					= f'{chunk_size}'
    z_root_out.attrs['num_chunks']					= f'{num_chunks}'
    z_root_out.attrs['transformer_input_length']	= f'{transformer_input_length}'
    z_root_out.attrs['script_path']					= os.path.abspath( __file__ )
    z_root_out.attrs['output_path']					= os.path.abspath( prebatch_output_file )
    z_root_out.attrs['included_species']			= included_species

    return z_subset_sizes, z_shuffled_accessions, z_differential_genespace

@click.command()
@click.option('--genespace_input_file', 					'-i_genespace', 															help='Specifies the genespace warehouse to use.',				type=click.Path(exists=True, file_okay=False))
@click.option('--prebatch_output_file', 					'-o_prebatch', 																help='Specifies output file location to use.',					type=click.Path(exists=False, file_okay=True))
@click.option('--narms_metadata_2017_table_input_file', 	'-i_2017_metadata', 														help='Specifies the NARMS 2017 metadata table to use.',			type=click.Path(exists=True, file_okay=True))
@click.option('--narms_metadata_2022_table_input_file', 	'-i_2022_metadata', 														help='Specifies the NARMS 2022 metadata table to use.',			type=click.Path(exists=True, file_okay=True))
@click.option('--narms_phenotypes_2017_table_input_file', 	'-i_2017_phenotypes', 														help='Specifies the NARMS 2017 phenotypes table to use.',		type=click.Path(exists=True, file_okay=True))
@click.option('--narms_phenotypes_2022_table_input_file', 	'-i_2022_phenotypes', 														help='Specifies the NARMS 2022 phenotypes table to use.',		type=click.Path(exists=True, file_okay=True))
@click.option('--transformer_input_length',					'-input_len',																help='The total number of inputs that a transformer will use.',	type=click.INT)
@click.option('--included_species', 						'-s',					default='salmonella_enterica_2017;shigella_sonnei',	help='The species to include, delimited by semi-colon.',		type=click.STRING)
@click.option('--num_chunks',					            '-n',				                                                        help='The total number of input chunks to generate.',								type=click.INT)
@click.option('--chunk_size', 	     						'-c', 																		help='Specifies how many samples to generate at a time.', 		type=click.INT)
@click.option('--world_size', 	     						'-n', 																		help='Specifies the world size.', 								type=click.INT)
@click.option('--local_rank', 	     						'-r', 																		help='Specifies the local rank.', 								type=click.INT)
def main( # pylint: disable=too-many-arguments, too-many-locals
    genespace_input_file,
    prebatch_output_file,
    narms_metadata_2017_table_input_file,
    narms_metadata_2022_table_input_file,
    narms_phenotypes_2017_table_input_file,
    narms_phenotypes_2022_table_input_file,
    transformer_input_length,
    included_species,
    num_chunks,
    chunk_size,
    world_size,
    local_rank,
):
    assert num_chunks > world_size, f'Total chunk count ({num_chunks}) are smaller than the total number of nodes ({world_size})'

    numcodecs.blosc.set_nthreads(32) # pylint: disable=c-extension-no-member

    # Load, align, filter/clean, and process all NARMS data (genespace + metadata + phenotypes)
    included_species_list 		= included_species.split(';')
    included_phenotypes_list 	= [
        'mic_amikacin', 				'mic_amoxicillin-clavulanic_acid', 		'mic_ampicillin', 				'mic_apramycin', 
        'mic_azithromycin', 			'mic_aztreonam', 						'mic_benzalkonium-chloride', 	'mic_cefepime', 
        'mic_cefotaxime', 				'mic_cefoxitin', 						'mic_cefquinome', 				'mic_ceftazidime', 
        'mic_ceftiofur', 				'mic_ceftriaxone', 						'mic_cephalothin', 				'mic_chloramphenicol', 
        'mic_ciprofloxacin', 			'mic_clindamycin', 						'mic_daptomycin', 				'mic_doxycycline', 
        'mic_erythromycin', 			'mic_florfenicol', 						'mic_gentamicin', 				'mic_imipenem', 
        'mic_kanamycin', 				'mic_lincomycin', 						'mic_linezolid', 				'mic_meropenem', 
        'mic_nalidixic_acid', 			'mic_nitrofurantoin', 					'mic_olistin', 					'mic_penicillin', 
        'mic_piperacillin-tazobactam', 	'mic_quinupristin-dalfopristin', 		'mic_streptomycin', 			'mic_sulfamethoxazole', 
        'mic_sulfisoxazole', 			'mic_telithromcyin', 					'mic_telithromycin', 			'mic_tetracycline', 
        'mic_tigecycline', 				'mic_trimethoprim-sulfamethoxazole', 	'mic_tylosin', 					'mic_vancomycin'
    ]

    genespace_matrix, _, _, _, _, _, num_accessions, num_genes = helper.load_and_process_narms_data(
        included_species_list,
        included_phenotypes_list,
        genespace_input_file,
        narms_metadata_2017_table_input_file,
        narms_metadata_2022_table_input_file,
        narms_phenotypes_2017_table_input_file,
        narms_phenotypes_2022_table_input_file
    )

    # Specify some parameters
    segment_length 							= transformer_input_length // 2
    transformer_output_classifier_length 	= num_genes * 2

    # Initialize output
    if local_rank == 0:
        z_subset_sizes, z_shuffled_accessions, z_differential_genespace = initialize_output_database(
            prebatch_output_file,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            transformer_input_length=transformer_input_length, 
            transformer_output_classifier_length=transformer_output_classifier_length, 
            included_species=included_species
        )

    # Wait until the Zarr directory is built by the root node
    else:
        while pathlib.Path(prebatch_output_file).is_dir() is False:
            time.sleep(10)

        time.sleep(60)

        z_store_out					= zarr.DirectoryStore(prebatch_output_file)
        z_root_out 					= zarr.group(store=z_store_out)
        z_subset_sizes				= z_root_out['subset_sizes']
        z_shuffled_accessions		= z_root_out['shuffled_accessions']
        z_differential_genespace	= z_root_out['differential_genespace']

    # Allocate enough prebuffer memory to cover one chunk
    buffer_ordered_accessions		= np.arange(num_accessions).repeat(chunk_size).reshape(num_accessions, -1).T.astype(np.int32)
    buffer_shuffled_accessions		= np.empty_like(buffer_ordered_accessions)[:, 0 : transformer_input_length]
    buffer_differential_genespace	= np.full((chunk_size, transformer_output_classifier_length), False)

    # Distribute the chunk prebatches
    chunk_idxs 						= np.arange(num_chunks)
    starts, ends 					= helper.chunk_task_indices( n=num_chunks, k=world_size )
    s, e 							= starts[local_rank], ends[local_rank]

    for chunk_idx in tqdm.tqdm( chunk_idxs[s : e], desc='Generating chunks of prebatched samples', unit=f'Prebatch chunk (n={chunk_size})' ):

        # Generate a 2-column matrix full of random values representing left/right subset sizes
        buffer_subset_sizes = randint(
            low=1,
            high=segment_length + 1,
            shape_n=chunk_size,
            shape_m=2,
        )

        # Generate the randomly shuffled accessions
        shuffle_matrix_per_row(
            input_matrix=buffer_ordered_accessions,
            output_shuffled_matrix=buffer_shuffled_accessions,
        )

        # Finally, generate based on the left/right subset sizes and the shuffled accessions
        generate_differential_abspace(
            input_space_matrix=genespace_matrix,
            input_accession_matrix=buffer_shuffled_accessions,
            input_subset_sizes=buffer_subset_sizes,
            output_differential_space=buffer_differential_genespace
        )

        z_subset_sizes[chunk_idx] 			= buffer_subset_sizes
        z_shuffled_accessions[chunk_idx] 	= buffer_shuffled_accessions
        z_differential_genespace[chunk_idx] = buffer_differential_genespace

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
