'''A module that holds a collection of useful functions for prebatching and preprocessing data.'''

from typing import Tuple

import numpy as np
import pandas as pd
import zarr

def collate_metadata_and_phenotypes(
    z_root_genespace : zarr.Group, 
    df_metadata_2017 : pd.DataFrame, 
    df_metadata_2022 : pd.DataFrame, 
    df_phenotypes_2017 : pd.DataFrame, 
    df_phenotypes_2022 : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''A function that aligns the genespace, metadata, and phenotypes into three dataframes.
    
    Args:
        z_root_genespace (zarr.Group): The root group of the genespace zarr store.
        df_metadata_2017 (pd.DataFrame): The metadata table for the 2017 NARMS data.
        df_metadata_2022 (pd.DataFrame): The metadata table for the 2022 NARMS data.
        df_phenotypes_2017 (pd.DataFrame): The phenotypes table for the 2017 NARMS data.
        df_phenotypes_2022 (pd.DataFrame): The phenotypes table for the 2022 NARMS data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the genespace, metadata, and phenotypes dataframes.
    '''

    # 1.) Generate table for genespace
    df_genespace = pd.DataFrame(
        data=z_root_genespace['genotypes']['gene_presence_absence_matrix'][:], 
        index=z_root_genespace['genotypes']['ids'][:],
        columns=z_root_genespace['genotypes']['gene_names'][:],
    )

    # 2.) Prepare metadata, need both 2017 and 2022 together to perform accurate filtering
    # 3.) Annotate S. enterica as 2017 or 2022 depending on the source dataframe
    df_metadata_2017, df_metadata_2022													= df_metadata_2017.set_index('accession_id'), df_metadata_2022.set_index('accession_id')
    df_metadata_2017.loc[ df_metadata_2017.species == 'salmonella_enterica', 'species']	= 'salmonella_enterica_2017'						
    df_metadata_2022.loc[ df_metadata_2022.species == 'salmonella_enterica', 'species']	= 'salmonella_enterica_2022'						
    df_metadata_2017, df_metadata_2022													= df_metadata_2017[['species', 'serotype']], df_metadata_2022[['species', 'serotype']]
    df_metadata																			= pd.concat([df_metadata_2017, df_metadata_2022])

    # 4.) Prepare phenotypes
    df_phenotypes_2017, df_phenotypes_2022												= df_phenotypes_2017.set_index('accession_id'), df_phenotypes_2022.set_index('accession_id')
    df_phenotypes																		= pd.concat([df_phenotypes_2017, df_phenotypes_2022])

    assert len( set(df_metadata.index.tolist()) ) == len(df_metadata.index.tolist()), 'Combined 2017+2022 metadata dataframe has duplicate accessions'
    assert len( set(df_phenotypes.index.tolist()) ) == len(df_phenotypes.index.tolist()), '2017 phenotypes dataframe has duplicate accessions'

    # 5.) Align the metadata and phenotypes dataframes to the genespace
    _, df_metadata 		= df_genespace.align(df_metadata, join='left', axis=0)
    _, df_phenotypes 	= df_genespace.align(df_phenotypes, join='left', axis=0)

    # Make sure that we have cooresponding species information for all individuals in the genespace
    assert df_metadata['species'].isnull().sum() == 0, 'Collated metadata has missing species identifers for individuals in the genespace'

    return df_genespace, df_metadata, df_phenotypes

def filter_by_species(
    df_genespace : pd.DataFrame,
    df_metadata : pd.DataFrame,
    df_phenotypes : pd.DataFrame,
    species_to_keep : list =[
        'shigella_sonnei', 		'vibrio_cincinnatiensis', 	'vibrio_furnissii', 		'shigella_flexneri', 	'vibrio_other', 		'vibrio_parahaemolyticus', 	'campylobacter_coli', 
        'campylobacter_jejuni', 'vibrio_alginolyticus', 	'escherichia_coli', 		'vibrio_mimicus', 		'vibrio_navarrensis', 	'vibrio_fluvialis', 		'shigella_boydii', 
        'shigella_unknown', 	'salmonella_enterica_2017', 'salmonella_enterica_2022',	'vibrio_vulnificus', 	'shigella_dysenteriae'
    ]
):
    '''A function that filters the genespace, metadata, and phenotypes dataframes by species.
    
    Args:
        df_genespace (pd.DataFrame): The genespace dataframe.
        df_metadata (pd.DataFrame): The metadata dataframe.
        df_phenotypes (pd.DataFrame): The phenotypes dataframe.
        species_to_keep (list): A list of species to keep. Defaults to all species.
    '''

    # Make sure all the indices are correctly lined up
    assert len(df_genespace) > 0, 'No accessions found in genespace table'
    assert df_genespace.index.equals( df_metadata.index ), 'Accession index mismatch between genespace and metadata tables'
    assert df_genespace.index.equals( df_phenotypes.index ), 'Accession index mismatch between genespace and phenotype tables'

    # Build and apply the species mask
    species_mask 	= df_metadata['species'].isin(species_to_keep)

    df_genespace	= df_genespace.loc[species_mask]
    df_metadata		= df_metadata.loc[species_mask]
    df_phenotypes	= df_phenotypes.loc[species_mask]

    assert len(df_genespace) > 0, 'After removing species, there are no accessions left genespace table'

    return df_genespace, df_metadata, df_phenotypes

def load_and_process_narms_data(
    included_species_list : list,
    included_phenotypes_list : list,
    genespace_input_file : str,
    narms_metadata_2017_table_input_file : str,
    narms_metadata_2022_table_input_file : str,
    narms_phenotypes_2017_table_input_file : str,
    narms_phenotypes_2022_table_input_file : str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, int, int]:
    '''A function that loads, aligns, filters, and processes all NARMS data (genespace + metadata + phenotypes).
    
    Args:
        included_species_list (list): A list of species to include.
        included_phenotypes_list (list): A list of phenotypes to include.
        genespace_input_file (str): The path to the genespace zarr store.
        narms_metadata_2017_table_input_file (str): The path to the 2017 metadata table.
        narms_metadata_2022_table_input_file (str): The path to the 2022 metadata table.
        narms_phenotypes_2017_table_input_file (str): The path to the 2017 phenotypes table.
        narms_phenotypes_2022_table_input_file (str): The path to the 2022 phenotypes table.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, int, int]: A tuple containing the genespace matrix, phenotype matrix, accession ids, species vector, 
        serotype vector, phenotype names, number of accessions, and number of genes.'''

    # Load genespace data from the Zarr
    z_store_genespace	= zarr.DirectoryStore(genespace_input_file)
    z_root_genespace	= zarr.open_group(z_store_genespace, mode='r')

    # Load all the tabular phenotype & metadata
    df_metadata_2017	= pd.read_csv(narms_metadata_2017_table_input_file)
    df_metadata_2022	= pd.read_csv(narms_metadata_2022_table_input_file)
    df_phenotypes_2017	= pd.read_csv(narms_phenotypes_2017_table_input_file)
    df_phenotypes_2022	= pd.read_csv(narms_phenotypes_2022_table_input_file)

    # Combine/align the genespace and all the tables using several left joins with the genespace as the anchor
    df_genespace, df_metadata, df_phenotypes = collate_metadata_and_phenotypes(z_root_genespace, df_metadata_2017, df_metadata_2022, df_phenotypes_2017, df_phenotypes_2022)

    # Remove the species that will not be included
    df_genespace, df_metadata, df_phenotypes = filter_by_species(df_genespace, df_metadata, df_phenotypes, species_to_keep=included_species_list)

    # After filtering by species, trim the genespace of genes that contain no variation
    num_unique			= df_genespace.nunique()
    column_idxs_to_drop	= num_unique[num_unique == 1].index
    df_genespace		= df_genespace.drop(column_idxs_to_drop, axis=1)

    df_phenotypes		= df_phenotypes[ included_phenotypes_list ]

    # Export the below data and parameters
    genespace_matrix	= df_genespace.to_numpy()
    phenotype_matrix	= df_phenotypes.to_numpy()

    accession_ids 		= df_genespace.index.to_numpy()
    phenotype_names		= list( df_phenotypes.columns )
    species_vector 		= df_metadata['species'].to_numpy()
    serotype_vector		= df_metadata['serotype'].to_numpy()

    num_accessions 		= genespace_matrix.shape[0]
    num_genes 			= genespace_matrix.shape[1]
    
    return genespace_matrix, phenotype_matrix, accession_ids, species_vector, serotype_vector, phenotype_names, num_accessions, num_genes

def chunk_task_indices( n, k ) -> Tuple[np.ndarray, np.ndarray]:
    '''A function that returns the start and end indices for a chunk of tasks.
    
    Args:
        n (int): The total number of tasks.
        k (int): World size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the start and end indices for a chunk of tasks.
    '''

    segments = [(n // k) + (1 if i < (n % k) else 0) for i in range(k)]
    ends	= np.cumsum(segments)
    starts	= ends - segments

    return starts, ends