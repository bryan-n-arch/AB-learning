import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import zarr
from numcodecs import blosc, Blosc, VLenUTF8

def CollateMetadataAndPhenotypes(z_root_genespace, df_metadata_2017, df_metadata_2022, df_phenotypes_2017, df_phenotypes_2022):

	# 1.) Generate table for genespace
	df_genespace = pd.DataFrame(
		data=z_root_genespace['genotypes']['gene_presence_absence_matrix'][:], 
		index=z_root_genespace['genotypes']['ids'][:],
		columns=z_root_genespace['genotypes']['gene_names'][:],
	);

	# 2.) Prepare metadata, need both 2017 and 2022 together to perform accurate filtering
	# 3.) Annotate S. enterica as 2017 or 2022 depending on the source dataframe
	df_metadata_2017, df_metadata_2022													= df_metadata_2017.set_index('accession_id'), df_metadata_2022.set_index('accession_id')
	df_metadata_2017.loc[ df_metadata_2017.species == 'salmonella_enterica', 'species']	= 'salmonella_enterica_2017';						
	df_metadata_2022.loc[ df_metadata_2022.species == 'salmonella_enterica', 'species']	= 'salmonella_enterica_2022';						
	df_metadata_2017, df_metadata_2022													= df_metadata_2017[['species', 'serotype']], df_metadata_2022[['species', 'serotype']];
	df_metadata																			= pd.concat([df_metadata_2017, df_metadata_2022]);

	# 4.) Prepare phenotypes
	df_phenotypes_2017, df_phenotypes_2022												= df_phenotypes_2017.set_index('accession_id'), df_phenotypes_2022.set_index('accession_id')
	df_phenotypes																		= pd.concat([df_phenotypes_2017, df_phenotypes_2022]);

	assert len( set(df_metadata.index.tolist()) ) == len(df_metadata.index.tolist()), 'Combined 2017+2022 metadata dataframe has duplicate accessions';
	assert len( set(df_phenotypes.index.tolist()) ) == len(df_phenotypes.index.tolist()), '2017 phenotypes dataframe has duplicate accessions';

	# 5.) Align the metadata and phenotypes dataframes to the genespace
	_, df_metadata 		= df_genespace.align(df_metadata, join='left', axis=0);
	_, df_phenotypes 	= df_genespace.align(df_phenotypes, join='left', axis=0);

	# Make sure that we have cooresponding species information for all individuals in the genespace
	assert df_metadata['species'].isnull().sum() == 0, 'Collated metadata has missing species identifers for individuals in the genespace'

	return df_genespace, df_metadata, df_phenotypes;

def FilterBySpecies(
	df_genespace, 
	df_metadata, 
	df_phenotypes, 
	species_to_keep=[
		'shigella_sonnei', 		'vibrio_cincinnatiensis', 	'vibrio_furnissii', 		'shigella_flexneri', 	'vibrio_other', 		'vibrio_parahaemolyticus', 	'campylobacter_coli', 
		'campylobacter_jejuni', 'vibrio_alginolyticus', 	'escherichia_coli', 		'vibrio_mimicus', 		'vibrio_navarrensis', 	'vibrio_fluvialis', 		'shigella_boydii', 
		'shigella_unknown', 	'salmonella_enterica_2017', 'salmonella_enterica_2022',	'vibrio_vulnificus', 	'shigella_dysenteriae'
	], 
):

	# Make sure all the indices are correctly lined up
	assert len(df_genespace) > 0, 'No accessions found in genespace table'
	assert df_genespace.index.equals( df_metadata.index ), 'Accession index mismatch between genespace and metadata tables';
	assert df_genespace.index.equals( df_phenotypes.index ), 'Accession index mismatch between genespace and phenotype tables';

	# Build and apply the species mask
	species_mask 	= df_metadata['species'].isin(species_to_keep);

	df_genespace	= df_genespace.loc[species_mask];
	df_metadata		= df_metadata.loc[species_mask];
	df_phenotypes	= df_phenotypes.loc[species_mask];

	assert len(df_genespace) > 0, 'After removing species, there are no accessions left genespace table'

	return df_genespace, df_metadata, df_phenotypes;

def LoadAndProcessNarmsData(
	included_species_list,
	included_phenotypes_list, 
	genespace_input_file, 
	narms_metadata_2017_table_input_file, 
	narms_metadata_2022_table_input_file,
	narms_phenotypes_2017_table_input_file,
	narms_phenotypes_2022_table_input_file,
):

	# Load genespace data
	z_store_genespace	= zarr.DirectoryStore(genespace_input_file);
	z_root_genespace	= zarr.open_group(z_store_genespace, mode='r');

	# Load all the tabular phenotype & metadata
	df_metadata_2017	= pd.read_csv(narms_metadata_2017_table_input_file);
	df_metadata_2022	= pd.read_csv(narms_metadata_2022_table_input_file);
	df_phenotypes_2017	= pd.read_csv(narms_phenotypes_2017_table_input_file);
	df_phenotypes_2022	= pd.read_csv(narms_phenotypes_2022_table_input_file);

	# Combine/align the genespace and all the tables using a bunch of left joins with the genespace as the anchor
	df_genespace, df_metadata, df_phenotypes = CollateMetadataAndPhenotypes(z_root_genespace, df_metadata_2017, df_metadata_2022, df_phenotypes_2017, df_phenotypes_2022);

	# Remove the species that will not be included
	df_genespace, df_metadata, df_phenotypes = FilterBySpecies(df_genespace, df_metadata, df_phenotypes, species_to_keep=included_species_list);

	# After filtering by species, trim the genespace of genes that contain no variation
	num_unique			= df_genespace.nunique();
	column_idxs_to_drop	= num_unique[num_unique == 1].index;
	df_genespace		= df_genespace.drop(column_idxs_to_drop, axis=1)

	# Keep only the MIC columns in the phenotype table
	# mic_column_names 	= [
	# 	'mic_amikacin', 		'mic_amoxicillin',		'mic_amoxicillin-clavulanic_acid', 	'mic_ampicillin', 
	# 	'mic_azithromycin', 	'mic_aztreonam',		'mic_cefepime', 					'mic_cefotaxime', 
	# 	'mic_cefoxitin', 		'mic_cefquinome',		'mic_ceftazidime', 					'mic_ceftiofur', 
	# 	'mic_ceftriaxone', 		'mic_cephalothin',		'mic_chloramphenicol', 				'mic_ciprofloxacin', 
	# 	'mic_clindamycin', 		'mic_erythromycin',		'mic_florfenicol', 					'mic_gentamicin', 
	# 	'mic_imipenem', 		'mic_kanamycin',		'mic_meropenem', 					'mic_nalidixic_acid', 
	# 	'mic_olistin', 			'mic_piperacillin',		'mic_streptomycin', 				'mic_sulfamethoxazole', 
	# 	'mic_sulfisoxazole', 	'mic_telithromycin',	'mic_tetracycline', 				'mic_trimethoprim', 
	# 	'mic_trimethoprim-sulfamethoxazole'
	# ];

	df_phenotypes		= df_phenotypes[ included_phenotypes_list ];

	# Export the below data and parameters
	genespace_matrix	= df_genespace.to_numpy();
	phenotype_matrix	= df_phenotypes.to_numpy();

	accession_ids 		= df_genespace.index.to_numpy();
	phenotype_names		= list( df_phenotypes.columns )
	species_vector 		= df_metadata['species'].to_numpy();
	serotype_vector		= df_metadata['serotype'].to_numpy();

	num_accessions 		= genespace_matrix.shape[0];
	num_genes 			= genespace_matrix.shape[1];
	
	return genespace_matrix, phenotype_matrix, accession_ids, species_vector, serotype_vector, phenotype_names, num_accessions, num_genes;

def ShuffleTrainValidation(idxs, random_seed, iteration):

	# Time to train / test split using dummy indices
	dummy_idxs 								= np.arange( 0, len(idxs) ).astype(int);
	
	kf 										= KFold( n_splits=10, shuffle=True, random_state=random_seed );
	dummy_train_idxs, dummy_validation_idxs	= list( kf.split(dummy_idxs) )[iteration];

	return idxs[dummy_train_idxs], idxs[dummy_validation_idxs];

# This protocol is more useful for multi-species prediction (population -> population)
# Here, we carefully manage which individuals make it into the training and validation sets based on their species & phenotypes
# We also balance the number of individuals that make it into each set depending on the @train_counts and @validation_counts arguments
def GetOneTimeTrainValidationIndices(num_accessions, species_included, train_counts, validation_counts, species_vector, phenotype_vector):

	all_idxs			= np.arange(num_accessions);
	train_idxs			= [];
	validation_idxs		= [];

	for (species, train_count, test_count) in zip(species_included, train_counts, validation_counts):

		total_requested_samples	= train_count + test_count;

		# Use the species mask to mark the samples belonging to the species of interest
		# Also use the phenotype mask to mask away samples without a phenotype value
		species_mask			= species_vector == species;
		phenotype_valid_mask	= ~np.isnan(phenotype_vector);
		sample_mask 			= species_mask & phenotype_valid_mask;

		assert total_requested_samples <= np.sum(sample_mask), f'Not enough samples avaiable in [{species}], requested [{total_requested_samples}] / available [{np.sum(sample_mask)}] '

		species_idxs	= all_idxs[sample_mask];
		species_idxs	= np.random.choice(species_idxs, size=(total_requested_samples), replace=False);

		train_idxs.append(species_idxs[:train_count]);
		validation_idxs.append(species_idxs[train_count:]);

	train_idxs		= np.concatenate(train_idxs);
	validation_idxs	= np.concatenate(validation_idxs);

	return train_idxs, validation_idxs;

# This protocol is more useful for within-species 10-fold cross-validation
# Here, we produce 10 training & validation sets for a single phenotype using the whole population
# There's no need to consider the species type. 
# All embeddings are included (though you can still specify included_species at runtime to limit the species inclusions during the unsupervised step)
def Get10FoldCrossValidationIndices(num_accessions, phenotype_vector):

	all_idxs 				= np.arange(num_accessions);
	phenotype_valid_mask	= ~np.isnan(phenotype_vector);
	valid_idxs 				= all_idxs[phenotype_valid_mask]

	train_idxs_list			= [];
	validation_idxs_list	= [];

	for i in range(0, 10):
		train_idxs, validation_idxs = ShuffleTrainValidation( valid_idxs, random_seed=123, iteration=i);

		train_idxs_list.append( train_idxs );
		validation_idxs_list.append( validation_idxs );

	return train_idxs_list, validation_idxs_list

def DeriveAccessionColorScheme(num_accessions, species_vector, serotype_vector):

	# 21 (supposedly) distinct colors
	kellys_21_colors = [
		'#2B3514', '#1D1D1D', '#EBCE2B', '#702C8C', '#DB6917', 
		'#96CDE6', '#BA1C30', '#C0BD7F', '#7F7E80', '#5FA641', 
		'#D485B2', '#4277B6', '#DF8461', '#463397', '#E1A11A', 
		'#91218C', '#E8E948', '#7E1510', '#92AE31', '#6F340D', 
		'#D32B1E',
	];

	default_color_mask				= np.repeat( '#303030', num_accessions );
	species_color_mask				= np.repeat( '#303030', num_accessions );
	serotype_color_mask				= np.repeat( '#303030', num_accessions );
	misclassification_color_mask	= np.repeat( '#303030', num_accessions ); # This one, the user will populate dynamically``

	# Provide a color scheme for:
	
	# a.) Species
	for i, value in enumerate( Counter( species_vector ).most_common(21) ):

		if i >= len(kellys_21_colors): break; # Ran out of colors, the remaining species will be colored gray

		species_color_mask[ species_vector == value ] = kellys_21_colors[i];


	# b.) Serotype classification
	for i, value in enumerate( Counter( serotype_vector ).most_common(21) ):

		if i >= len(kellys_21_colors): break; # Ran out of colors, the remaining serotypes will be colored gray

		serotype_color_mask[ serotype_vector == value ] = kellys_21_colors[i];


	return default_color_mask, species_color_mask, serotype_color_mask, misclassification_color_mask;

def SaveEmbeddingsToDisk(
	embedding_matrix, genotype_ids, current_epoch, total_epochs, filename,
	genespace_input_file,
	embedding_dimension,
	subset_length,
	num_layers,
	num_heads,
	proj_dropout_prob,
	attn_dropout_prob,
	num_warmup_steps,
	weight_decay
):

	# Reference the Zarr store
	z_store_out	= zarr.DirectoryStore(filename);
	z_root_out	= zarr.group(store=z_store_out);
	compressor	= Blosc(cname='zstd', clevel=5, shuffle=2);

	num_embeddings_rows, num_embedding_columns = embedding_matrix.shape[0], embedding_matrix.shape[1];

	# Test to see if the array needs to be created
	if current_epoch == 0:

		# Create hierarchy
		z_genos_out			= z_root_out.create_group('genotypes');
		z_ids_out			= z_genos_out.array( 'ids', genotype_ids, dtype=object, object_codec=VLenUTF8() );

		# Create storage arrays
		z_embeddings_out = z_genos_out.zeros(
			'embeddings', 
			shape=(total_epochs, num_embeddings_rows, num_embedding_columns), 
			chunks=(1, num_embeddings_rows, num_embedding_columns), 
			dtype=np.float64, 
			compressor=compressor
		);

		# Populate metadata fields
		z_root_out.attrs['creation_date'] 			= datetime.datetime.now().strftime('%b_%d_%Y__%H:%M:%S');
		z_root_out.attrs['genespace_input_file']	= f'{genespace_input_file}';
		z_root_out.attrs['embedding_dimension']		= f'{embedding_dimension}';
		z_root_out.attrs['subset_length']			= f'{subset_length}';
		z_root_out.attrs['num_layers']				= f'{num_layers}';
		z_root_out.attrs['num_heads']				= f'{num_heads}';
		z_root_out.attrs['proj_dropout_prob']		= f'{proj_dropout_prob}';
		z_root_out.attrs['attn_dropout_prob']		= f'{attn_dropout_prob}';
		z_root_out.attrs['num_warmup_steps']		= f'{num_warmup_steps}';
		z_root_out.attrs['weight_decay']			= f'{weight_decay}';

	else:
		z_embeddings_out = z_root_out['genotypes']['embeddings'];

	z_embeddings_out[current_epoch] = embedding_matrix.astype(np.float64);

	return;