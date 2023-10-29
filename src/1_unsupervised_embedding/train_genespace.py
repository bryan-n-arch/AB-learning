import os
import datetime
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from dataset import GenespaceDataset_Dynamic, GenespaceDataset_Disk, batched_collate_with_padding_ab
from transformer import AB_Transformer
import multi_gpu_util
from dashboard import TransformerDashboard
from helper import LoadAndProcessNarmsData, DeriveAccessionColorScheme, Get10FoldCrossValidationIndices, SaveEmbeddingsToDisk

import zarr
from numcodecs import blosc, Blosc, VLenUTF8
import click

class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling'''

	def __init__(self, optimizer, d_model, n_warmup_steps):
		self._optimizer = optimizer
		self.n_warmup_steps = n_warmup_steps
		self.n_current_steps = 0
		self.init_lr = np.power(d_model, -0.5)

	def step_and_update_lr(self):
		"Step with the inner optimizer"
		self._update_learning_rate()
		# self._optimizer.step()

	def zero_grad(self):
		"Zero out the gradients by the inner optimizer"
		self._optimizer.zero_grad()

	def _get_lr_scale(self):
		return np.min([
			np.power(self.n_current_steps, -0.5),
			np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

	def _update_learning_rate(self):
		''' Learning rate scheduling per step '''

		self.n_current_steps += 1
		lr = self.init_lr * self._get_lr_scale()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr

def Train(model, train_loader , optimizer, scaler, optim_schedule, criterion, world_size, global_rank, dashboard):

	model.train();
	
	ab_padding_idx_tensor = torch.tensor([2], requires_grad=False).cuda();

	for batch_idx, data in enumerate(train_loader):

		optimizer.zero_grad();

		# 0. batch_data will be sent into the device(GPU or cpu)
		data = { key: value.cuda() for key, value in data.items() };

		# 1. forward the next_sentence_prediction and masked_lm model
		with torch.cuda.amp.autocast():

			# Generate logits from inputs
			output_class_logits 	= model.forward( data['bert_inputs'], data['bert_ab_inputs'], ab_padding_idx_tensor );

			# Compare to labels and compute
			loss 					= criterion( output_class_logits, data["bert_genespace_labels"] );

			# Compute positive and negative accuracy
			positives_mask 			= data['bert_genespace_labels'] == 1;
			negatives_mask 			= data['bert_genespace_labels'] == 0;

			predictions				= torch.round( torch.nn.functional.sigmoid(output_class_logits) ).long();
			positive_predictions	= predictions[positives_mask];
			negative_predictions	= predictions[negatives_mask];

			positive_accuracy		= (positive_predictions == 1).sum() / len( positive_predictions.flatten() );
			negative_accuracy		= (negative_predictions == 0).sum() / len( negative_predictions.flatten() );

		# For FP16
		scaler.scale(loss).backward();
		scaler.step(optimizer);
		scaler.update();
		optim_schedule.step_and_update_lr();

		# Aggregate loss, positive accuracy, and negative accuracy to a list
		batch_output = multi_gpu_util.GatherTensor(    
			torch.cat(
				[ 
					loss.view([-1, 1]), 
					positive_accuracy.view([-1, 1]), 
					negative_accuracy.view([-1, 1]) 
				],
				1
			), 
			world_size
		);

		# Merge the list of tensors into a single tensor
		batch_output = torch.cat(batch_output, dim=0);

		# Populate dashboard
		if global_rank == 0:

			# # Gather loss and accuracies
			dash_loss 				= torch.mean(batch_output[:, 0]).cpu().detach().numpy();
			dash_positive_accuracy 	= torch.mean(batch_output[:, 1]).cpu().detach().numpy();
			dash_negative_accuracy 	= torch.mean(batch_output[:, 2]).cpu().detach().numpy();

			dashboard.update_loss(dash_loss, 'bce_loss');

			dashboard.update_train_accuracy(dash_positive_accuracy, 'pos_g_space_accuracy');
			dashboard.update_train_accuracy(dash_negative_accuracy, 'neg_g_space_accuracy');

	return;

def Validate(supervised_model, genome_embeddings, phenotype_matrix, phenotype_names, train_val_idxs_dict, misclassification_color_mask, dashboard):

	X = genome_embeddings;

	# Loop over each column in the phenotype matrix
	for j, phenotype in enumerate(phenotype_names):
		y 										= phenotype_matrix[:, j];
		train_idxs_list, validation_idxs_list 	= train_val_idxs_dict[phenotype];

		# Perform the full 10-fold cross-validation
		y_trus, y_hats = EvaluateSupervisedModel(supervised_model, X, y, train_idxs_list, validation_idxs_list);

		# Compute the accuracy and the f1-score
		accuracy_result = accuracy_score(y_trus, y_hats, normalize=True);
		f1_score_result = f1_score(y_trus, y_hats, average='weighted');

		# dashboard.update_validation_a_accuracy(accuracy_result, phenotype);
		# dashboard.update_validation_b_accuracy(f1_score_result, phenotype);

	return;

def SetupDashboard(num_accessions, accession_ids, species_vector, serotype_vector, phenotype_names):

	# Establish the dashboard and color schemes
	default_color_mask, species_color_mask, serotype_color_mask, misclassification_color_mask = DeriveAccessionColorScheme(
		num_accessions, species_vector, serotype_vector
	);

	dashboard 			= TransformerDashboard(
		titles=(
			'<b>BCE Loss</b>', 		'<b>Train accuracy</b>', 	'<b>Validation accuracy</b>',	'<b>Validation weighted f1-score</b>', 
			'<b>Accessions</b>',	'<b>Species</b>',			'<b>Serotype</b>',				'<b>Classification error</b>'
		),
		loss_step_size=1, 
		training_step_size=1, 
		validation_step_size=5
	);
	kellys_21_colors 	= [
		'#1D1D1D', '#EBCE2B', '#702C8C', '#DB6917', '#96CDE6', 
		'#BA1C30', '#C0BD7F', '#7F7E80', '#5FA641', '#D485B2', 
		'#4277B6', '#DF8461', '#463397', '#E1A11A', '#91218C', 
		'#E8E948', '#7E1510', '#92AE31', '#6F340D', '#D32B1E', 
		'#2B3514'
	];	
	dashboard.register_loss('bce_loss', '#ff4524', has_dash=False);

	dashboard.register_train_accuracies('pos_g_space_accuracy', '#09a5ec', has_dash=False);
	dashboard.register_train_accuracies('neg_g_space_accuracy', '#0973ed', has_dash=False);

	# for phenotype, color in zip(phenotype_names, kellys_21_colors):
	# 	dashboard.register_validation_a_accuracies(phenotype, color, has_dash=False);

	# for phenotype, color in zip(phenotype_names, kellys_21_colors):
	# 	dashboard.register_validation_b_accuracies(phenotype, color, has_dash=False);

	dashboard.register_embedding_names_a(accession_ids);
	dashboard.register_embedding_names_b(species_vector);
	dashboard.register_embedding_names_c(serotype_vector);
	dashboard.register_embedding_names_d( np.array([''] * len(accession_ids)) );

	dashboard.register_embedding_color_mask_a(default_color_mask);
	dashboard.register_embedding_color_mask_b(species_color_mask);
	dashboard.register_embedding_color_mask_c(serotype_color_mask);
	dashboard.register_embedding_color_mask_d(misclassification_color_mask);

	# dashboard.set_train_y_range(1.0);
	# dashboard.set_validation_a_y_range(1.0);
	# dashboard.set_validation_b_y_range(1.0);

	return dashboard, misclassification_color_mask;

def InitalizeSupervisedModel(num_accessions, phenotype_names, phenotype_matrix):

	# Specify parameters for the supervised model step:
	# a.) Establish the supervised model
	# b.) Get the training and validation set for each phenotype (i.e. a dictionary of training/validation indicies)
	# 	- Currently, we use 10-fold cross-validation (but other options are available for species A -> species B prediction)
	# supervised_model 	= KNeighborsClassifier(n_neighbors=5);
	supervised_model 	= LogisticRegression(max_iter=1000);
	train_val_idxs_dict = {};

	for j, phenotype in enumerate(phenotype_names):
		train_idxs_list, validation_idxs_list 	= Get10FoldCrossValidationIndices( num_accessions, phenotype_matrix[:, j] );
		train_val_idxs_dict[phenotype]			= (train_idxs_list, validation_idxs_list);

	return supervised_model, train_val_idxs_dict;

def EvaluateSupervisedModel(model, X, y, train_idxs_list, validation_idxs_list):

	y_hats, y_trus = [], [];

	for train_idxs, validation_idxs in zip(train_idxs_list, validation_idxs_list):

		X_train	= X[train_idxs];
		X_test	= X[validation_idxs];

		y_train	= y[train_idxs];
		y_test 	= y[validation_idxs];

		# For NARMS, linearize MICs:
		y_train	= np.round( np.log2(y_train) );
		y_test	= np.round( np.log2(y_test) );

		# Fit model to training set
		model.fit(X_train, y_train)
		y_predictions = model.predict(X_test);

		y_trus.append( y_test );
		y_hats.append( y_predictions );

	return np.concatenate(y_trus), np.concatenate(y_hats);

def PrintInitizationString(
	genespace_input_file,
	prebatch_input_file,
	narms_metadata_2017_table_input_file,
	narms_metadata_2022_table_input_file,
	narms_phenotypes_2017_table_input_file,
	narms_phenotypes_2022_table_input_file,
	num_accessions,
	num_genes,
	included_species_list,
	included_phenotypes_list,
	model,
	input_length,
	embedding_dimension,
	num_heads,
	num_layers,
	proj_dropout_prob,
	attn_dropout_prob,
	num_warmup_steps,
	weight_decay,
	supervised_model,
	embeddings_file_name,
	log_file_name,
):
	
	# Output the settings
	print(f'date={datetime.datetime.now()}\n')
	
	print(f'genespace_input_file={genespace_input_file}');
	print(f'prebatch_input_file={prebatch_input_file}');
	print(f'narms_metadata_2017_table_input_file={narms_metadata_2017_table_input_file}');
	print(f'narms_metadata_2022_table_input_file={narms_metadata_2022_table_input_file}');
	print(f'narms_phenotypes_2017_table_input_file={narms_phenotypes_2017_table_input_file}');
	print(f'narms_phenotypes_2022_table_input_file={narms_phenotypes_2022_table_input_file}\n');

	print(f'num_accession={num_accessions}');
	print(f'num_genes={num_genes}\n');

	print(f'included_species_list={included_species_list}');
	print(f'included_phenotypes_list={included_phenotypes_list}\n');

	print(f'model={model}');
	print(f'input_length={input_length}');
	print(f'embedding_dimension={embedding_dimension}');
	print(f'depth={num_heads}');
	print(f'n_heads={num_layers}');
	print(f'mlp_ratio=4.0');
	print(f'qkv_bias=True');
	print(f'p={proj_dropout_prob}');
	print(f'attn_p={attn_dropout_prob}');
	print(f'num_warmup_steps={num_warmup_steps}');
	print(f'weight_decay={weight_decay}\n');
	
	print(f'supervised_model={supervised_model}\n');

	print(f'embeddings_file_name={embeddings_file_name}');
	print(f'log_file_name={log_file_name}');

# Remaining TODO:
# 	a.) Extract embeddings and put into supervised model				[GOOD]
# 	b.) Predict cross-validation and compute accuracies and f1-scores	[GOOD]
# 	c.) Allow the dashboard axes and titles to be customized			[GOOD]
# 	d.) Configure the step-sizes for everything							[GOOD]
# 	e.) Set axes ranges													[GOOD]
# 	f.) Sort serotype coloring by size									[Test this]
# 	f.) Prebatched AB-learning											[]
# 	g.) Classification error vector										[]
# 	h.) Multi-threaded linear regression / k-NN							[]
# 	i.) Faster t-SNE													[]
# 	j.) Add smoothing options to dashboard + standard deviation			[]

@click.command()
@click.option('--genespace_input_file', 					'-iz', 																				help='Specifies the genespace warehouse to use.',					type=click.Path(exists=True, file_okay=False))
@click.option('--prebatch_input_file', 						'-ip', 																				help='Specifies prebatched indexing warehouse to use.',				type=click.Path(exists=True, file_okay=False))
@click.option('--narms_metadata_2017_table_input_file', 	'-i_2017_metadata', 																help='Specifies the NARMS 2017 metadata table to use.',				type=click.Path(exists=True, dir_okay=False))
@click.option('--narms_metadata_2022_table_input_file', 	'-i_2022_metadata', 																help='Specifies the NARMS 2022 metadata table to use.',				type=click.Path(exists=True, dir_okay=False))
@click.option('--narms_phenotypes_2017_table_input_file', 	'-i_2017_phenotypes', 																help='Specifies the NARMS 2017 phenotypes table to use.',			type=click.Path(exists=True, dir_okay=False))
@click.option('--narms_phenotypes_2022_table_input_file', 	'-i_2022_phenotypes', 																help='Specifies the NARMS 2022 phenotypes table to use.',			type=click.Path(exists=True, dir_okay=False))
@click.option('--included_phenotypes', 						'-p',						default='mic_apramycin;mic_aztreonam;mic_cephalothin',	help='The phenotypes to test, delimited by semi-colon.',			type=click.STRING)
@click.option('--included_species', 						'-s',						default='salmonella_enterica_2017;shigella_sonnei',		help='The species to include, delimited by semi-colon.',			type=click.STRING)
@click.option('--embedding_dimension', 						'-d',						default=128,											help='The dimension of each embedding.', 							type=click.INT)
@click.option('--input_length', 							'-l',						default=20,												help='Specifies the sentence length.', 								type=click.INT)
@click.option('--num_layers', 								'-nl',						default=4,												help='Specifies the number of attention layers.', 					type=click.INT)
@click.option('--num_heads', 								'-nh',						default=4,												help='Specifies the number of attention heads.', 					type=click.INT)
@click.option('--proj_dropout_prob', 						'-pd', 						default=0.4,											help='Specifies the drop probability for projections.', 			type=click.FLOAT)
@click.option('--attn_dropout_prob', 						'-ad', 						default=0.0,											help='Specifies the drop probability for attention.', 				type=click.FLOAT)
@click.option('--weight_decay', 							'-a', 						default=0.1,											help='Specifies the weight decay to use.', 							type=click.FLOAT)
@click.option('--num_warmup_steps', 						'-w', 						default=5000,											help='Specifies the number of warmup steps for the scheduler',		type=click.INT)
@click.option('--log_file_name', 	   						'-o', 																				help='Specifies the log file name.', 								type=str)
@click.option('--embeddings_file_name',						'-e', 																				help='Specifies the file name to save embeddings into.',			type=str)
@click.option('--model_file_name',							'-m', 																				help='Specifies the file name to save the model into.',				type=str)
@click.option('--local_rank', 	     						'-r', 																				help='Specifies the local rank.', 									type=click.INT)
def Main(
	genespace_input_file, 
	prebatch_input_file,
	narms_metadata_2017_table_input_file,
	narms_metadata_2022_table_input_file,
	narms_phenotypes_2017_table_input_file,
	narms_phenotypes_2022_table_input_file,
	included_phenotypes,
	included_species,
	embedding_dimension, 
	input_length, 
	num_layers, 
	num_heads, 
	proj_dropout_prob, 
	attn_dropout_prob,
	num_warmup_steps,
	weight_decay, 
	log_file_name, 
	embeddings_file_name,
	model_file_name,
	local_rank
):

	blosc.set_nthreads(20);

	# If we generate the indices dynamically, then seperate (unique) random seeds are critical
	# If we use a prebatch Zarr, then we need the random number generator to be fully synchronized
	if prebatch_input_file is None:	np.random.seed(seed=None);
	else:							np.random.seed(seed=123);

	# Load, align, filter/clean, and process all NARMS data (genespace + metadata + phenotypes)
	included_species_list 		= included_species.split(';');
	included_phenotypes_list 	= included_phenotypes.split(';');

	genespace_matrix, phenotype_matrix, accession_ids, species_vector, serotype_vector, phenotype_names, num_accessions, num_genes = LoadAndProcessNarmsData(
		included_species_list,
		included_phenotypes_list,
		genespace_input_file,
		narms_metadata_2017_table_input_file,
		narms_metadata_2022_table_input_file,
		narms_phenotypes_2017_table_input_file,
		narms_phenotypes_2022_table_input_file
	);

	# Specify some parameters
	num_outputs 		= num_genes * 2;

	segment_length 		= input_length // 2;

	input_padding_idx 	= genespace_matrix.shape[0];
	ab_padding_idx 		= 2;

	# Build the model
	model = AB_Transformer(
		num_tokens=num_accessions,
		num_outputs=num_outputs,
		input_padding_idx=input_padding_idx,
		embed_dim=embedding_dimension,
		depth=num_heads,
		n_heads=num_layers,
		mlp_ratio=4.0,
		qkv_bias=True, 
		p=proj_dropout_prob,
		attn_p=attn_dropout_prob,
	)

	# Initalize the model and define the loss
	model, optimizer, genome_embeddings	= multi_gpu_util.InitEnvironment( model, weight_decay, local_rank);
	criterion 							= nn.BCEWithLogitsLoss();

	optim_schedule 						= ScheduledOptim( optimizer, d_model=embedding_dimension, n_warmup_steps=num_warmup_steps );
	world_size 							= torch.distributed.get_world_size();
	global_rank							= torch.distributed.get_rank(); # This is different from the local rank, who's index starts at 0 for each node. The global rank is unique for each process.
	scaler								= torch.cuda.amp.GradScaler();

	# Create datasets and use distributed samplers to scatter samples across the GPU
	if prebatch_input_file is None:
		train_dataset 					= GenespaceDataset_Dynamic(genespace_matrix, total_input_length=input_length, prebuffer_size=75_000);
	else:

		z_store							= zarr.DirectoryStore(prebatch_input_file);
		z_root							= zarr.open_group(z_store, mode='r');

		train_dataset					= GenespaceDataset_Disk( 
			# genespace_matrix, 
			# z_subset_sizes=z_root['subset_sizes'], 
			# z_shuffled_accessions=z_root['shuffled_accessions'], 
			# z_differential_genespace=z_root['differential_genespace'], 
			# transformer_input_length=input_length, 
			# chunk_size=10_000,
			# rank=global_rank, 
			# world_size=world_size

			genespace_matrix, 
			z_root['subset_sizes'], 
			z_root['shuffled_accessions'], 
			z_root['differential_genespace'], 
			input_length, 
			global_rank, 
			world_size
		);
	train_sampler 						= torch.utils.data.distributed.DistributedSampler(train_dataset);

	# # batch_size = 4096 // world_size;
	batch_size = 2048 // world_size;

	train_loader = torch.utils.data.DataLoader( 
		train_dataset, batch_size=batch_size, 
		shuffle=None, # Shuffle must be turned off when using distributed sampling, because the distributed sampler does its own "subset-shuffling"
		num_workers=0, pin_memory=True, sampler=train_sampler,
		# prefetch_factor=1024, 
		collate_fn=partial(
			batched_collate_with_padding_ab, 
			segment_length=segment_length, 
			embedding_pad_idx=input_padding_idx, 
			ab_pad_idx=ab_padding_idx,
		)
	);

	dashboard, misclassification_color_mask	= SetupDashboard(num_accessions, accession_ids, species_vector, serotype_vector, phenotype_names);
	supervised_model, train_val_idxs_dict 	= InitalizeSupervisedModel(num_accessions, phenotype_names, phenotype_matrix);

	# Output the settings
	if global_rank == 0:
		PrintInitizationString(
			genespace_input_file,
			prebatch_input_file,
			narms_metadata_2017_table_input_file,		narms_metadata_2022_table_input_file,
			narms_phenotypes_2017_table_input_file,		narms_phenotypes_2022_table_input_file,
			num_accessions,								num_genes,
			included_species_list,						included_phenotypes_list,
			model,
			input_length,								embedding_dimension,			num_heads,				
			num_layers,									proj_dropout_prob,				attn_dropout_prob,		
			num_warmup_steps,							weight_decay,					supervised_model,
			embeddings_file_name,						log_file_name,
		);

	total_epochs = 700;
	
	for epoch in range(0, total_epochs):

		train_sampler.set_epoch(epoch);
		Train(model, train_loader, optimizer, scaler, optim_schedule, criterion, world_size, global_rank, dashboard);

		if global_rank == 0:

			with torch.no_grad():

				# Extract embeddings into a numpy array
				genome_embeddings_cpu = genome_embeddings(torch.arange(0, genome_embeddings.num_embeddings - 1).cuda()).cpu().detach().numpy();

				if epoch % 50 == 0:
					tsne_model		= TSNE();
					tsne_embeddings	= tsne_model.fit_transform(genome_embeddings_cpu);
					dashboard.update_embeddings(tsne_embeddings);

				if epoch % 25 == 0:
					Validate(supervised_model, genome_embeddings_cpu, phenotype_matrix, phenotype_names, train_val_idxs_dict, misclassification_color_mask, dashboard);

			dashboard.plot(log_file_name);
			SaveEmbeddingsToDisk(
				embedding_matrix=genome_embeddings_cpu, 
				genotype_ids=accession_ids, 
				current_epoch=epoch, 
				total_epochs=total_epochs, 
				filename=embeddings_file_name,
				genespace_input_file=genespace_input_file, 
				embedding_dimension=embedding_dimension, 
				subset_length=segment_length, 
				num_layers=num_layers, 
				num_heads=num_heads, 
				proj_dropout_prob=proj_dropout_prob, 
				attn_dropout_prob=attn_dropout_prob,
				num_warmup_steps=num_warmup_steps,
				weight_decay=weight_decay, 
			);
	# 		if os.path.exists(model_file_name): os.remove(model_file_name) # Delete last version of the model
	# 		torch.save(model.module.state_dict(), model_file_name); 
			print(f'Epoch {epoch}')

	return;

if __name__ == '__main__':
	Main();    

# python -m torch.distributed.launch --port 123456 --nproc_per_node=4 --nnodes=1 src/1_unsupervised_embedding/ab_learning/train_genespace.py \
# --genespace_input_file "data/genespace/narms_2017_and_2022_genespace.dir.zarr" \
# --prebatch_input_file "data/pre_batching/ab_salmonella_enterica_2017.dir.zarr" \
# --narms_metadata_2017_table_input_file "data/phenotypes_and_metadata/narms_metadata_2017.csv" \
# --narms_metadata_2022_table_input_file "data/phenotypes_and_metadata/narms_metadata_2022.csv" \
# --narms_phenotypes_2017_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2017.csv" \
# --narms_phenotypes_2022_table_input_file "data/phenotypes_and_metadata/narms_phenotypes_2022.csv" \
# --included_phenotypes "mic_amoxicillin-clavulanic_acid;mic_azithromycin;mic_cefoxitin;mic_ceftiofur;mic_ceftriaxone;mic_ciprofloxacin;mic_gentamicin;mic_kanamycin;mic_nalidixic_acid;mic_sulfisoxazole;mic_tetracycline;mic_trimethoprim-sulfamethoxazole" \
# --included_species "salmonella_enterica_2017" \
# --embedding_dimension 128 \
# --input_length 10 \
# --num_layers 4 \
# --num_heads 4 \
# --proj_dropout_prob 0.3 \
# --attn_dropout_prob 0.0 \
# --num_warmup_steps 5000 \
# --weight_decay 0.10 \
# --log_file_name "viz/s_enteria_2017_ab.html" \
# --embeddings_file_name "s_enteria_2017_ab.dir.zarr" \
# --model_file_name "models/s_enteria_2017_ab.pt"

