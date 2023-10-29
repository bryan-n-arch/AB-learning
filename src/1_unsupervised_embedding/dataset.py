import numpy as np

from sklearn.utils.random import sample_without_replacement

import torch.tensor
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from fast_differential_space import randint, shuffle_matrix_per_row, generate_differential_abspace

def ChunkPieces(n, k):

	ret = [ [i, i + k] for i in range(0, n, k) ]

	if n % k != 0:
		ret[-1][-1] = n;

	return list( zip(*ret) );

class GenespaceDataset_Dynamic(Dataset):
	def __init__(self, genespace_matrix, total_input_length, prebuffer_size):

		assert total_input_length % 2 == 0, 'Total input size must be divisible by 2'
		
		self.genespace_matrix				= genespace_matrix;

		# General genespace statistics
		self.num_accessions					= self.genespace_matrix.shape[0];
		self.num_genes						= self.genespace_matrix.shape[1];
		self.total_input_length				= total_input_length;
		self.segment_length					= (self.total_input_length // 2);

		# Token types
		self.token_idx_segment_a			= 0;
		self.token_idx_segment_b			= 1;

		# Prebuffer statistics
		self.prebuffer_size					= prebuffer_size;
		self.prebuffer_pointer				= self.prebuffer_size + 1;

		# Prebuffer memory
		self.buffer_ordered_accessions		= np.arange(self.num_accessions).repeat(prebuffer_size).reshape(self.num_accessions, -1).T.astype(np.int32);
		self.buffer_shuffled_accessions		= np.empty_like(self.buffer_ordered_accessions)[:, 0 : self.segment_length * 2];
		self.buffer_differential_genespace	= np.full((prebuffer_size, self.num_genes * 2), False);

	def __getitem__(self, _):

		# Check to see if we need to bulk generate samples
		if self.prebuffer_pointer >= self.prebuffer_size:
			self.populate_prebuffer();
			self.prebuffer_pointer = 0;

		# 1.) Extract the segment sizes bulk pre-generated samples
		# 2.) Get the accession indicies for sentence A and B
		# 3.) Get the differential genespace set(A) - set(B)
		# 4.) Specify the segment token indices
		segment_size_a				= self.buffer_subset_sizes[ self.prebuffer_pointer, 0 ];
		segment_size_b				= self.buffer_subset_sizes[ self.prebuffer_pointer, 1 ];

		segment_idxs_a				= self.buffer_shuffled_accessions[ self.prebuffer_pointer, 0				: segment_size_a 				];
		segment_idxs_b				= self.buffer_shuffled_accessions[ self.prebuffer_pointer, segment_size_a	: segment_size_a + segment_size_b ];

		genespace_left_diff_labels	= self.buffer_differential_genespace[ self.prebuffer_pointer ];

		ab_idxs_a					= np.full_like(segment_idxs_a, fill_value=self.token_idx_segment_a)
		ab_idxs_b					= np.full_like(segment_idxs_b, fill_value=self.token_idx_segment_a)

		self.prebuffer_pointer 		+= 1;

		output = {
			'bert_left_inputs':			torch.tensor(segment_idxs_a),
			'bert_right_inputs':		torch.tensor(segment_idxs_b),
			'bert_ab_a_inputs':			torch.tensor(ab_idxs_a),
			'bert_ab_b_inputs':			torch.tensor(ab_idxs_b),
			'bert_genespace_labels':	torch.tensor(genespace_left_diff_labels).float(),
		};

		return output;

	def __len__(self):
		return 500_000;

	def populate_prebuffer(self):

		# Randomly generate a 2-column matrix of segment lengths
		self.buffer_subset_sizes = randint(
			low=1,
			high=self.segment_length + 1,
			shape_n=self.prebuffer_size,
			shape_m=2,
		);

		# Create a bunch of shuffled accessions we can sample from
		shuffle_matrix_per_row(
			input_matrix=self.buffer_ordered_accessions,
			output_shuffled_matrix=self.buffer_shuffled_accessions,
		);

		generate_differential_abspace(
			input_space_matrix=self.genespace_matrix,
			input_accession_matrix=self.buffer_shuffled_accessions,
			input_subset_sizes=self.buffer_subset_sizes,
			output_differential_space=self.buffer_differential_genespace
		)

		return;

class GenespaceDiskStream:
	def __init__(
		self,
		z_subset_sizes,
		z_shuffled_accessions,
		z_differential_genespace,
		rank,
		world_size
	):

		# World size and rank values
		self.world_size 						= world_size;
		self.rank 								= rank;

		# Zarr handles
		self.z_subset_sizes						= z_subset_sizes;
		self.z_shuffled_accessions				= z_shuffled_accessions;
		self.z_differential_genespace			= z_differential_genespace;

		# Pre-batch statistics
		self.prebatch_num_chunks				= z_subset_sizes.shape[0];
		self.prebatch_chunk_size				= z_subset_sizes.shape[1];

		self.buffer_chunk_idxs 					= np.array([]);
		self.buffer_chunk_pointer 				= 1;
		self.buffer_sample_pointer 				= self.prebatch_chunk_size + 1;

		return;

	def get_next_sample(self):

		# Check if we're out of samples in the current chunk
		if self.buffer_sample_pointer >= self.prebatch_chunk_size:
			
			# If yes, further check if we're out of chunks
			if self.buffer_chunk_pointer >= len(self.buffer_chunk_idxs):
	
				# If yes, reshuffle completely and fetch a new chunk
				self.global_shuffle_indices();
	
			# Fetch a new chunk
			self.fetch_new_chunk();

		segment_size_a				= self.buffer_subset_sizes[ self.buffer_sample_pointer, 0 ];
		segment_size_b				= self.buffer_subset_sizes[ self.buffer_sample_pointer, 1 ];

		segment_idxs_a				= self.buffer_shuffled_accessions[ self.buffer_sample_pointer, 0				: segment_size_a 				  ];
		segment_idxs_b				= self.buffer_shuffled_accessions[ self.buffer_sample_pointer, segment_size_a	: segment_size_a + segment_size_b ];

		genespace_left_diff_labels	= self.buffer_differential_genespace[ self.buffer_sample_pointer ];

		self.buffer_sample_pointer	+= 1;

		return segment_idxs_a, segment_idxs_b, genespace_left_diff_labels;

	def fetch_new_chunk(self):

		# Bulk read
		self.buffer_subset_sizes			= self.z_subset_sizes[ self.buffer_chunk_pointer ];
		self.buffer_shuffled_accessions		= self.z_shuffled_accessions[ self.buffer_chunk_pointer ];
		self.buffer_differential_genespace	= self.z_differential_genespace[ self.buffer_chunk_pointer ];

		# Locally, start at the beginning of the chunk
		# Also, increment the chunk pointer
		self.buffer_sample_pointer 			= 0;
		self.buffer_chunk_pointer			+= 1;

	# 	return;

	def global_shuffle_indices(self):

		# Generate new shuffled chunk indicies
		chunk_idxs 						= np.arange(self.prebatch_num_chunks);
		np.random.shuffle(chunk_idxs);
		self.buffer_chunk_idxs 			= np.split(chunk_idxs, self.world_size)[self.rank];

		# Globally, reset the chunk pointer to the beginning
		self.buffer_chunk_pointer		= 0;

		return;

class GenespaceDataset_Disk(Dataset):
	def __init__(
		self, 
		genespace_matrix, 
		z_subset_sizes, 
		z_shuffled_accessions, 
		z_differential_genespace, 
		transformer_input_length, 
		rank, 
		world_size
	):

		assert transformer_input_length % 2 == 0, 'Total input size must be divisible by 2'
		assert len(z_subset_sizes) == len(z_shuffled_accessions) and len(z_subset_sizes) == len(z_differential_genespace)
		
		# General genespace statistics
		self.num_accessions					= genespace_matrix.shape[0];
		self.num_genes						= genespace_matrix.shape[1];
		self.transformer_input_length		= transformer_input_length;

		# World size and rank values
		self.world_size 					= world_size;
		self.rank 							= rank;

		# Zarr stream
		self.datastream 					= GenespaceDiskStream(
			z_subset_sizes,
			z_shuffled_accessions,
			z_differential_genespace,
			rank,
			world_size
		)

		# Token types
		self.token_idx_segment_a			= 0;
		self.token_idx_segment_b			= 1;


	def __getitem__(self, _):

		segment_idxs_a, segment_idxs_b, genespace_left_diff_labels = self.datastream.get_next_sample();

		ab_idxs_a = np.full_like(segment_idxs_a, fill_value=self.token_idx_segment_a)
		ab_idxs_b = np.full_like(segment_idxs_b, fill_value=self.token_idx_segment_b)

		output = {
			'bert_left_inputs':			torch.tensor(segment_idxs_a),
			'bert_right_inputs':		torch.tensor(segment_idxs_b),
			'bert_ab_a_inputs':			torch.tensor(ab_idxs_a),
			'bert_ab_b_inputs':			torch.tensor(ab_idxs_b),
			'bert_genespace_labels':	torch.tensor(genespace_left_diff_labels).float(),
		};

		return output;

	def __len__(self):
		return 500_000;

# class GenespaceDataset_Disk(Dataset):
# 	def __init__(
# 		self, 
# 		genespace_matrix, 
# 		z_subset_sizes, 
# 		z_shuffled_accessions, 
# 		z_differential_genespace, 
# 		total_input_length, 
# 		rank, 
# 		world_size
# 	):

# 		assert total_input_length % 2 == 0, 'Total input size must be divisible by 2'
# 		np.random.seed(123);
		
# 		self.genespace_matrix				= genespace_matrix;

# 		# General genespace statistics
# 		self.num_accessions					= self.genespace_matrix.shape[0];
# 		self.num_genes						= self.genespace_matrix.shape[1];
# 		self.total_input_length				= total_input_length;
# 		self.sentence_length				= (self.total_input_length // 2);

# 		# World size and rank values
# 		self.world_size 					= world_size;
# 		self.rank 							= rank;

# 		# assert z_differential_genespace.chunks[0] == z_shuffled_accessions.chunks[0] and z_differential_genespace.chunks[0] == z_subset_sizes.chunks[0];
# 		# assert z_differential_genespace.chunks[0] % world_size == 0, f'Could not split [{z_differential_genespace.chunks[0]}] between [{world_size}] GPUs';

# 		# Save Zarr handles
# 		self.z_subset_sizes					= z_subset_sizes;
# 		self.z_shuffled_accessions			= z_shuffled_accessions;
# 		self.z_differential_genespace		= z_differential_genespace;

# 		# Pre-buffer statistics
# 		self.prebuffer_size					= z_differential_genespace.shape[1];
# 		self.num_prebuffers 				= z_differential_genespace.shape[0];

# 		# Select a set of pre-buffer indicies (not batch indices) based on rank
# 		self.prebuffer_idxs 				= np.split(np.arange(self.num_prebuffers), world_size)[rank];
# 		self.sample_pointer 				= self.prebuffer_size + 1;
# 		self.prebuffer_pointer				= 0;

# 		# Allocate prebuffer memory
# 		self.buffer_subset_sizes			= np.empty((self.prebuffer_size, 2), dtype=np.int32);
# 		self.buffer_ordered_accessions		= np.arange(self.num_accessions).repeat(self.prebuffer_size).reshape(self.num_accessions, -1).T.astype(np.int32);
# 		self.buffer_shuffled_accessions		= np.empty_like(self.buffer_ordered_accessions)[:, 0 : self.sentence_length * 2];
# 		self.buffer_differential_genespace	= np.full((self.prebuffer_size, self.num_genes * 2), False);

# 		# Token types
# 		self.token_idx_segment_a			= 0;
# 		self.token_idx_segment_b			= 1;

# 	def __getitem__(self, _):

# 		# Check to see if we need to bulk generate samples
# 		if self.sample_pointer >= self.prebuffer_size:
# 			self.populate_prebuffer();
# 			self.sample_pointer			= 0;

# 		# 1.) Extract the subset sizes bulk pre-generated samples
# 		# 2.) Get the accession indicies for sentence A and B
# 		# 3.) Get the differential genespace set(A) - set(B)
# 		# 4.) Extract the domains of both sets
# 		subset_size_a					= self.buffer_subset_sizes[ self.sample_pointer, 0 ];
# 		subset_size_b					= self.buffer_subset_sizes[ self.sample_pointer, 1 ];

# 		subset_idxs_a					= self.buffer_shuffled_accessions[ self.sample_pointer, 0				: subset_size_a 				];
# 		subset_idxs_b					= self.buffer_shuffled_accessions[ self.sample_pointer, subset_size_a	: subset_size_a + subset_size_b ];

# 		genespace_left_diff_labels		= self.buffer_differential_genespace[ self.sample_pointer ];

# 		ab_idxs_a 						= np.full_like(subset_idxs_a, fill_value=self.token_idx_segment_a)
# 		ab_idxs_b 						= np.full_like(subset_idxs_b, fill_value=self.token_idx_segment_a)

# 		self.sample_pointer 			+= 1;

# 		output = {
# 			'bert_left_inputs':			torch.tensor(subset_idxs_a),
# 			'bert_right_inputs':		torch.tensor(subset_idxs_b),
# 			'bert_ab_a_inputs':			torch.tensor(ab_idxs_a),
# 			'bert_ab_b_inputs':			torch.tensor(ab_idxs_b),
# 			'bert_genespace_labels':	torch.tensor(genespace_left_diff_labels).float(),
# 		};

# 		# 'bert_left_inputs':			torch.tensor(segment_idxs_a),
# 		# 'bert_right_inputs':		torch.tensor(segment_idxs_b),
# 		# 'bert_ab_a_inputs':			torch.tensor(ab_idxs_a),
# 		# 'bert_ab_b_inputs':			torch.tensor(ab_idxs_b),
# 		# 'bert_genespace_labels':	torch.tensor(genespace_left_diff_labels).float(),

# 		return output;

# 	def __len__(self):
# 		return 500_000;

# 	def populate_prebuffer(self):

# 		# Check to see if we need to generate a new set of indices
# 		if self.prebuffer_pointer >= len(self.prebuffer_idxs):

# 			# Generate new shuffled indicies
# 			idxs 								= np.arange(self.num_prebuffers);
# 			np.shuffle(idxs);
# 			self.prebuffer_idxs 				= np.split(idxs, self.world_size)[self.rank];

# 			# Reset index
# 			self.prebuffer_pointer 				= 0;

# 		# Calculate the offset for reading into the Zarr
# 		prebuffer_idx 							= self.prebuffer_idxs[self.prebuffer_pointer];
# 		# start, end 								= prebuffer_idx * self.prebuffer_size, (prebuffer_idx + 1) * self.prebuffer_size;

# 		self.buffer_subset_sizes[:]				= self.z_subset_sizes[prebuffer_idx];
# 		self.buffer_shuffled_accessions[:]		= self.z_shuffled_accessions[prebuffer_idx];
# 		self.buffer_differential_genespace[:]	= self.z_differential_genespace[prebuffer_idx];

# 		self.prebuffer_pointer					+= 1;

# 		return;

def batched_collate_with_padding_ab(batch, segment_length, embedding_pad_idx, ab_pad_idx):

	# Collect all the left and right samples into their own lists
	left_inputs, right_inputs, ab_a_inputs, ab_b_inputs, genespace_labels = zip(* [
		(
			x['bert_left_inputs'], 
			x['bert_right_inputs'], 
			x['bert_ab_a_inputs'], 
			x['bert_ab_b_inputs'], 
			x['bert_genespace_labels']
		) for x in batch
	] );

	left_padded_inputs_batch 	= pad_packed_sequence( pack_sequence(left_inputs, enforce_sorted=False), batch_first=True, padding_value=embedding_pad_idx, total_length=segment_length )[0];
	right_padded_inputs_batch 	= pad_packed_sequence( pack_sequence(right_inputs, enforce_sorted=False), batch_first=True, padding_value=embedding_pad_idx, total_length=segment_length )[0];

	ab_a_padded_inputs_batch	= pad_packed_sequence(pack_sequence(ab_a_inputs, enforce_sorted=False), batch_first=True, padding_value=ab_pad_idx, total_length=segment_length)[0];
	ab_b_padded_inputs_batch	= pad_packed_sequence(pack_sequence(ab_b_inputs, enforce_sorted=False), batch_first=True, padding_value=ab_pad_idx, total_length=segment_length)[0];

	return {
		'bert_inputs' 			: torch.hstack([left_padded_inputs_batch, right_padded_inputs_batch]),
		'bert_ab_inputs' 		: torch.hstack([ab_a_padded_inputs_batch, ab_b_padded_inputs_batch]),
		'bert_genespace_labels'	: torch.vstack(genespace_labels)
	};

