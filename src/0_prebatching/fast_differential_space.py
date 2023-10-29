import time

import numpy as np
from numba import jit, prange

class Timer:
	def __init__(self, start_msg='Starting now', end_msg='Timer took {}'):
		self.start_msg	= start_msg;
		self.end_msg	= end_msg;
		return;
	def __enter__(self):
		if self.start_msg is not None:
			print(self.start_msg);
		self.start		= time.perf_counter()
		return self;
	def __exit__(self, *args):
		end				 = time.perf_counter();
		secs			 = end - self.start
		total_time		 = time.strftime('%Hh %Mm %Ss', time.gmtime(secs))
		print( self.end_msg.format(total_time) );

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def randint(
	low, high, shape_n, shape_m
):
	return np.random.randint(low=low, high=high, size=(shape_n, shape_m));

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def shuffle_matrix_per_row(
	input_matrix,
	output_shuffled_matrix
):
	max_output_len = output_shuffled_matrix.shape[1];

	for i in prange( len(input_matrix) ):
		output_shuffled_matrix[i] = np.random.permutation( input_matrix[i] )[0 : max_output_len];

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def generate_differential_genespace(
	num_samples,
	num_genes,
	genespace_matrix,
	scrambled_accessions,
	subset_sizes,
	differential_genespace,
):
	for i in prange(num_samples):

		# Extract the accession indicies for this row
		accession_subset_a		= scrambled_accessions[ i, 0					: subset_sizes[i, 0] ];
		accession_subset_b		= scrambled_accessions[ i, subset_sizes[i, 0] 	: subset_sizes[i, 0] + subset_sizes[i, 1] ];

		# Identify tight/loose columns with respect to the accession subsets
		tight_genespace_mask_a 	= genespace_matrix[accession_subset_a, :].sum(0) == subset_sizes[i, 0];
		tight_genespace_mask_b 	= genespace_matrix[accession_subset_b, :].sum(0) == subset_sizes[i, 1];

		loose_genespace_mask_a 	= genespace_matrix[accession_subset_a, :].sum(0) > 0;
		loose_genespace_mask_b 	= genespace_matrix[accession_subset_b, :].sum(0) > 0;

		# Compute the differential genespace using the above masks and the XOR and AND operators
		# 	- This is equivalent to a set difference (to the left)
		tight_genespace_a_diff	= tight_genespace_mask_a ^ (tight_genespace_mask_a & tight_genespace_mask_b);
		loose_genespace_a_diff	= loose_genespace_mask_a ^ (loose_genespace_mask_a & loose_genespace_mask_b);

		# Write results to differential genespace matrix
		differential_genespace[i, 0 		: num_genes]		= tight_genespace_a_diff;
		differential_genespace[i, num_genes : num_genes * 2]	= loose_genespace_a_diff;

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def generate_differential_abspace(
	input_space_matrix,
	input_accession_matrix,
	input_subset_sizes,
	output_differential_space
):
	num_rows 				= len(input_accession_matrix);
	output_matrix_half_size = input_space_matrix.shape[1];
	output_matrix_full_size = output_matrix_half_size * 2;

	for i in prange(num_rows):

		# Extract the accession indicies for this row
		accession_subset_a		= input_accession_matrix[ i, 0							: input_subset_sizes[i, 0] ];
		accession_subset_b		= input_accession_matrix[ i, input_subset_sizes[i, 0] 	: input_subset_sizes[i, 0] + input_subset_sizes[i, 1] ];

		# Identify tight columns with respect to the accession subsets
		tight_space_mask_a 		= input_space_matrix[accession_subset_a, :].sum(0) == input_subset_sizes[i, 0];
		tight_space_mask_b 		= input_space_matrix[accession_subset_b, :].sum(0) == input_subset_sizes[i, 1];

		loose_space_mask_a 		= input_space_matrix[accession_subset_a, :].sum(0) > 0;
		loose_space_mask_b 		= input_space_matrix[accession_subset_b, :].sum(0) > 0;

		# Compute the differential space using the above masks and the XOR and AND operators
		# 	- This is equivalent to a set difference (to the left)
		tight_genespace_a_diff 	= tight_space_mask_a ^ (tight_space_mask_a & tight_space_mask_b);
		loose_genespace_a_diff	= loose_space_mask_a ^ (loose_space_mask_a & loose_space_mask_b);

		# Write results to differential space matrix
		output_differential_space[i, 0 							: output_matrix_half_size]	= tight_genespace_a_diff;
		output_differential_space[i, output_matrix_half_size 	: output_matrix_full_size]	= loose_genespace_a_diff;

def UnitTest(genespace_matrix, scrambled_accessions, subset_sizes, differential_genespace):

	def mask_to_idxs(mask):
		return np.arange(len(mask))[mask.astype(bool)];

	def genespace_left_difference(genespace_left, genespace_right):

		left_idxs, right_idxs 		= mask_to_idxs(genespace_left), mask_to_idxs(genespace_right);
		unique_left_idxs 			= left_idxs[ ~np.isin( left_idxs, right_idxs ) ]

		labels 						= np.zeros_like(genespace_left);
		labels[unique_left_idxs]	= 1;

		return labels;

	assert len(scrambled_accessions) == len(subset_sizes) and len(subset_sizes) == len(differential_genespace);

	for accessions, subset_size, diff_genespace in zip(scrambled_accessions, subset_sizes, differential_genespace):

		subset_size_a, subset_size_b	= subset_size[0], subset_size[1];

		subset_idxs_a					= accessions[ 0				: subset_size_a ];
		subset_idxs_b					= accessions[ subset_size_a	: subset_size_a + subset_size_b ];

		tight_genespace_mask_a 			= genespace_matrix[subset_idxs_a, :].sum(0) == subset_size_a;
		tight_genespace_mask_b 			= genespace_matrix[subset_idxs_b, :].sum(0) == subset_size_b;
		loose_genespace_mask_a 			= genespace_matrix[subset_idxs_a, :].sum(0) > 0;
		loose_genespace_mask_b 			= genespace_matrix[subset_idxs_b, :].sum(0) > 0;

		tight_genespace_a_diff			= genespace_left_difference(tight_genespace_mask_a, tight_genespace_mask_b)
		loose_genespace_a_diff			= genespace_left_difference(loose_genespace_mask_a, loose_genespace_mask_b)
		# tight_genespace_a_diff			= tight_genespace_mask_a ^ (tight_genespace_mask_a & tight_genespace_mask_b);
		# loose_genespace_a_diff			= loose_genespace_mask_a ^ (loose_genespace_mask_a & loose_genespace_mask_b);

		genespace_left_diff_labels		= np.concatenate([tight_genespace_a_diff, loose_genespace_a_diff]);

		assert np.array_equal(genespace_left_diff_labels, diff_genespace);

	return;

# if __name__ == '__main__':

# 	# Create fake genespace matrix
# 	genespace_matrix = np.random.randint( 0, 2, size=(16_228, 27_508), dtype=np.int8 )

# 	# Run code once to compile
# 	sample_genespaces_from_accessions(n_samples=10_000, sentence_length=10, genespace_matrix=genespace_matrix);
# 	# sample_genespaces_from_accessions_batch(n_samples=10_000, sentence_length=10, genespace_matrix=genespace_matrix);

# 	# with Timer(start_msg='Single-threaded version (320,000 samples)', end_msg='Took {}'):
# 	# 	scrambled_accessions, subset_sizes, differential_genespace = sample_genespaces_from_accessions(
# 	# 		n_samples=320_000, sentence_length=10, genespace_matrix=genespace_matrix
# 	# 	);
# 	n_threads						= 20;
# 	n_samples						= 200_000;
# 	chunk_size						= n_samples // n_threads;

# 	with ThreadPool(n_threads) as pool:
# 		with Timer(start_msg=f'Multi-threaded version (20,000 (x{n_threads}) samples)', end_msg='Took {}'):
# 			out						= pool.map(partial(sample_genespaces_from_accessions, sentence_length=10, genespace_matrix=genespace_matrix), [chunk_size] * n_threads);

# 			scrambled_accessions_list, subset_sizes_list, differential_genespace_list = zip(*out);
			 
# 			# scrambled_accessions	= np.vstack(tuple(scrambled_accessions_list));
# 			# subset_sizes			= np.vstack(tuple(subset_sizes_list));
# 			# differential_genespace	= np.vstack(tuple(differential_genespace_list));



# 		# with Timer(start_msg=f'Multi-threaded, write-to-matrix version (20,000 (x{n_threads}) samples)', end_msg='Took {}'):
# 		# 	scrambled_accessions	= np.empty( (n_samples, genespace_matrix.shape[0]), dtype=np.int32 );
# 		# 	subset_sizes			= np.empty( (n_samples, 2), dtype=np.int32 );
# 		# 	# differential_genespace	= np.full( (n_samples, genespace_matrix.shape[1] * 2), False );
# 		# 	differential_genespace	= np.empty( (n_samples, genespace_matrix.shape[1] * 2), dtype=np.bool );

# 		# 	input_tuples			= [ 
# 		# 		(scrambled_accessions[i * chunk_size : i * chunk_size + chunk_size], subset_sizes[i * chunk_size : i * chunk_size + chunk_size], differential_genespace[i * chunk_size : i * chunk_size + chunk_size]) for i in range(n_threads) 
# 		# 	];

# 		# 	pool.map(partial(sample_genespaces_from_accessions_sharedmem, n_samples=chunk_size, sentence_length=10, genespace_matrix=genespace_matrix), input_tuples);

# 	# print(scrambled_accessions, scrambled_accessions.shape)
# 	# print(subset_sizes, subset_sizes.shape)
# 	# print(differential_genespace, differential_genespace.shape)

# 	# UnitTest(genespace_matrix, scrambled_accessions, subset_sizes, differential_genespace)


# # Total	algo + vstack		:	00h 01m 06s
# # Random number generation	:	00h 00m 16s