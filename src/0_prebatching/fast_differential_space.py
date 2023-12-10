'''A module for generating differential spaces from genespace matrices using optimized numpy code via numba.'''

import time

import numpy as np
from numba import jit, prange

class Timer:
    '''A simple timer class for printing how long a block of code took to run.'''

    def __init__(self, start_msg : str ='Starting now', end_msg : str='Timer took {}'):
        '''Initialize the timer with a start and end message.
        
        Args:
            start_msg (str): The message to print when the timer starts. Defaults to 'Starting now'.
            end_msg (str): The message to print when the timer ends. Defaults to 'Timer took {}'.
        '''
        self.start_msg	= start_msg
        self.end_msg	= end_msg
        self.start      = None

    def __enter__(self):

        if self.start_msg is not None:
            print(self.start_msg)

        self.start		= time.perf_counter()

        return self

    def __exit__(self, *args):

        end				 = time.perf_counter()
        secs			 = end - self.start
        total_time		 = time.strftime('%Hh %Mm %Ss', time.gmtime(secs))

        print( self.end_msg.format(total_time) )

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def randint(
    low, high, shape_n, shape_m
):
    return np.random.randint(low=low, high=high, size=(shape_n, shape_m))

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def shuffle_matrix_per_row(
    input_matrix,
    output_shuffled_matrix
):
    max_output_len = output_shuffled_matrix.shape[1]

    for i in prange( len(input_matrix) ):
        output_shuffled_matrix[i] = np.random.permutation( input_matrix[i] )[0 : max_output_len]

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
        accession_subset_a		= scrambled_accessions[ i, 0					: subset_sizes[i, 0] ]
        accession_subset_b		= scrambled_accessions[ i, subset_sizes[i, 0] 	: subset_sizes[i, 0] + subset_sizes[i, 1] ]

        # Identify tight/loose columns with respect to the accession subsets
        tight_genespace_mask_a 	= genespace_matrix[accession_subset_a, :].sum(0) == subset_sizes[i, 0]
        tight_genespace_mask_b 	= genespace_matrix[accession_subset_b, :].sum(0) == subset_sizes[i, 1]

        loose_genespace_mask_a 	= genespace_matrix[accession_subset_a, :].sum(0) > 0
        loose_genespace_mask_b 	= genespace_matrix[accession_subset_b, :].sum(0) > 0

        # Compute the differential genespace using the above masks and the XOR and AND operators
        # 	- This is equivalent to a set difference (to the left)
        tight_genespace_a_diff	= tight_genespace_mask_a ^ (tight_genespace_mask_a & tight_genespace_mask_b)
        loose_genespace_a_diff	= loose_genespace_mask_a ^ (loose_genespace_mask_a & loose_genespace_mask_b)

        # Write results to differential genespace matrix
        differential_genespace[i, 0 		: num_genes]		= tight_genespace_a_diff
        differential_genespace[i, num_genes : num_genes * 2]	= loose_genespace_a_diff

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def generate_differential_abspace(
    input_space_matrix,
    input_accession_matrix,
    input_subset_sizes,
    output_differential_space
):
    num_rows 				= len(input_accession_matrix)
    output_matrix_half_size = input_space_matrix.shape[1]
    output_matrix_full_size = output_matrix_half_size * 2

    for i in prange(num_rows):

        # Extract the accession indicies for this row
        accession_subset_a		= input_accession_matrix[ i, 0							: input_subset_sizes[i, 0] ]
        accession_subset_b		= input_accession_matrix[ i, input_subset_sizes[i, 0] 	: input_subset_sizes[i, 0] + input_subset_sizes[i, 1] ]

        # Identify tight columns with respect to the accession subsets
        tight_space_mask_a 		= input_space_matrix[accession_subset_a, :].sum(0) == input_subset_sizes[i, 0]
        tight_space_mask_b 		= input_space_matrix[accession_subset_b, :].sum(0) == input_subset_sizes[i, 1]

        loose_space_mask_a 		= input_space_matrix[accession_subset_a, :].sum(0) > 0
        loose_space_mask_b 		= input_space_matrix[accession_subset_b, :].sum(0) > 0

        # Compute the differential space using the above masks and the XOR and AND operators
        # 	- This is equivalent to a set difference (to the left)
        tight_genespace_a_diff 	= tight_space_mask_a ^ (tight_space_mask_a & tight_space_mask_b)
        loose_genespace_a_diff	= loose_space_mask_a ^ (loose_space_mask_a & loose_space_mask_b)

        # Write results to differential space matrix
        output_differential_space[i, 0 							: output_matrix_half_size]	= tight_genespace_a_diff
        output_differential_space[i, output_matrix_half_size 	: output_matrix_full_size]	= loose_genespace_a_diff

def unit_test(genespace_matrix : np.ndarray, scrambled_accessions : np.ndarray, subset_sizes : np.ndarray, differential_genespace : np.ndarray):
    '''A unit test for the generate_differential_genespace function.
    
    Args:
        genespace_matrix (np.ndarray): A binary matrix of shape (num_accessions, num_genes) where
            a 1 in the (i,j) position indicates that accession i has gene j.
        scrambled_accessions (np.ndarray): A matrix of shape (num_samples, num_accessions) where
            each row is a random permutation of the accessions.
        subset_sizes (np.ndarray): A matrix of shape (num_samples, 2) where each row is a random
            pair of numbers that sum to the segment length.
        differential_genespace (np.ndarray): A matrix of shape (num_samples, num_genes * 2) where
            each row is the differential genespace for the corresponding row in scrambled_accessions.
    
    Raises:
        AssertionError: If any of the differential genespace matrices do not match the expected output.
    '''

    def mask_to_idxs(mask):
        return np.arange(len(mask))[mask.astype(bool)]

    def genespace_left_difference(genespace_left, genespace_right):

        left_idxs, right_idxs 		= mask_to_idxs(genespace_left), mask_to_idxs(genespace_right)
        unique_left_idxs 			= left_idxs[ ~np.isin( left_idxs, right_idxs ) ]

        labels 						= np.zeros_like(genespace_left)
        labels[unique_left_idxs]	= 1

        return labels

    assert len(scrambled_accessions) == len(subset_sizes) and len(subset_sizes) == len(differential_genespace)

    for accessions, subset_size, diff_genespace in zip(scrambled_accessions, subset_sizes, differential_genespace):

        subset_size_a, subset_size_b	= subset_size[0], subset_size[1]

        subset_idxs_a					= accessions[ 0				: subset_size_a ]
        subset_idxs_b					= accessions[ subset_size_a	: subset_size_a + subset_size_b ]

        tight_genespace_mask_a 			= genespace_matrix[subset_idxs_a, :].sum(0) == subset_size_a
        tight_genespace_mask_b 			= genespace_matrix[subset_idxs_b, :].sum(0) == subset_size_b
        loose_genespace_mask_a 			= genespace_matrix[subset_idxs_a, :].sum(0) > 0
        loose_genespace_mask_b 			= genespace_matrix[subset_idxs_b, :].sum(0) > 0

        tight_genespace_a_diff			= genespace_left_difference(tight_genespace_mask_a, tight_genespace_mask_b)
        loose_genespace_a_diff			= genespace_left_difference(loose_genespace_mask_a, loose_genespace_mask_b)
        # tight_genespace_a_diff			= tight_genespace_mask_a ^ (tight_genespace_mask_a & tight_genespace_mask_b)
        # loose_genespace_a_diff			= loose_genespace_mask_a ^ (loose_genespace_mask_a & loose_genespace_mask_b)

        genespace_left_diff_labels		= np.concatenate([tight_genespace_a_diff, loose_genespace_a_diff])

        assert np.array_equal(genespace_left_diff_labels, diff_genespace)
