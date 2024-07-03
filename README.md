# AB-learning: Learning whole genomes from genespaces.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

AB-learning is an deep-learning methodology for learning real-valued whole-genome latent vectors (embeddings) without explict supervision by relying on semantic signals derived from genes.

We evaluate AB-learning on a collection of bacterial genomes provided by the CDC's [NARMS project](https://www.cdc.gov/narms/index.html). The resulting genome embeddings reflect important biological and epidemiological characteristics like serotype delination, food source preference, and emergence timepoint. They are also competitive in genomic prediction settings (i.e. antibiotic resistance classification) when compared to prior studies.

## Getting Started

Follow these steps to get started with AB-learning:

1. **Installation**: Clone the repository and install the required dependencies.
   ```bash
   git clone https://github.com/bryan-n-arch/AB-learning.git
   cd AB-learning
   pip install -r requirements.txt
	```

2. **Acquire genomic data**: This model is designed to work on the genespaces from bacterial genomes. To run this example, you will need the genespace files, NARMS metadata, and NARMS phenotypes. These are found at 10.5061/dryad.vx0k6djzn

3. **Create pre-batches**: To avoid overhead during training, input sentences and output AB-genespaces need to computed ahead of time. See build_prebatch.sh for an example
   ```bash
    cd AB-learning
	./src/0_prebatching/build_prebatch.sh
	```

2. **Run model**: To generate genome embeddings, run the model and specify how many GPUs are needed. For a dataset of this size, 1 GPU can successfully train the model in a day. An example submission can be seen in train_model.sh.
   ```bash
    cd AB-learning
	./src/1_unsupervised_embedding/train_model.sh
	```

NOTE: These run scripts are designed to be ran on a supercomputing SLURM system. To run them locally, manually replace `$SLURM_ARRAY_TASK_ID` variables with a local rank for each process in the total `$WORLD_SIZE`.

If you use these models, please cite the following paper:
```
@article{naidenov2023,
  title={AB-learning},
  author={Naidenov, Bryan and Chen, Charles},
  journal={arXiv preprint arXiv:0000000000},
  year={2023}
}
```
