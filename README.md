# Iterative Nullspace Projection (INLP)

This repository contains the code for the experiments and algorithm from the paper ["Null it out: guarding protected attributes by iterative nullspsace projection"](https://arxiv.org/abs/2004.07667) (accepted as a long paper in ACL 2020).

# Algorithm

The implementaion of the Iterative Nullspace Projection (INLP) method is available under `src/inlp-oop`. Given a dataset of examples `X` and annotation for any kind of information `Z` we want to remove from `X` (e.g. gender, sentiment, tense) the algorithm learns a proejction matrix `P` which aims to exhaustively remove the ability to linearly predict `Z` from `X` ("linear guardness"). 

`src/inlp-oop` contains an implementations of the algorithm that allows learning a projection matrix for several common objectives, such as neutralizing the information captured by linear classifiers (a classification objective), or by linear siamese networks (a metric learning objective). The notebook `usage_example.ipynb` demonstrates the use of the algorithm for those purposes. A more bare-bone implementaton of the same algorithm for the common use case of classification (removing the ability to classify `Z` based on `X` - the focus of the paper) is avaialble under `src/debias.py`.

# Experiments

Start a new virtual environment:
```sh
conda create -n null_space python=3.7 anaconda
conda activate null_space
```

## Setup
download the data used for this project:
```sh
./download_data.sh
```


## Word Embedding experiments (Section 6.1 in the paper)

```py

python src/data/filter_vecs.py \
--input-path data/embeddings/glove.42B.300d.txt \
--output-dir data/embeddings/ \
--top-k 150000  \
--keep-inherently-gendered  \
--keep-names 
```

And run the notebook `word_vectors_debiasing.ipynb` (under "`notebooks`")

## Controlled Demographic experiments (Section 6.2 in the paper)


export PYTHONPATH=/path_to/nullspace_projection
```sh 
./run_deepmoji_debiasing.sh
```


## Bias Bios experiments (Section 6.3 in the paper)

Assumes the bias-in-bios dataset from [De-Arteaga, Maria, et al. 2019](https://arxiv.org/abs/1901.09451) saved at `data/biasbios/BIOS.pkl`.


```py
python src/data/create_dataset_biasbios.py \
        --input-path data/biasbios/BIOS.pkl \
        --output-dir data/biasbios/ \
        --vocab-size 250000
```


```sh
./run_bias_bios.sh
```

And run the notebooks `biasbios_fasttext.ipynb` and `biasbios_bert.ipynb`.
