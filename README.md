# Iterative Nullspace Projection (INLP)

This repository contains the code for the experiments and algorithm from the paper ["Null it out: guarding protected attributes by iterative nullspsace projection"](https://arxiv.org/abs/2004.07667) (accepted as a long paper in ACL 2020).

To cite:

```@article{DBLP:journals/corr/abs-2004-07667,
  author    = {Shauli Ravfogel and
               Yanai Elazar and
               Hila Gonen and
               Michael Twiton and
               Yoav Goldberg},
  title     = {Null It Out: Guarding Protected Attributes by Iterative Nullspace
               Projection},
  journal   = {CoRR},
  volume    = {abs/2004.07667},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.07667},
  archivePrefix = {arXiv},
  eprint    = {2004.07667},
  timestamp = {Tue, 21 Apr 2020 16:51:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-07667.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


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

python src/data/to_word2vec_format.py data/embeddings/glove.42B
.300d.txt

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

# Models
We [release](
https://storage.cloud.google.com/ai2i/nullspace/after-gender-projection/gender-projection.zip) the following models and projection layers: "debiased" GloVe embeddings (`glove.42B.300d.projected.txt`) and "gender-neutralizing projection" (`P.glove.dim=300.iters=35.npy`) from Section 6.1 in the paper, and BERT-base "gender-neutralizing projection" (`P.bert_base.iters=300.npy`), over the biographies dataset, from Section 6.3 in the paper. `glove.42B.300d.projected.txt` was created by applying the transformation `P.glove.dim=300.iters=35.npy` over the original 300-dim GloVe embeddings. Note that the BERT projection `P.bert_base.iters=300.npy` is designed to remove the ability to predict gender from the CLS token, over a specific profession-prediction dataset. It is to be applied on layer 12 of BERT-base,a nd requires the finetuning of the subsequent lienar layer. 

**Usage guidelines**: We urge practitioners *not* to treat those as "gender-neutral embeddings": naturally, as a research paper, the debiasing process was guided by one relatively simple definition of gender association, and was evaluated only on certain benchmarks. As such, it is likely that various gender-related biases are still present in the vectors. Rather, we hope that this model would encourage the community to explore which kinds of biases were mitigated by our intervention -- and which were not, shedding light on thw ways by which bias is manifested.
