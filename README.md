# NullSpace


Start a new virtual environment:
```sh
conda create -n null_space python=3.7 anaconda
conda activate null_space
```


## Word Embedding experiments (Section 6.1 in the paper)

Assumes uncased Common Crawl GloVe vectors saved at `data/embeddings`.

```py
python src/data/filter_vecs.py \
        --input-path data/embeddings/glove.42B.300d.txt \
        --output_dir data/embeddings/
        --top-k 150000
```

## Controlled Demographic experiments (Section 6.2 in the paper)

```py
python src/data/deepmoji_split.py \
        --input_dir data/deepmoji/ \
        --output_dir data/deepmoji/ \
        --keep-inherently-gendered \
        --keep-names
```

```sh 
./run_deepmoji_debiasing.sh
```


## Bias Bios experiments (Section 6.3 in the paper)

```sh
./run_bias_bios.sh
```

