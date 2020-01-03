# NullSpace


Start a new virtual environment:
```sh
conda create -n null_space python=3.7 anaconda
conda activate null_space
```


## Word Embedding experiments (Section 6.1 in the paper)
TODO - complete

## Controlled Demographic experiments (Section 6.2 in the paper)

```py
python src/data/deepmoji_split.py \
        --input_dir data/deepmoji/ \
        --output_dir data/deepmoji/
```

```sh 
./run_deepmoji_debiasing.sh
```


## Bias Bios experiments (Section 6.3 in the paper)

```sh
./run_bias_bios.sh
```

