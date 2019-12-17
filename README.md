# NullSpace


## Word Embedding experiments

## Bias Bios experiments
```
mkdir data/bert_encode_biasbios

python src/data/encode_bert_states.py \
        --input_file data/biasbios/train.pickle \
        --output_dir data/bert_encode_biasbios/ \
        --split train

python src/data/encode_bert_states.py \
        --input_file data/biasbios/dev.pickle \
        --output_dir data/bert_encode_biasbios/ \
        --split dev

python src/data/encode_bert_states.py \
        --input_file data/biasbios/test.pickle \
        --output_dir data/bert_encode_biasbios/ \
        --split test
```

## Controlled Demographic experiments

```py
python src/data/deepmoji_split.py \
        --input_dir data/deepmoji/ \
        --output_dir data/deepmoji/
```

```sh 
./run_deepmoji_debiasing.sh
```