
# creating the dataset folder
mkdir data/bert_encode_biasbios

# Encode text using Bert
for split in 'train' 'dev' 'test'
do
    python src/data/encode_bert_states.py \
        --input_file data/biasbios/$split.pickle \
        --output_dir data/bert_encode_biasbios/ \
        --split $split
done
