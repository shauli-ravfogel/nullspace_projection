# splitting the deepmoji vector representations into train/dev/test
python src/data/deepmoji_split.py \
        --input_dir data/deepmoji/ \
        --output_dir data/deepmoji/

# running all the ratios between sentiment/demographic
for ratio in 0.5 0.6 0.7 0.8
do
    cd src
    # running a model to predict the main task: sentiment
    allennlp train framework/config/deep_moji.jsonnet \
           --include-package framework \
            -s allen_logs/deep_moji_$ratio \
            -o "{'dataset_reader': {'ratio': $ratio}, 'trainer': {'cuda_device': 0}}"
    cd ..
    # extracting the last layer representation - the one we wish to debias
    mkdir data/emoji_sent_race_$ratio
    python src/deepmoji/last_layer_extraction.py \
            --input_dir data/deepmoji/ \
            --output_dir data/emoji_sent_race_$ratio/ \
            --model src/allen_logs/deep_moji_$ratio/model.tar.gz

    # running the debiasing method on the last layer
    python src/deepmoji/deepmoji_debias.py \
            --input_dir data/emoji_sent_race_$ratio/ \
            --output_dir data/emoji_sent_race_$ratio/ \
            --n 300
done
