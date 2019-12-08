for ratio in 0.5 0.6 0.7 0.8 0.9
do
    python src/deepmoji/last_layer_extraction.py \
            --input_dir data/emoji_sent_race/ \
            --output_dir data/emoji_sent_race_$ratio/ \
            --model src/allen_logs/deep_moji_$ratio/model.tar.gz
done