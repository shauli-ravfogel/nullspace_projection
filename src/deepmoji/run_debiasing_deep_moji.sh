for ratio in 0.5 0.6 0.7 0.8 0.9
do
    for i in 50 80 100 120
    do
        python src/deepmoji/main.py --input_dir data/emoji_sent_race_{$ratio}/ --output_dir data/emoji_sent_race_{$ratio}/ -n $i
    done
done