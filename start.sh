SAMPLE_RATE=16000

tensorboard --logdir=./logs --bind_all --port=8008 &

python src/preprocess_1.py --data_dir ./data/ --output_dir ./processed --target_sr $SAMPLE_RATE --target_length 5 --sample_multiplier 8 --mmap
python src/Pretrained.py --sample_rate $SAMPLE_RATE --data_dir ./processed
