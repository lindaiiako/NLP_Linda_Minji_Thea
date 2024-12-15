CUDA_VISIBLE_DEVICES=$DEVICE PYTHONPATH=$(pwd) python3 Decoding/et_decoder_only_evaluation.py \
    --model_path /data2/linda_minji/et_prediction_gemma \
    --output_dir Decoding/results \
    --model_type gemma \
    --num_beams 5 \
    --num_samples 5 \
    --temperature 0.7 \
    --gpus 4