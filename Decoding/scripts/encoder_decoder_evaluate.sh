SCRIPT_PATH="Decoding/et_encoder_decoder_evaluation.py"
MODEL_PATH="/data2/linda_minji/et_prediction_t5/checkpoint-2128"
CUDA_VISIBLE_DEVICES=$DEVICE PYTHONPATH=$(pwd) python3 $SCRIPT_PATH \
    --model_path $MODEL_PATH \
    --output_dir Decoding/results \
    --batch_size 4 \
    --num_beams 5 \
    --num_samples 5 \
    --temperature 0.7 \
    --gpus 4