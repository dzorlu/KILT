
python consolidate_rag_checkpoint.py \
    --model_type "rag_token" \
    --config_name_or_path "facebook/rag-token-base" \
    --generator_name_or_path "facebook/bart-base" \
    --question_encoder_name_or_path "facebook/dpr-question_encoder-multiset-base" \
    --dest "/hdd/kilt_pretrained_model/"
