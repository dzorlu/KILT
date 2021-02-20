# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
export DATADIR="/hdd/kilt_tasks"
export PASSAGES_PATH=
export INDEX_PATH=
export HF_HOME='/hdd/'



# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

#    --attention_dropout 0.1 \
#    --max_grad_norm 0.1 \
#    --dropout 0.1 \
#    --weight_decay 0.001 \
#    --label_smoothing 0.1 \
#    --fp16 \

python finetune_rag.py \
    --data_dir '/hdd/kilt_tasks' \
    --output_dir '/hdd/kilt_results2' \
    --passages_path '/hdd/kilt' \
    --index_path '/hdd/kilt/index_v0.faiss' \
    --model_name_or_path '/hdd/kilt_pretrained_model' \
    --model_type rag_token \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_train 40000 \
    --n_val 4000 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --adam_epsilon 1e-08 \
    --lr_scheduler linear \
    --learning_rate 1e-05 \
    --num_train_epochs 5 \
    --warmup_steps 1000 \
    --num_sanity_val_steps 10 \
    --gradient_accumulation_steps 2 \
    --index_name custom \
    --logger_name wandb \
    --check_val_every_n_epoch 1 \
    
    
