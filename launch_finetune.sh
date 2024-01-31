CUDA_VISIBLE_DEVICES=0,1,2,3 python3 finetune/finetune.py \
    --model_name i-jepa_ft_run1 --save_dir save/save_run1 --devices 0 1 2 3 \
    --data_dir /share/datasets/imagenet --max_grad_norm 0.0 \
    --train_effective_batch_size 1024 --max_batch_size_per_device 16 \
    --val_effective_batch_size 256 --learning_rate 0.001 --warmup_steps 5 \
    --total_epochs 50 --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --smoothing 0.1 --finetune --master_port 5104 --layer_decay 0.75 \
    --pretrained_path pre-training_weights/IN1K-vit.h.14-300e.pth.tar \
    --input_size 224 --debug 
