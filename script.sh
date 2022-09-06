CUDA_VISIBLE_DEVICES=5 python train.py --exp_dir exp/aidr_decay --dataset_train train_new.txt --dataset_val val_new.txt \
                    --lr 4e-5 --beta 0.05 --max_epoch 1 --save_interval 1000 --pretrained /work_base/baidu_blur/exp/aidr/model/best_model/model.pdparams
