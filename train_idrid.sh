CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_idrid.yaml \
                                      -t \
                                      --actual_resume models/ldm/stable-diffusion/sd-v1-4-full-ema.ckpt \
                                      -n exp_idrid \
                                      --gpus 0, \
                                      --data_root dataset/idrid \
                                      --train_txt_file dataset/idrid/idrid_train.txt \
                                      --val_txt_file dataset/idrid/idrid_val.txt
