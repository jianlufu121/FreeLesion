CUDA_VISIBLE_DEVICES=0 python scripts/LIS.py --batch_size 1 \
                                             --config configs/stable-diffusion/v1-finetune_idrid.yaml \
                                             --ckpt /path/to/trained_model/best.ckpt \
                                             --dataset idrid \
                                             --outdir outputs/idrid_sample \
                                             --txt_file dataset/idrid/idrid_val.txt \
                                             --data_root dataset/idrid \
                                             --plms 
