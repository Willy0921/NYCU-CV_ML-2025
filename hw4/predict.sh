CKPT_DIR=./prompt_ir/u8sq2cn2/checkpoints/epoch39-map24.6237.ckpt
ZIP_NAME=./results/MSELoss_no_aug.zip
python predict.py --ckpt=$CKPT_DIR
zip $ZIP_NAME pred.npz