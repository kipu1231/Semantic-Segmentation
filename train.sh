VALEPOCH=1
LR=0.00007
python train.py --data_dir $1 --val_epoch $VALEPOCH --lr $LR
