# TODO: create shell script for running the testing code of the baseline model
wget https://www.dropbox.com/s/mhxobhsraafrq3r/model_best.pth-17.tar?dl=1
RESUME='model_best.pth-17.tar?dl=1'
python test.py --resume $RESUME --data_dir $1 --save_dir $2