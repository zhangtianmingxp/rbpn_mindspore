if [ $# != 4 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd RBPN"
  echo "bash run_standalone_train.sh [DEVICE_ID]  [DATA_DIR] [FILE_LIST] [BATCHSIZE] "
  echo "bash run_standalone_train.sh 0   dataset/vimeo_septuplet/sequencesdataset/vimeo_septuplet/sep_trainlist.txt  4 "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0


rm -rf ./train_rbpn
mkdir ./train_rbpn
cp -r ../src ./train_rbpn
cp -r ../*.py ./train_rbpn
cp -r ../*.so ./train_rbpn
cd ./train_rbpn || exit
python train.py --device_id=$1   \
                       --data_dir=$2  --file_list=$3  --batchSize=$4 > train.log 2>&1 &
