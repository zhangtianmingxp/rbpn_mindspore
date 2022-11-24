if [ $# != 4 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd RBPN"
  echo "Usage: bash run_eval.sh [DEVICE_ID] [CKPT]  [VAL_PATH] [FILE_LIST] "
  bash "run_eval.sh 0 weights/rbpn.ckpt  /dataset/Vid4 /dataset/Vid4/calendar3.txt"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

rm -rf ./eval
mkdir ./eval
cp -r ../src ./eval
cp -r ../*.py ./eval
cp -r ../*.so ./eval

env > env.log
python ./eval/eval.py  --device_id=$1 --ckpt=$2  --val_path=$3 --file_list=$4 > eval.log 2>&1 &
