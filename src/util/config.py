"""config param"""

import argparse
import ast

def get_args():
# Training settings
    parser = argparse.ArgumentParser(description='RBPN-mindspore')
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--other_dataset', type=int, default=0, help="use other dataset than vimeo-90k")
    parser.add_argument('--future_frame', type=int, default=1, help="use future frame")
    parser.add_argument('--nFrames', type=int, default=7)
    parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
    parser.add_argument('--data_augmentation', type=ast.literal_eval, default=True)
    parser.add_argument('--model_type', type=str, default='RBPN')
    parser.add_argument('--residual', type=ast.literal_eval, default=False)
    parser.add_argument('--pretrained_sr', default='117_RBPN.ckpt', help='sr pretrained base model')
    parser.add_argument('--pretrained', type=ast.literal_eval, default=False)
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--Results', default='Results/gen/', help='eval image')
    parser.add_argument("--valDataset", type=str, default="vimeo", choices=["vimoe", "vid4"], help="eval dataset type")
    parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8],
                        help="Super resolution upscale factor")
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
    # data resource configuration
    parser.add_argument('--data_dir', type=str, default='/dataset/vimeo_septuplet/sequences')
    parser.add_argument('--file_list', type=str, default='/dataset/vimeo_septuplet/sep_trainlist.txt')
    parser.add_argument('--cloud_file_list', type=str, default='/src/file/sep_trainlist.txt')
    # distribute
    parser.add_argument("--run_distribute", type=int, default=0, help="run distribute, default: false.")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 0.")
    parser.add_argument("--rank", type=int, default=0, help="rank id, default: 0.")
    # additional parameters
    parser.add_argument('--sens', type=float, default=1024.0)
    # cloud
    parser.add_argument('--cloud', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url',
                        help='path to training/inference dataset folder',
                        default='/home/work/user-job-dir/data/')
    parser.add_argument('--train_url',
                        help='model folder to save/load',
                        default='/home/work/user-job-dir/model/')

    args = parser.parse_args()
    return args
