# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    postprocess
"""
from os import path as osp
import argparse
import ast
import numpy as np
from mindspore import context
from src.datasets.dataset import RBPNDatasetTest, create_val_dataset
from src.util.utils import save_img, PSNR


parser = argparse.ArgumentParser('Postprocess')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--val_path", type=str, default=r'/dataset/Vid4')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8],
                    help="Super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--save_eval_path', type=str, default="./Results/eval", help='save eval image path')
parser.add_argument('--file_list', type=str, default='foliage.txt')
parser.add_argument('--other_dataset', type=ast.literal_eval, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=ast.literal_eval, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--residual', type=ast.literal_eval, default=False)
args = parser.parse_args()

def predict(prediction, target):
    """predict
    Args:
        ds(Dataset): eval dataset
        model(Cell): the generate model
    """


    prediction = prediction[0]
    prediction = prediction * 255.

    target = target.squeeze().asnumpy().astype(np.float32)
    target = target * 255.

    psnr_predicted = PSNR(prediction, target, shave_border=args.upscale_factor)
    return psnr_predicted


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, device_target="CPU")
    rst_path = "./result_Files"
    val_dataset = RBPNDatasetTest(args.val_path, args.nFrames, args.upscale_factor, args.file_list, args.other_dataset,
                                  args.future_frame)
    val_ds = create_val_dataset(val_dataset, args)

    save_pre_path = osp.join('./310_infer_image', 'prediction')
    save_gt_path = osp.join('./310_infer_image', 'gt')

    psnr_sum = 0
    for i, data in enumerate(val_ds.create_dict_iterator(), 1):
        tar = data['target_image']
        file_name = osp.join(rst_path, "RBPN_data_x" + str(args.testBatchSize) + '_' + str(i) + '_0.bin')
        pred = np.fromfile(file_name, np.float32).reshape(1, 3, 480, 720)
        save_img(pred, str(i), save_pre_path)
        save_img(tar, str(i), save_gt_path)

        psnr = predict(pred, tar)
        print("===> Processing: compute_psnr", psnr)
        psnr_sum += psnr
        psnr_mean = psnr_sum/i

    print("val ending psnr = ", psnr_mean)
    print("Generate images success!")
