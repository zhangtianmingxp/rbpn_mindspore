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
"""preprocess"""
import os
import argparse
import ast
from src.datasets.dataset import RBPNDatasetTest, create_val_dataset

parser = argparse.ArgumentParser(description="Preporcess")
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
print(args)

if __name__ == '__main__':
    result_path = "./preprocess_Result/"

    val_dataset = RBPNDatasetTest(args.val_path, args.nFrames, args.upscale_factor, args.file_list, args.other_dataset,
                                  args.future_frame)
    val_ds = create_val_dataset(val_dataset, args)

    x_path = os.path.join(result_path, "x_data")
    if not os.path.exists(x_path):
        os.makedirs(x_path)

    neighbor_path = os.path.join(result_path, "neighbor_data")
    if not os.path.exists(neighbor_path):
        os.makedirs(neighbor_path)

    flow_path = os.path.join(result_path, "flow_data")
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)


    for i, data in enumerate(val_ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        j = i+1
        file_x_name = "RBPN_data_x" + str(args.testBatchSize) + "_" + str(j) + ".bin"
        file_neighbor_name = "RBPN_data_neighbor" + str(args.testBatchSize) + "_" + str(j) + ".bin"
        file_flow_name = "RBPN_data_flow" + str(args.testBatchSize) + "_" + str(j) + ".bin"

        file_x_path = x_path + "/" + file_x_name
        file_neighbor_path = neighbor_path + "/" + file_neighbor_name
        file_flow_path = flow_path + "/" + file_flow_name

        x = data['input_image']
        neighbor = data['neighbor_image']
        flow = data['flow_image']

        x.tofile(file_x_path)
        neighbor.tofile(file_neighbor_path)
        flow.tofile(file_flow_path)

    print("=" * 20, "export bin files finished", "=" * 20)
