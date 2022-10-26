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


import os
import time
import mindspore
import mindspore.nn as nn
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from trainonestepgen import TrainOnestepGen
from src.datasets.dataset import RBPNDataset, create_train_dataset
from src.loss.generatorloss import GeneratorLoss
from src.model.rbpn import Net as RBPN
from src.util.config import get_args
from src.util.utils import save_losses, init_weights

args = get_args()
mindspore.set_seed(args.seed)
epoch_loss = []
eval_mean_psnr = []

save_loss_path = 'results/genloss/'
if not os.path.exists(save_loss_path):
    os.makedirs(save_loss_path)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

def train(trainoneStep, trainds):
    """train the generator
        Args:
            trainoneStep(Cell): the network of
            trainds(dataset): train datasets
        """
    trainoneStep.set_train()
    steps = trainds.get_dataset_size()
    train_loader = trainds.create_dict_iterator()
    best_avgpsnr = 10
    for epoch in range(args.start_iter, args.nEpochs + 1):
        e_loss = 0
        t0 = time.time()
        for iteration, batch in enumerate(train_loader, 1):
            x = batch['input_image']
            target = batch['target_image']
            neighbor_tensor = batch['neighbor_image']
            flow_tensor = batch['flow_image']
            loss = trainoneStep(target, x, neighbor_tensor, flow_tensor)
            e_loss += loss.asnumpy()
            print('Epoch[{}]({}/{}): loss: {:.4f}'.format(epoch, iteration, steps, loss.asnumpy()))

        t1 = time.time()
        mean = e_loss / steps
        epoch_loss.append(mean)
        print("Epoch {} Complete: Avg. Loss: {:.4f}|| Time: {} min {}s.".format(epoch, mean, int((t1 - t0) / 60),
                                                                                int(int(t1 - t0) % 60)))
        print('===> Saving model')
        if args.cloud:
            if (epoch) % (args.snapshots) == 0:
                print('train_dir:', train_dir)
                save_checkpoint_path = train_dir + '/device_' + os.getenv('DEVICE_ID') + '/'
                if not os.path.exists(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)

                model_name = 'rbpn_epoch%d.ckpt' % (epoch)
                #ckpt_dir_path = os.path.join(train_dir, f'rbpn_epoch{epoch}.ckpt')

                ckpt_dir_path = os.path.join(save_checkpoint_path, model_name)
                print("ckpt position:", ckpt_dir_path)
                save_checkpoint(trainoneStep.network, ckpt_dir_path)

                mox.file.copy_parallel(train_dir, obs_train_url)
                print("Successfully Upload {} to {}".format(train_dir, obs_train_url))



        else:
            if  args.run_distribute and args.device_id == 0:
                if  mean <= best_avgpsnr:
                    save_best_ckpt = os.path.join(args.save_folder, 'the_minest_loss.ckpt')
                    save_checkpoint(trainoneStep.network, save_best_ckpt)
                    best_avgpsnr = mean
                if (epoch) % (args.snapshots) == 0:
                    save_ckpt = os.path.join(args.save_folder, '{}_{}.ckpt'.format(epoch, args.model_type))
                    save_checkpoint(trainoneStep.network, save_ckpt)
                    name = os.path.join(save_loss_path, args.valDataset + '_' + args.model_type)
                    save_losses(epoch_loss, None, name)
            elif  args.run_distribute == 0:
                if  mean <= best_avgpsnr:
                    save_best_ckpt = os.path.join(args.save_folder, 'the_minest_loss.ckpt')
                    save_checkpoint(trainoneStep.network, save_best_ckpt)
                    best_avgpsnr = mean
                if (epoch) % (args.snapshots) == 0:
                    save_ckpt = os.path.join(args.save_folder, '{}_{}.ckpt'.format(epoch, args.model_type))
                    save_checkpoint(trainoneStep.network, save_ckpt)
                    name = os.path.join(save_loss_path, args.valDataset + '_' + args.model_type)
                    save_losses(epoch_loss, None, name)

        print("finish saving model")

if __name__ == '__main__':
    # distribute
    # parallel environment setting
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.run_distribute:
        print("distribute")
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = args.device_num
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank = get_rank()
    else:
        device_id = args.device_id
        context.set_context(device_id=device_id)

    zip_out_dir = args.data_dir
    file_list = args.file_list
    # cloud setting
    if args.cloud:
        import moxing as mox
        home = os.path.dirname(os.path.realpath(__file__))
        file_list = home + args.cloud_file_list
        data_dir = os.path.join(home, 'data')  # saving data path
        train_dir = os.path.join(home, 'checkpoints')  # saving model path
        # Initialize the data storage directory
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # Initialize the model storage directory
        obs_train_url = args.train_url
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        obs_data_url = args.data_url
        # Copy the data to the training environment

        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      data_dir))

        path = home
        datanames = os.listdir(path)
        zip_out_dir = home + '/data/vimeo1/sequences'

    # get dataset
    print("Preparing Data")
    start_time = time.perf_counter()
    train_dataset = RBPNDataset(zip_out_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
                                file_list, args.other_dataset, args.patch_size, args.future_frame)
    train_ds = create_train_dataset(train_dataset, args)
    train_steps = train_ds.get_dataset_size()
    end_time = time.perf_counter()
    print("preparing data use: {}min".format((end_time - start_time) / 60))

    print('===>Building model ', args.model_type)
    model = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=args.nFrames,
                 scale_factor=args.upscale_factor)
    init_weights(model, 'KaimingNormal', 0.02)
    print('====>start training')

    if args.pretrained:
        ckpt = os.path.join(args.save_folder, args.pretrained_sr)
        print('=====> load params into generator')
        params = load_checkpoint(ckpt)
        load_param_into_net(model, params)
        print('=====> finish load generator')

    lossNetwork = GeneratorLoss(model)

    milestone = [int(args.nEpochs / 2) * train_steps, args.nEpochs * train_steps]
    learning_rates = [args.lr, args.lr / 10.0]
    lr = piecewise_constant_lr(milestone, learning_rates)

    optimizer = nn.Adam(model.trainable_params(), lr, loss_scale=args.sens)

    trainonestepNet = TrainOnestepGen(lossNetwork, optimizer, sens=args.sens)


    train(trainonestepNet, train_ds)
    print("finish training")

    if args.cloud:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,
                                                    obs_train_url))
