# Contents

- [RBPN Description](#rbpn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script  Parameters](#script-and-sample-code)
        - [Preparation Before Training](#preparation-before-training)
        - [Training Script Parameters](#training-script-parameters)
        - [Training Result](#training-result)
        - [Preparation Before Evaluation](#preparation-before-evaluation)
        - [Evaluation Script Parameters](#evaluation-script-parameters)
        - [Evaluation Result](#evaluation-result)
    - [inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Export MindIR](#infer-on-Ascend310)
        - [Export Result](#export-result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [RBPN Description](#contents)

We proposed a novel architecture for the problem of video super-resolution. We integrate spatial and temporal contexts from continuous video frames using a recurrent encoder-decoder module, that fuses multi-frame information with the more traditional, single frame super-resolution path for the target frame. In contrast to most prior work where frames are pooled together by stacking or warping, our model, the Recurrent Back-Projection Network (RBPN) treats each context frame as a separate source of information. These sources are combined in an iterative refinement framework inspired by the idea of back-projection in multiple-image super-resolution. This is aided by explicitly representing estimated inter-frame motion with respect to the target, rather than explicitly aligning frames. We propose a new video super-resolution benchmark, allowing evaluation at a larger scale and considering videos in different motion regimes. Experimental results demonstrate that our RBPN is superior to existing methods on several datasets.

[Paper](https://arxiv.org/abs/1903.10128): Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita

# [Model Architecture](#contents)

The operation of RBPN can be divided into three stages: initial feature extraction, multiple projections,
and reconstruction. All of these steps mainly use convolution and deconvolution.

# [Dataset](#contents)

Train RBPN Dataset used : [Vimeo90k](<http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip>) , The original training + test set (82GB)

- Note: Data will be processed in src/dataset/dataset.py

```markdown
????????? vimeo_septuplet
  ?????? sequnences
  ?????? sep_trainlist.txt
  ?????? sep_testlist.txt
  ?????? fast_testlist.txt
  ?????? medium_testlist.txt
```

Validation and eval evaluationdataset
used: [Vimeo90k](<http://toflow.csail.mit.edu/) [Vid4](https://drive.google.com/drive/folders/1sI41DH5TUNBKkxRJ-_w5rUf90rN97UFn?usp=sharing) [SPMCS](https://drive.google.com/drive/folders/1sI41DH5TUNBKkxRJ-_w5rUf90rN97UFn?usp=sharing)

```markdown
????????? Vid4
  ?????? calendar
     ?????? 001.png
     ?????? 002.png
     ?????? 003.png
     ...
  ?????? city
  ?????? foliage
  ?????? walk
  ?????? carlendar.txt
  ?????? city.txt
  ?????? foliage.txt
  ?????? walk.txt
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below???
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```markdown
.RBPN
?????? README.md                       # descriptions about RBPN
????????? scripts
  ?????? run_distribute_train.sh       # launch ascend training(8 pcs)
  ?????? run_stranalone_train.sh       # launch ascend training(1 pcs)
  ?????? run_eval.sh                   # launch ascend eval
?????? src
  ?????? dataset
    ?????? dataset.py                  # dataset for training and evaling
  ?????? loss
    ?????? generatorloss.py            # compute_psnr losses function define
  ?????? model
    ?????? base_network.py             # the basic unit for build neural network
    ?????? dbpns.py                    # generator DBPNS define
    ?????? dbpns.py                    # generator RBPN define
  ?????? util
    ?????? config.py                   # parameters using for train and eval
    ?????? utils.py                    # initialization for srgan
?????? train.py                        # train rbpn  script
?????? eval.py                         # eval
?????? trainonestepgen.py              # training process for generator RBPN which add clip grad
?????? export.py                       # export mindir script
```

## [Script Parameters](#contents)

### [Preparation Before Training](#contents)

Need to download pyflow(https://github.com/alterzero/pyflow) before running, Then generate the relevant pyflow files.

```text
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
python setup.py build_ext -i
python demo.py    # -viz option to visualize output
```

After generating the pyflow file, we can see the .pyx or .so file, put it in the same directory as train.py, and then it can run normally.

### [Training Script Parameters](#contents)

```shell
# distributed training RBPN models
Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [DATA_DIR] [FILE_LIST] [BATCHSIZE]

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json /vimeo_septuplet/sequences /vimeo_septuplet/sep_trainlist.txt 4

# standalone training RBPN models
Usage: bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR] [FILE_LIST] [BATCHSIZE]

eg: bash run_standalone_train.sh 0   /vimeo_septuplet/sequences /vimeo_septuplet/sep_trainlist.txt 4

#python train standalone
Usage: python train.py [DEVICE_ID] [DATA_DIR] [FILE_LIST] [BATCHSIZE]

eg: python train.py --device_id=0 --data_dir=/vimeo_septuplet/sequences --file_list=/vimeo_septuplet/sep_trainlist.txt --batchSize=4

#tips
if you want to use pretrained ckpt , please modify /src/util/config.py as the follows:
parser.add_argument('--pretrained_sr', default='pretrained_rbpn.ckpt', help='sr pretrained base model')
parser.add_argument('--pretrained', type=ast.literal_eval, default=True)
```

### [Training Result](#content)

Training result of the model loss will be stored in result.
You can find checkpoint file in weights.

```bash
If you use rundistribute you can find the log in scripts/train_rbpn_parallel0/paralletrain.log
To view the loss or avg_loss you can use the following code:
eg: cat paralletrain.log |grep loss , cat paralletrain.log |grep Avg
```

### [Preparation Before Evaluation](#contents)

Please put the pyflow file in the same directory as eval.py.

Please check the eval.py ,if the hyperparameters are set as follows:

```bash
parser.add_argument('--future_frame', type=ast.literal_eval, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
```

Then, According to the description of the author's paper, when the above two hyperparameters are used, the 6Frame-PF mode is used, that is, the neighbor used is the past 3 frames and the next 3 frames. When the current frame has no past or future frames, the neighbor uses the current frame as the Instead, in the '.txt' file of the test set directory provided by the author, the frames all have the first frame and the last frame, and they do not have adjacent frames, so the author recommends deleting the first frame  and the last two frames in each test set.

Take the 'city.txt' as an example, delete city/001.PNG, city/033.PNG and city/034.PNG in the 'city.txt' file.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` for evaluation.

```bash
#evaling RBPN
Usage: bash run_eval.sh [DEVICE_ID] [CKPT] [VAL_PATH] [FILE_LIST]

eg: bash run_eval.sh 0 /weight/rbpn.ckpt  /dataset/Vid4  /dataset/Vid4/calendar3.txt

#evaling RBPN python
Usage: python eval.py [DEVICE_ID] [CKPT] [VAL_PATH] [FILE_LIST]

eg: nohup python eval.py --device_id=0 --ckpt=rbpn.ckpt --val_path=/dataset/Vid4
--file_list=/dataset/Vid4/calendar3.txt > eval.log 2>&1 &
```

### [Evaluation Result](#content)

Evaluation result will be stored in the Result. Under this, you can find generator pictures.

The log file is eval.log which displays the average precision.

## [Inference Process](#contents)

### [Export MindIR](#contents)

Since the image sizes of the test set can vary, and the input required by mindir must be fixed, you need to check the image size parameters in export.py before exporting mindir. The length and width should be one quarter of the original image size.

```text
# check the length and width
# the test dataset vid4/foliage (vim export.py , Correct the following)
eg: input_array = Tensor(np.zeros([1, 3, 120, 180], np.float32))
    neighbor_array = Tensor(np.zeros([1, 6, 3, 120, 180], np.float32))
    flow_array = Tensor(np.zeros([1, 6, 2, 120, 180], np.float32))
# the test dataset vid4/city (vim export.py , Correct the following)
eg: input_array = Tensor(np.zeros([1, 3, 144, 176], np.float32))
    neighbor_array = Tensor(np.zeros([1, 6, 3, 144, 176], np.float32))
    flow_array = Tensor(np.zeros([1, 6, 2, 144, 176], np.float32))
# export RBPN
python export.py  [CKPT] [FILE_NAME] [FILE_FORMAT]

eg: python export.py --ckpt_path=ckpt/RBPN_best.ckpt --file_name=rbpn --file_format=MINDIR
```

The pretrained parameter is required. EXPORT_FORMAT should be in ["AIR", "MINDIR"] Current batch_size can only be set to 1.

### [Infer on Ascend310](#contents)

**Before inference, please refer to [Environment Variable Setting Guide](https://gitee.com/mindspore/models/tree/master/utils/ascend310_env_set/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py`. Current batch_Size can only be set to 1.

```text
# Ascend310 inference about RBPN network
bash run_infer_310.sh [MINDIR_PATH] [VAL_PATH] [FILE_LIST] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]

eg: bash run_infer_310.sh /home/.../rbpn_best_model.mindir  /home/.../dataset/Vid4  foliage.txt  y  CPU  0
```

The [MINDIR_PATH] and [VAL_PATH] hyperarguments above must be absolute paths and the [FILE_LIST] hyperarguments above only need to be a file name like foliage.txt, but the file must be in [VAL_PATH]. For example, you can move the file foliage.txt to the val_path /home/.../dataset/Vid4.

### [Result](#contents)

Inference result is saved in the path ascend310_infer/, you can find result like this in acc.log file.

If you set --future_frame=True , then it is recommended to delete the first three items and the last three items in the file_list, which will make the accuracy more accurate

```text
'avg psnr': 24.75
```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                |                                                              |
| ------------------------- | ------------------------------------------------------------ |
| Model Version             | RBPN                                                         |
| Resource                  | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G   |
| MindSpore Version         | 1.5.1                                                        |
| Dataset                   | Vimeo90k                                                     |
| Training Parameters       | epoch=150, batch_size = 4bs*8p;                              |
| Optimizer                 | Adam                                                         |
| Loss Function             | L1loss                                                       |
| outputs                   | super-resolution pictures                                    |
| Accuracy(L1loss)          | avg_loss = 0.0094                                            |
| Speed                     | 1pc(Ascend): 277 ms/step                                     |
| Total time                | 8pcs: 81h                                                    |
| Checkpoint of RBPN models | 49M (.ckpt file)                                             |
| Scripts                   | [RBPN script](https://gitee.com/mindspore/models/tree/master/research/cv/RBPN) |

### [Evaluation Performance](#contents)

| Parameters          |                                                              |
| ------------------- | ------------------------------------------------------------ |
| Model Version       | RBPN                                                         |
| Resource            | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G   |
| MindSpore Version   | 1.5.1                                                        |
| Dataset             | Vimeo90k , Vid4 , SPMCS                                      |
| Accuracy(psnr/ssim) | Vid4: calendar(22.27/0.78); city(26.18/0.78); foliage(24.77/0.74); walk(29.25/0.90) ; |
|                     | SPMCS: car05_001(30.50/0.89);  hdclub_003_001(20.60/0.72);  hitachi_isee5_001(24.87/0.89) |
|                     | hk004_001(31.38/0.88);  HKVTG_004(28.12/0.78);  jvc_009_001(28.63/0.90);  NYVTG_006(31.63/0.91) |
|                     | PRVTG_012(26.12/0.81);  RMVTG_011(26.29/0.80);  veni3_011(33.56/0.99);  veni5_015(30.88/0.94) |
|                     | Vimeo90k: fast_testset(38.35/0.95)??? medium_testset(35.70/0.94) |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models/tree/master/)
