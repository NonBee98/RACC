## Dataset
PolyU RGB+ALS Color Constancy Dataset can be downloaded from https://1drv.ms/u/c/73ea3202965da3e5/ESZRlNV74jhKr0kDQn61FOgBVMiJ8NK2TMyjLwnTLY454A?e=daP3Il

## Usage

### Prepraration
Python version >=3.9 is required.

#### Install dependencies
```bash
pip install --upgrade -r requirements.txt
```
### Running testing script

```bash
python test.py \
  --channels <int> \
  --test_dir <path> \
  --out_dir <path>
```
Options:
- `--channels` (optional): number of used spectrum channels, should be same as training.
- `--test_dir` (optional): dataset directory for testing, default is `input` folder. Test images should be in `PNG` format
- `--out_dir` (optional): where to save the output images, default is `output` folder.
- `--ckpt` (optional): path to the model weight file for testing, default is `checkpoints/best.ckpt`.
- `--post_process` (optional): whether to apply post-processing to the output white points (e.g. avoiding extreme values). This option requires calibrated Planckian locus data, default is `Flase`.

For example:
```bash
python test.py --test_dir input --out_dir output
```
**Note**: The input ALS data should be save in a `txt` file, and the corresponding images should be in `dng` format, sample test data is provided in `input` folder. The output white points and the corresponding white balanced images will be saved in `output` folder by default. Since the test images are in dng format, **ExifTool** is required to be added into PATH or put in the same directory with the `test.py` script (for Windows user, the ExifTool executable file is already included). If you are using Linux or macOS, please download the latest version of ExifTool from [here](https://exiftool.org).

### Runing training script

```bash
python train.py \
  --data_dir <path> \
  --val_dir <path> \
  --channels <int> \
  --epochs <int> \
  --batch_size <int> \
  --lr <float> \
  --model_name <str>
```
Options:
- `--data_dir` (required): dataset directory for training
- `--val_dir` (required): dataset directory for validation
- `--channels` (required): number of used spectrum channels, should be same as testing.
- `--epochs` (optional): number of epochs to train, default is 300.
- `--batch_size` (optional): batch size for training, default is 32.
- `--lr` (optional): learning rate for training, default is 0.002.
- `--resume` (optional): whether to resume training from a previous checkpoint, default is False.

Other more options can be found in `args.py`.

For example:
```bash
python train.py --data_dir mi_o1 --val_dir mi_o1 --epochs 500 --batch_size 32 --lr 0.002 --channels 13
```

**Note**: We have provided a sample dataset in `mi_o1` folder, where 80% of the images are used for training and 20% for validation. If you want to train your own model, please prepare your own dataset in the same format.

### Runing evaluation script
```bash
python evaluate.py \
  --val_dir <path> \
  --channels <int> \
  --ckpt <path>
```
Options:
- `--val_dir` (required): dataset directory for validation
- `--channels` (optional): number of used spectrum channels, should be same as training.
- `--ckpt` (optional): path to the model weight file for validation, default is `checkpoints/best.ckpt`.

The evaluation results will be saved in ``evaluation_results`` folder.

For example:
```bash
python evaluate.py --val_dir mi_o1 --ckpt checkpoints/best.ckpt
```
This will evaluate the model on the `mi_o1` dataset using the checkpoint file "checkpoints/best.ckpt".

