## Implement [Pix2Pix](https://phillipi.github.io/pix2pix/) with [Fully Convolutional Layers](https://arxiv.org/abs/1411.4038)

### Import the Datasets

- Method 1

```bash
bash download_facades_dataset.sh
```

- Method 2

Download and unzip the [dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz) to `/02_DIPwithPyTorch/Pix2Pix/datasets/facedes/`,then run

```cmd
python generate_file_list.py
```

### Run

```cmd
python train.py
```