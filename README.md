ECC - Edge-Conditioned Convolution on Graphs
=========

This is the official PyTorch port of the original Torch implementation of our CVPR'17 paper *Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs* <https://arxiv.org/abs/1704.02901> for the task of graph classification. 

<img src=http://imagine.enpc.fr/~simonovm/ecc_net.gif width="75%" height="75%">


## Code structure 

* `./ecc/*` - General-purpose graph convolution and pooling modules (graphs represented in [igraph](http://igraph.org) framework).
* `./pointcloud_*` - Point cloud dataset loading and conversion of point clouds to graphs.
* `./*` - Model definition and the main script


## Requirements

1. Install [PyTorch](https://pytorch.org) and then [torchnet](https://github.com/pytorch/tnt) with `pip install git+https://github.com/pytorch/tnt.git@master`.

2. The point cloud classification part of the code relies on the Point cloud library, which is unfortunately a bit troublesome to set up for Python. Install [PCL](http://pointclouds.org) and then its python wrapper with `git clone https://github.com/strawlab/python-pcl.git ; cd python-pcl; pip install cython; python setup.py install`. While simple `sudo apt-get pcl` should work in theory, I had success with installing full-sized PCL 1.8 from sources with qhull support, which I got with `sudo apt-get install libqhull-dev` beforehand, and then manually took care of an file naming [issue](https://github.com/strawlab/python-pcl/issues/97) in PCL.

3. Install additional Python packages: `pip install future python-igraph tqdm transforms3d pynvrtc cupy`.

The code was tested on Ubuntu 14.04 with Python 2.7 or 3.6.



## Point cloud classification

### Sydney Urban Objects

The dataset can be downloaded from [http://www.acfr.usyd.edu.au/papers/data/sydney-urban-objects-dataset.tar.gz] and extracted into `./datasets` (or `SYDNEY_PATH` has to be changed in `pointcloud_dataset.py` otherwise). To train the model described in the paper, run

```
for FOLD in 0 1 2 3; do \
CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.1 --lr_steps '[200,245]' --epochs 250  --batch_size 32 \
--model_config 'i_0.1_0.2, c_16,b,r, c_32,b,r, m_0.25_0.5, c_32,b,r, c_32,b,r, m_0.75_1.5, c_64,b,r, m_1.5_1.5,a_1e10_1e10, f_64,b,r,d_0.2,f_14' \
--fnet_widths '[16,32]' --fnet_llbias 0 --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 \
--nworkers 3 --edgecompaction 0 --cvfold $FOLD --odir "results/sydney_${FOLD}"; \
done
```

The average of the final F1 scores (found in respective JSON log files `trainlog.txt` in the last `f1_test` field) should be around 0.78, subject to GPU non-determinism.

### ModelNet

[ModelNet](http://modelnet.cs.princeton.edu/) is a dataset of meshes, which for our purpose had to be converted to synthetic point clouds by uniformly sampling 1000 points on mesh faces according to face area (code was based on PCL [example](https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp), not in this repository). We provide the resulting point clouds for download [here](http://imagine.enpc.fr/~simonovm/ModelNetsPcd.tar.gz). The archive can be extracted to `./datasets` (or `MODELNET10_PATH` and `MODELNET40_PATH` have to be changed in `pointcloud_dataset.py` otherwise).

To train the ModelNet10 model described in the paper, run:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset modelnet10 --test_nth_epoch 25 --lr 0.1 --lr_steps '[50,100,150]' --epochs 175 --batch_size 64 --batch_parts 4 \
--model_config 'i_1_2, c_16,b,r, c_32,b,r, m_2.5_7.5, c_32,b,r, c_32,b,r, m_7.5_22.5, c_64,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_10' \
--fnet_llbias 0 --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 \
--nworkers 3 --edgecompaction 1 --odir results/modelnet10
```

To train the ModelNet40 model described in the paper, run:

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset modelnet40 --test_nth_epoch 25 --lr 0.1 --lr_steps '[30,60,90]' --epochs 100 --batch_size 64 --batch_parts 4 \
--model_config 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40'  \
--fnet_llbias 0 --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 \
--nworkers 3 --edgecompaction 1 --edge_mem_limit 1000 --odir results/modelnet40  
```

Test set mean class accuracies should be around 89 and 82 respectively ('acc_test' is mean instance accuracy, 'cacc_test' is mean class accuracy in `trainlog.txt`), subject to GPU non-determinism.

In order to evaluate the trained model with test-time voting, run:

```
CUDA_VISIBLE_DEVICES=2 python main.py \
--dataset modelnet10 --epochs -1 --test_sample_avging vote --nworkers 3 --edgecompaction 1 \
--resume results/modelnet10/model.pth.tar --odir results/modelnet10_vote
```

### Custom datasets

Adding a custom dataset should be relatively easy by adaptating functions `get_modelnet_info()` and `get_modelnet()` in `pointcloud_dataset.py` and registering them in `main.py`. The model has to be configured properly in the argument `--model_config` to use sensible grid and neighborhood sizes for graph pyramid construction (layers `i`, `m`, and `a`, see `models.py`).



## General graph classification

TBD



## Issues

* The model may be quite memory hungry for point cloud training especially, where nearly all edges have their unique edge features and thus also filter weights. This can be currently  circumvented in three ways:
	* Compacting edges with identical attributes by setting `--edgecompaction 1`
	* Computing convolutions and poolings in shards of #E edges by `--edge_mem_limit #E`
	* Evaluating the batch in multiple runs by setting `--batch_parts` to a divisor of `--batch_size`.
A yet unimplemented alternative would be to implement (randomized) clustering in `cloud_edge_feats()`.


## Citation

```
@inproceedings{Simonovsky2017ecc,
    author = {Martin Simonovsky and Nikos Komodakis},
    title = {Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs},
    url = {https://arxiv.org/abs/1704.02901},
    booktitle = {CVPR},
    year = {2017}}
```