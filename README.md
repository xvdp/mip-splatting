<p align="center">
  <h1 align="center">Mip-Splatting: Alias-free 3D Gaussian Splatting</h1>

## low_memory branch: xvdp
Mip Splatting (as Gaussian Splatting original) codes load all images in gpu memory allowing only to train with limited number and size of images.  

This branch -- similarly to https://github.com/xvdp/gaussian-splatting implements 2 alternative ways of loading images. Via torch dataloader and with a memory map h5 file. This ois useful on large image datasets(e.g. SonyAR7V shoots ~ 61Mpix images), and/or with large number of images.

mip-splatting trining iteration average for the first 1000 iterations on the tested system (Razer Laptop w/ 16GB RTX3080) is approx 50ms/it for 1Mpix files increasing quadratically, 350ms/it for 3.8Mpix files 3.4s/it for 14Mpix files. It is not clear that data from larger images are taken advandage for training.

### h5 memory mapped dataset 
***load_images_mode=2***
This branch defaults to creating a .h5 file and loading slices from it.

* Can train with unlimited number of larger images. 
* Requires disk space -e.g. A pyramid of 2000 8bit images with highest dimension ~(9600, 6400) requires 480 GB of disk.
* Nearly matches speed to preloading on GPUs on small images. Slicing from h5 is: ~5ms/Mpix,increasing linearly with size.

With a multiprocess and queue this cut be cut further.

### torch Dataset/Dataloader 
***load_images_mode=0***

* Can train with unlimited number of larger images.
* Nearly matches speed to preloading on GPUs on small images with 10 workers. Single process PIL requires ~35ms/Mpix increasing linearly with size. Multiprocessing may be bottlenecked at some image size but as training loop time increases quadratically with image size, above 10Mpix loading from disk is again usable.

### load into GPU: original default
***load_images_mode=1*** orignal code, store to memory.

------------


  <p align="center">
    <a href="https://niujinshuchong.github.io/">Zehao Yu</a>
    ·
    <a href="https://apchenstu.github.io/">Anpei Chen</a>
    ·
    <a href="https://github.com/hbb1">Binbin Huang</a>
    ·
    <a href="https://tsattler.github.io/">Torsten Sattler</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>

  </p>
  <h3 align="center"><a href="https://drive.google.com/file/d/1Q7KgGbynzcIEyFJV1I17HgrYz6xrOwRJ/view?usp=sharing">Paper</a> | <a href="https://arxiv.org/pdf/2311.16493.pdf">arXiv</a> | <a href="https://niujinshuchong.github.io/mip-splatting/">Project Page</a>  | <a href="https://niujinshuchong.github.io/mip-splatting-demo/">Online Viewer</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/bicycle_3dgs_vs_ours.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce a 3D smoothing filter and a 2D Mip filter for 3D Gaussian Splatting (3DGS), eliminating multiple artifacts and achieving alias-free renderings.  
</p>
<br>


# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:autonomousvision/mip-splatting.git
cd mip-splatting

conda create -y -n mip-splatting python=3.8
conda activate mip-splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

# Dataset
## Blender Dataset
Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then generate multi-scale blender dataset with
```
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale
```

## Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill and flowers scenes.

# Training and Evaluation
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Online viewer
After training, you can fuse the 3D smoothing filter to the Gaussian parameters with
```
python create_fused_ply.py -m {model_dir}/{scene} --output_ply fused/{scene}_fused.ply"
```
Then use our [online viewer](https://niujinshuchong.github.io/mip-splatting-demo) to visualize the trained model.

# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 

# Citation
If you find our code or paper useful, please cite
```bibtex
@article{Yu2023MipSplatting,
  author    = {Yu, Zehao and Chen, Anpei and Huang, Binbin and Sattler, Torsten and Geiger, Andreas},
  title     = {Mip-Splatting: Alias-free 3D Gaussian Splatting},
  journal   = {arXiv:2311.16493},
  year      = {2023},
}
```
