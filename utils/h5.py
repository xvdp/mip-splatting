""" @xvdp
data handling with h5
When image sizes are large or many, loading to gpu memory is not feasible.
When image sizes are large, loading images from disk even multiprocessed will be slow

This file has a few methods to load images from h5, precomputing an image pyramid
with default downscaling of (2,4,8)

functions:
    H5  wrapper to h5py.File which acts as a passtrhu when None is passed
    with H5(h5filename) as fi:
        out = fi.get_image(index)
    with H5(None) as fi:
        ... # 

    get_images_db() # creates h5 file with downscaled dbs
    h5_imagesdb_add() # adds images to h5 db
    
"""
from typing import Union, Optional, Tuple, Any
import time
import os
import os.path as osp
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from enum import Enum

import numpy as np
import torch
from PIL import Image
import h5py


# pylint: disable=no-member
class LoadMode(Enum):
    """ Lower memory options
    access: LoadMode(1), LoadMode['ON_MEMORY'], LoadMode.ON_MEMORY
        .name -> ON_MEMORY
        .value -> 1
    """
    FROM_DISK = 0
    ON_MEMORY = 1
    ON_H5 = 2

class H5:
    """ context manager wrapper for h5py.File
    if no file is passed to filename, acts as a pass-through
    with H5("myfile.h5") as hf:
        torchimg = hf.get_image(index=0, dbname='images')
        name = hf.get_text(index=0, dbname='names')
    """
    def __init__(self, filename: Optional[str] = None) -> None:
        self.filename = filename
        self.file = None

    def __enter__(self) -> Any:
        if self.filename is not None:
            self.file = h5py.File(self.filename, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def get_image(self, index=None, dbname='images', device='cuda', dtype=torch.float32):
        """ serve single image"""
        if self.file is not None:
            if index is None:
                index = torch.randint(len(self.file[dbname]))
            image = np.array(self.file[dbname][index])
            image = (torch.as_tensor(image).permute(2,0,1).contiguous().to(device=device,
                                                                          dtype=dtype))/255.
            image[:3] = torch.clamp(image[:3], min=0, max=1.0)
            if image.shape[0] == 4:
                image[:3]  *= image[3:4]
            return image
        return None

    def get_text(self,
                 dbname='names',
                 index: Union[None, int, tuple, list, np.ndarray] = None) -> Union[list, str, None]:
        """
        Args
            dbname  (str) dataset name - must be a str database, not checked
            index   (int, list, ndarray, tuple, None)
                int             returns string at index
                tuple, None     returns slice of strings, all if None
                list, ndarray   returns indexed 
        """
        if self.file is not None:
            assert dbname in self.file, f"'{dbname}' not in {self.file.keys()}"
            assert isinstance(index, (type(None), int, tuple,
                                      list, np.ndarray)), f"invalid index got {type(index)}"
            if isinstance(index, int):
                return self.file[dbname][index].decode()

            if isinstance(index, tuple) or index is None:
                index = range(len(self.file[dbname])) if index is None else range(*index)

            return [self.file[dbname][i].decode() for i in index]

        return None


def get_images(images_path: str, convert_to_h5: bool = False) -> str:
    """ converts to h5 resolution stack if images.h5 does not exists
    """
    images_path = osp.abspath(osp.expanduser(images_path))
    assert osp.exists(images_path), f"path <{images_path}> not found"

    # if h5 is found, return hf
    if is_h5file(images_path):
        print(f"found image file {images_path}")
        return images_path # -> .h5

    if osp.isdir(images_path):
        fname = f"{images_path}.h5"
        if osp.isfile(fname):
            print(f"found image file {fname}")
            return fname # -> .h5

    if convert_to_h5:
        fname = f"{images_path}.h5"
        print(f"converting images to {fname}")
        h5_images(images_path, fname)
        return fname # -> .h5

    print(f"images found in folder {fname}")
    return images_path # -> folder


def get_images_db(path: str,
                  ext: Union[str, tuple] = ('.jpg', '.png'),
                  **kwargs) -> Tuple[str, bool]:
    """ creates an h5 db file if it does not exist named <path>.h5
    returns path to .h5 and True if created anew
    Args
        path    (str) image folder or h5 file
        ext     (str or tuple) extensions of image files
        kwargs:
            buffer_size (int [2]): # rdcc_nbytes = buffer_size*image.shape
            down_scales (tuple [2,4,8]) create downscaled versions of database
            channel_dim (int [2 or 0]) 2: save as numpy (h, w, c), 0 save as torch (c, h, w)
            count       (int) number of images if fewer than on path
    """
    path = osp.abspath(osp.expanduser(path))
    assert osp.exists(path), f"path <{path}> not found"
    created = False

    # if h5 is found, return hf
    if is_h5file(path):
        print(f"found image file {path}")
        fname = path

    elif osp.isdir(path):
        fname = f"{path}.h5"
        if osp.isfile(fname):
            print(f"found image file {fname}")
        else: # create new h5 file to fill w images
            ext = (ext,) if isinstance(ext, str) else ext
            assert isinstance(ext, (tuple, list)), f"expected tuple of extensions, got {ext}"
            files = [f.path for f in os.scandir(path) if osp.splitext(f.name)[-1].lower() in ext]
            kw = {k:v for k,v in kwargs.items() if k in
                  ('buffer_size', 'down_scales', 'channel_dim')}
            count = kwargs.get('count', len(files))
            fname = h5_imagesdb_make(files[0], fname, count, **kw)
            created = True

    assert is_h5file(fname), f"failed to find or create a .h5 file in {path}"
    return fname, created

def h5_read_strdb(h5name: str, dbname: str) -> list:
    """ reads and decodes a str db"""
    with h5py.File(h5name, 'r') as hf:
        return [name.decode() for name in hf[dbname]]

def is_h5file(path: str) -> bool:
    return osp.isfile(path) and osp.splitext(path)[-1] == '.h5'

def h5_imagesdb_make(image: str,
                     h5name: str,
                     count: int,
                     buffer_size: int = 2, # rdcc_nbytes = buffer_size*shape
                     down_scales: Optional[tuple] = (2,4,8),
                     channel_dim: int = 2,
                     dbname: str = "images") -> str:
    """ Create h5 dataset
    Args
        image       (str) first image of the dataset
        h5name      (str) non existing fname
        count       (int) number of images
        buffer_size (int [2]): # rdcc_nbytes = buffer_size*shape
        down_scales (tuple [2,4,8]) create downscaled versions of database
        channel_dim (int [2 or 0]) 2: save as numpy, 0 save as torch
        dbname      (str ['images']) default db name
    """
    # validate inputs
    # only create if not existing
    assert not osp.isfile(h5name), f"{h5name} exists... nothing done"

    # set up formats (store down scaled, order (c,h,w) or (h,w,c))
    down_scales = down_scales or []
    shape = _reshape(np.asarray(Image.open(image)).shape, channel_dim=channel_dim)

    with h5py.File(h5name, 'w') as hf:
        # create datasets
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('names', (count,), dtype=dt)

        print(f"dataset: images {shape}")
        hf.create_dataset(dbname, shape=(count, *shape), dtype='uint8',
                          chunks=(1, *shape), rdcc_nbytes=buffer_size*np.prod(shape))
        shapes = {}
        for j in down_scales:
            shapes[j] = _scale_shape(shape, j)
            print(f"dataset: {dbname}_{j} {shapes[j]}")
            hf.create_dataset(f'{dbname}_{j}', shape=(count, *shapes[j]), dtype='uint8',
                              chunks=(1, *shapes[j]), rdcc_nbytes=buffer_size*np.prod(shape))
    return h5name

def h5_imagesdb_add(image: str,
                    h5name: str,
                    index: int,
                    dbname: str = 'images'):
    """ open image and add it to h5, at index, scale to sizes determined by existing dbs, add name
    Args
        image   (str) valid image path
        h5name  (str) existing h5 file with db['images'] containing images same size as input image
        index   (int) desired index where to include image in db
        dbname  (str ['images']) default db name
    """
    with h5py.File(h5name, 'a') as hf:
        assert dbname in hf, f"expected dataset called 'images', got {hf.keys()}"
        num = len(hf[dbname])
        assert(index%num == index or index%num == index + num), f"max index allowed: {num-1}"

        # store name
        hf['names'][index] = osp.basename(image)
        _img = Image.open(image)
        nimg = np.asarray(_img)
        # ensure that image size matches stored image
        if nimg.shape[-1] == hf[dbname].shape[1]:
            channel_dim = 0
            nimg = nimg.transpose(2,0,1)
        else:
            channel_dim = 2
        assert nimg.shape == hf[dbname].shape[1:], f" expects all images same size \
            {hf[dbname].shape[1:]}, got image {image} withs shape {nimg.shape}"
        # store highest level of image
        hf[dbname][index] = nimg

        # down scale images to database sizes
        shapes = {}
        for k,v in hf.items():
            if f'{dbname}_' in k:
                _shape = list(v.shape[1:])
                _shape.pop(channel_dim)
                shapes[k]= _shape

        for db, size in shapes.items():
            nimg = np.asarray(_img.resize(size[::-1]))
            if channel_dim == 0:
                nimg = nimg.transpose(2,0,1)
            hf[db][index] = nimg


def h5_images(images: Union[str, tuple],
              h5name: str,
              buffer_size: int = 2, # rdcc_nbytes = buffer_size*shape
              down_scales: Optional[tuple] = (2,4,8),
              channel_dim: int = 2,
              maxnum: Optional[int] = None,
              dbname: str = 'images'):
    """ save images to h5 dataset
    Overly complicated
    Args
        images      (str) folder with identically sized images
                    (tuple) list of images
        h5name      (str) non existing fname
        buffer_size (int [2]): # rdcc_nbytes = buffer_size*shape
        down_scales (tuple [2,4,8]) create downscaled versions of database
        channel_dim (int [2 or 0]) 2: save as numpy (h, w, c), 0 save as torch (c, h, w)
        maxnum      (int [None]) of images debugging purposes
        dbname      (str ['images']) default db name
    """
    # validate inputs
    # only create if not existing
    assert not osp.isfile(h5name), f"{h5name} exists... nothing done"

    assert isinstance(images, (list, str, tuple)), f"expected folder or images tuple got {images}"
    if isinstance(images, str):
        images = [f.path for f in os.scandir(images)
                if osp.splitext(f.name)[-1].lower() in ('.jpg', '.png')]
        assert images, f"no imgs fund in {images}"
        images.sort()
    else:
        assert all([osp.isfile(img) for img in images]), f" expected tuple of files, got {images}"

    images = images[:maxnum]

    # set up formats (store down scaled, order (c,h,w) or (h,w,c))
    down_scales = down_scales or []
    shape = _reshape(np.asarray(Image.open(images[0])).shape, channel_dim=channel_dim)

    with h5py.File(h5name, 'w') as hf:
        # create datasets
        dt = h5py.string_dtype(encoding='utf-8')
        dataset = hf.create_dataset('names', (len(images),), dtype=dt)
        for i, name in enumerate(images):
            dataset[i] = osp.basename(name)

        print(f"dataset: images {shape}")
        hf.create_dataset(dbname, shape=(len(images), *shape), dtype='uint8',
                          chunks=(1, *shape), rdcc_nbytes=buffer_size*np.prod(shape))
        shapes = {}
        for j in down_scales:
            shapes[j] = _scale_shape(shape, j)
            print(f"dataset: {dbname}_{j} {shapes[j]}")
            hf.create_dataset(f'{dbname}_{j}', shape=(len(images), *shapes[j]), dtype='uint8',
                    chunks=(1, *shapes[j]), rdcc_nbytes=buffer_size*np.prod(shape))

        for i, img in enumerate(images):
            print(f"img[{i}/{len(images)}] {img}")
            _img =Image.open(img)
            nimg = np.asarray( _img)
            if channel_dim == 0:
                nimg = nimg.transpose(2,0,1)
            hf[dbname][i] = nimg
            if i == 0:
                print(nimg.shape)

            for j, size in shapes.items():
                if i == 0:
                    size.pop(channel_dim)
                print(f" -> {j}")
                nimg = np.asarray( _img.resize(size[::-1]))
                if channel_dim == 0:
                    nimg = nimg.transpose(2,0,1)
                hf[f'{dbname}_{j}'][i] = nimg
                if i == 0:
                    print(f" -> {j} {nimg.shape}")


def _reshape(shape, channel_dim=2):
    if len(shape) == 2:
        shape = (*shape, 1)
    if not channel_dim:
        shape = (shape[2], *shape[:2])
    return shape

def _scale_shape(shape, div=2):
    shape = list(shape)
    for i, a in enumerate(shape):
        shape[i] = a if a in (1,3,4) else round(a/div)
    return shape


def shuffle(indices: Union[list, int]) -> list:
    if isinstance(indices, int):
        indices = list(range(indices))
    random.shuffle(indices)
    return indices

def sample_images(hf, indices, dname='images'):
    """ requires an open h5 dataset
    with h5py.File(h5name, 'r') as hf:
        indices = len(hf["images"])
        for i in range(training steps):
            indices = shuffle(indices)
            inputs = sample_images(hf, indices[10:])    
    """
    return np.array(hf[dname][indices])

def get_image(h5name: str, index: int, dname: str = 'images', resolution: int = 0) -> np.ndarray:
    """ loads a (6413,9657,3) image in 0.16s approx 143 X faster than np.array(Image.open()) 22.66s
    opening h5py file has overhead so single images should all work within single with h5py
    Args
        h5name      (str) valid h5 file
        index       (int) index of image
        dname       (str) dataset name
        resolution  (int [0]) downscale factor
    plt.imshow(get_image(S.images_path, r.image_id, resolution=8));plt.show()
    """
    if resolution:
        dname = f"{dname}_{resolution}"
    with h5py.File(h5name, 'r') as hf:
        assert dname in hf, f'{dname} not found in {h5name}'
        assert index in range(len(hf[dname])), \
            f"requested [{dname}][{index}], exceeds {len(hf[dname])}"
        return np.array(hf[dname][index])

def speed_check(image, hfname, cycles=10):
    """ compare times loading image w PIL from disk vs h5py
    """
    times = {}
    sizes = []

    # reads headers only
    _start = time.time()
    for i in range(cycles):
        sizes += [Image.open(image).size]
    times['read_header'] = (time.time() - _start)/cycles

    # reads image to numpy
    _start = time.time()
    for i in range(cycles):
        im = np.array(Image.open(image))
        sizes += [im.shape]
    times['read_image'] = (time.time() - _start)/cycles

    # reads image to numpy from h5
    _start = time.time()
    for i in range(cycles):
        im = get_image(hfname, 0)
        sizes += [im.shape]
    times['read_h5'] = (time.time() - _start)/cycles

# H% dataset DEBUG
# #

# class H5Dataset0(torch.utils.data.Dataset):
#     # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
#     def __init__(self, path):
#         self.fname = path
#         self.dataset = None
#         with h5py.File(self.fname, 'r') as file:
#             self.dataset_len = len(file["dataset"])

#     def __getitem__(self, index):
#         if self.dataset is None:
#             self.dataset = h5py.File(self.fname, 'r')["dataset"]
#         return self.dataset[index]

#     def __len__(self):
#         return self.dataset_len


# class H5Dataset(torch.utils.data.Dataset):
#     def __init__(self, path, name='images', levels=(2,4,8)):
#         self.fname = path
#         self.datasets = {}
#         self.name = name
#         with h5py.File(self.fname, 'r') as file:
#             self.dataset_len = len(file[name])
#             keys = file.keys()
#         for k in keys:
#             self.datasets[k] =  h5py.File(self.fname, 'r')[k]


#     def __getitem__(self, index=None, level=None, section=None):
#         if index is None:
#             index = torch.randint(self.dataset_len)
#         levels = {0:8,2:4,4:2,8:0}
#         if level is None:
#             level = list(levels.keys())[torch.randint(4)]
#         sections = levels[level]**2
#         if section is None:
#             section = torch.randint(sections)
#         section = section % sections

#         name = self.name if level is None else f"{self.name}_{level}"
#         if name not in self.datasets:
#             self.datasets[name] = h5py.File(self.fname, 'r')[name]

#         return self.datasets[name][index]

#     def __len__(self):
#         return self.dataset_len
    
#
#
# multiprocessing h5 twriting DEBUG
#


def load_and_resize_image(fname: str, sizes: tuple, dbname: str = "images") -> dict:
    """ return dict with downscaled images
    Args
        fname   (str) file path
        sizes   (tuple [2,4,8])
    out  
    """
    with Image.open(fname) as img:
        images = {dbname: np.array(img)}
        for scale in sizes:
            size = (round(img.size[0]/scale), round(img.size[1]/scale))
            images[f'{dbname}_{scale}'] = np.array(img.resize(size, Image.BICUBIC))
        return images

def h5_add_file(images: dict, h5f: h5py.File, sizes: tuple, num_images: int, i: int, dbname: str = "images"):
    """
    """
    fullsize_image = images[dbname]
    rdcc_nbytes_fullsize = fullsize_image.nbytes  # Cache size for one full-size image

    for scale in sizes:
        dname = f'{dbname}_{scale}' if scale != 1 else dbname
        image = images[dname]

        if dname not in h5f:
            # Calculate the number of smaller images that fit in the rdcc_nbytes of a full-size image
            num_images_in_cache = rdcc_nbytes_fullsize // image.nbytes
            num_images_in_cache = max(1, num_images_in_cache)  # Ensure at least one image is cached

            # Create the dataset with chunking
            dset = h5f.create_dataset(dname,  shape=(num_images,) + image.shape,
                dtype=image.dtype, chunks=image.shape, rdcc_nbytes=rdcc_nbytes_fullsize )
            # Adjust the cache size for the dataset based on scale
            h5f[dname].id.set_chunk_cache(image.shape, num_images_in_cache, 1.0)
        else:
            dset = h5f[dname]

        dset[i] = image
        # # Resize the dataset to accommodate the new image
        # dset.resize(dset.shape[0] + 1, axis=0)
        # dset[-1] = image
    print(f"{i}/{num_images}")

def h5_images_mp(folder: str, h5name: str, sizes: tuple = (2, 4, 8)) -> None:
    """
    Create h5 dataset multiprocessed
    untested
    """
    image_paths = [f.path for f in os.scandir(folder)
                   if osp.splitext(f.name)[-1].lower() in ('.jpg', '.png')]
    image_paths.sort()

    num_images = len(image_paths)

    with Image.open(image_paths[0]) as img:
        fullsize_image = np.array(img)
        rdcc_nbytes_fullsize = fullsize_image.nbytes

    # Create the output h5py file
    with h5py.File(h5name, 'w', rdcc_nbytes=rdcc_nbytes_fullsize) as h5f:
        with Pool(cpu_count()) as pool:
            for i, imagedict in enumerate(pool.map(partial(load_and_resize_image, sizes=sizes), image_paths)):
                h5_add_file(imagedict, h5f, sizes=sizes, num_images=num_images, i=i)

