"""@ xvdp
Dataset to serve view points while loading cameras from disk
using torch Dataloader
"""
from typing import TypeVar, Union, Optional, Any
import os
import os.path as osp
from copy import deepcopy
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset


_T = TypeVar('_T')

# pylint: disable=no-member

class CamInfoDataset(Dataset):
    """ load viewpoints
    """
    def __init__(self, camera_list, root_dir):
        self.camera_list = camera_list
        self.names = {osp.splitext(f.name)[0]:f.path for f in os.scandir(root_dir)}

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        """return only what is needed for training"""
        out = self.camera_list[idx]
        image = image_8bit_to_tensor(self.names[out.image_name])
        return [image, out.world_view_transform, out.full_proj_transform,
                out.camera_center, out.FoVx, out.FoVy, out.image_width, out.image_height]


def image_8bit_to_tensor(image: Union[str, ImageFile.ImageFile, np.ndarray],
                         device: Union[torch.device, str, None] = None,
                         dtype: Optional[torch.dtype] = None ) -> torch.Tensor:
    """ converts, 8bit image path, PIL.Image, or ndarray to
    applies alpha if present
    out Tensor (1|3, H, W)
    Args:
        image   (str, PIL.Image, ndarray) 8 bit image
        devie   (str, torch.device [None]
    """
    if isinstance(image, str):
        image = Image.open(image)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    image = torch.as_tensor(np.array(image), dtype=dtype,device=device) / 255.0
    if image.ndim == 3:
        image = image.permute(2,0,1).contiguous()
    elif image.ndim == 2:
        image = image[None]
    if len(image) == 4:
        image[:3] *= image[3:]
    return image[:3]


class ObjDict(dict):
    """ dict with attrs
    Examples:
    >>> d = ObjDict(**{'some_key':some_value})
    >>> d = ObjDict(some_key=some_value)
    subset of koreto.ObjDict 
        """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
        if isinstance(value, dict):
            self[name] = ObjDict(self[name])
            self[name]._recurse_obj()

    def __delattr__(self, name: str) -> None:
        del self[name]

    def getkey(self, index: int) -> Any:
        """ get key by index"""
        return list(self.keys())[index]

    def getvalue(self, index: int) -> Any:
        """ get value by index"""
        return list(self.values())[index]

    def getitem(self, index: int) -> tuple:
        """ get (key,value) by index"""
        return list(self.items())[index]

    def update_exclusive(self, *args, **kwargs) -> None:
        """ update only existing kwargs
        """
        for a in args:
            if isinstance(a, dict):
                kwargs.update(a)
        upk = {k:v for k,v in kwargs.items() if k in self}
        self.update(**upk)

    def copyobj(self: _T) -> _T:
        """ .copy() returns a dict, not ObjDict"""
        return ObjDict(self.copy())

    def deepcopy(self: _T) -> _T:
        """this is ok except in case of pytorch
        """
        return deepcopy(self)
