from collections import OrderedDict
from typing import Dict, Tuple, Union

import torch
import torchvision
from data.base_dataset import get_transform
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from PIL import Image

from api.const import Global_config
from api.util import timing


class SwAeController:
    load_size: int = -1
    transform: Union[torchvision.transforms.Compose, None] = None
    global_sty: Union[torch.Tensor, None] = None
    global_tex: Union[torch.Tensor, None] = None
    structure_path: Union[str, None] = None
    cache: Dict = {}
    sty_argumentation: OrderedDict = OrderedDict()

    @timing
    def __init__(self, name: str) -> None:
        """Initilise the model and other options

        Args:
            name (str): [description]
        """
        self.opt = Global_config(isTrain=False, name=name)
        self.model = SwappingAutoencoderModel(self.opt)
        self.model.initialize()

    @timing
    def _get_transform(self) -> torchvision.transforms.Compose:
        kwarg = {}
        if self.load_size != -1:
            kwarg["load_size"] = self.load_size
        return get_transform(self.opt, **kwarg)

    @timing
    def set_size(self, size: int) -> None:
        """Sets transform to load images with the `size`. Output is also of width `size`. It must be greater than 128 and must be a multiple of 4.


        Args:
            size (int): size of the ouput image.

        Raises:
            ValueError: if the size is not a valid integer.
        """
        if size < 0 or size % 2 == 1 or size < 128:
            raise ValueError("invalid size")
        self.load_size = size

        # need to reload transforms with new size
        self.transform = self._get_transform()

    @timing
    def _load_image(self, path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.transform == None:
            self.transform = self._get_transform()
        tensor = self.transform(img).unsqueeze(0)
        return tensor

    @timing
    def set_structure(self, structure_path: str) -> None:
        """set the structure, must be called before compute(). Doesn't cache the image. But, sets the noise input for the model

        Args:
            structure_path (str): path to the structure image
        """
        if structure_path == None and self.structure_path == structure_path:
            return
        if self.structure_path == None:
            self.structure_path = structure_path
        source = self._load_image(structure_path).to("cuda")
        with torch.no_grad():
            self.model(sample_image=source, command="fix_noise")
            self.global_tex, self.global_sty = self.encode(source)

    @timing
    def encode(self, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tex, sty = self.model(im.to("cuda"), command="encode")
            im = im.to("cpu")
        return tex.to("cpu"), sty.to("cpu")

    @timing
    def load_encode_cache(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if path in self.cache:
            return self.cache[path]
        im = self._load_image(path)
        tex, sty = self.encode(im)
        self.cache[path] = tex, sty
        return tex, sty

    @timing
    def mix_style(self, style_path: str, alpha: float) -> None:
        """Mixes the style of the image given with the current structure image by the factor of alpha. Caches the encoded image. actual mixing happens when `compute()` is called

        Args:
            style_path (str): Path to the image whose style you want to mix
            alpha (float): Value of mix factor. 0 would remove this image from the mix, 1 implies using NONE of the original styles
        """
        if alpha == 0:
            if style_path in self.sty_argumentation:
                del self.sty_argumentation[style_path]
            return

        if style_path not in self.cache:
            tex, sty = self.load_encode_cache(style_path)
        # assume alpha has changed if same path is sent
        self.sty_argumentation[style_path] = alpha

    @timing
    def compute(self) -> torch.Tensor:
        """Computes the output of the operations performed by the mix_style and gives the output image

        Returns:
            torch.Tensor: output tensor with the shape Tensor with shape (1, 3, h, w) where `h` and `w` are height and width.
        """
        assert self.global_sty != None and self.global_tex != None
        torch.cuda.empty_cache()
        local_sty = self.global_sty.clone()
        for path, alpha in self.sty_argumentation.items():
            cached_sty = self.cache[path][1].clone()
            local_sty = self.lerp(local_sty, cached_sty, alpha)
        with torch.no_grad():
            out = self.model(self.global_tex.to("cuda"), local_sty.to("cuda"), command="decode")
        local_sty = local_sty.to("cpu")
        return out

    @staticmethod
    @timing
    def lerp(source: torch.Tensor, target: torch.Tensor, alpha: int) -> torch.Tensor:
        return source * (1 - alpha) + target * alpha
