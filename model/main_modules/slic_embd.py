import torch
import torch.nn as nn
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
from timm.models.layers import to_2tuple


class SlicPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding with Superpixel Segmentation """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, use_superpixel=True, n_segments=100):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.use_superpixel = use_superpixel
        self.n_segments = n_segments
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        if self.use_superpixel:
            superpixels = np.zeros((B, self.n_segments, *self.patch_size, C))
            for i, image in enumerate(x):
                # image = np.transpose(image, (1, 2, 0))  # Correct channel position for skimage
                segments = slic(image, n_segments=self.n_segments, compactness=10, start_label=0)
                for s in range(self.n_segments):
                    mask = segments == s
                    segment = image * np.expand_dims(mask, axis=-1)  # Apply mask over the last dimension
                    resized_segment = resize(segment.permute(1, 2, 0), self.patch_size, anti_aliasing=True)
                    superpixels[i, s] = resized_segment
            x = torch.tensor(superpixels, dtype=torch.float32).to(x.device)  # Convert back to tensor and move to the original device
            x = x.view(B, self.n_segments * C, *self.patch_size)
            x = x.flatten(2).transpose(1, 2)  # Apply convolution and flatten output

        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = SlicPatchEmbed()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
