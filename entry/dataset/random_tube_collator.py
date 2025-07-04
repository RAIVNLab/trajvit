# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing import Value

from logging import getLogger

import torch
import numpy as np

logger = getLogger()


class TubeMaskCollator(object):

    def __init__(
        self,
        default_collator=None,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
        ratio=0.5,  # mask ratio
    ):
        super(TubeMaskCollator, self).__init__()

        self.default_collator = default_collator
        self.mask_generator = _MaskGenerator(
            crop_size=crop_size,
            num_frames=num_frames,
            spatial_patch_size=patch_size,
            temporal_patch_size=tubelet_size,
            ratio=ratio,
        )

    def step(self):
        self.mask_generator.step()

    def __call__(self, batch):

        batch_size = len(batch)
        collated_batch = self.default_collator(batch)
        
        masks_enc = self.mask_generator(batch_size)

        return collated_batch, masks_enc


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size[0], crop_size[1] // spatial_patch_size[1]
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_spatial = self.height*self.width

        self.ratio = ratio

        self.num_keep_spatial = int(self.num_patches_spatial*(1.-self.ratio))
        self.num_keep = self.num_keep_spatial * self.duration

        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch_size):
        def sample_mask():
            mask = np.hstack([
                np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                np.ones(self.num_keep_spatial),
            ])
            np.random.shuffle(mask)
            mask = torch.tensor(np.tile(mask, (self.duration, 1)))
            mask = mask.flatten()
            mask_e = torch.nonzero(mask).squeeze()
            return mask_e

        collated_masks_enc = []
        for _ in range(batch_size):
            mask_e = sample_mask()
            collated_masks_enc.append(mask_e)

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc
