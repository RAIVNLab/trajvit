from multiprocessing import Value

from logging import getLogger

import torch
import numpy as np

logger = getLogger()



class RandomMaskCollator(object):

    def __init__(
        self,
        default_collator,
        ratio,
    ):
        super(RandomMaskCollator, self).__init__()
        self.default_collator = default_collator
        self.mask_generator = _MaskGenerator(
            ratio=ratio
        )
    
    def step(self):
        self.mask_generator.step()
            
    def __call__(self, batch):

        batch_size = len(batch)
        collated_batch = self.default_collator(batch)
        
        # for i, b in enumerate(collated_batch):
        #     if type(b)!=list: print(i, b.shape)
        #     else: print(i, "length", len(b))
        
        max_num_token = collated_batch[-2].shape[1]-1
        # print("max token", max_num_token)
        # print("=========================")
        masks_enc = self.mask_generator(batch_size, max_num_token)

        return collated_batch, masks_enc
    


class _MaskGenerator(object):
    
    def __init__(
        self,
        ratio=0.9,
    ) -> None:
        super(_MaskGenerator, self).__init__()
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def __call__(self, batch_size, max_token_num):
        # """ generate mask indices based on different token numbers for videos in the same batch """
        """ generate mask indices based on the maximum token number in a batch """
        def sample_mask():
            num_token_keep = int(max_token_num * (1-self.ratio))
            mask = np.hstack([
                np.zeros(max_token_num - num_token_keep),
                np.ones(num_token_keep)
            ])
            np.random.shuffle(mask)
            mask = torch.tensor(mask)
            mask_e = torch.nonzero(mask).squeeze()
            return mask_e

        collated_masks_enc = []
        for _ in range(batch_size):
            mask_e = sample_mask()
            collated_masks_enc.append(mask_e)

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
                    
        return collated_masks_enc
        
            
            