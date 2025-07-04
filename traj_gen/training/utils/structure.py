import torch

class BackboneOut:
    def __init__(self, data):
        self.store = data

    def __len__(self):
        return len(self.store['vision_features'])

    def __getitem__(self, idx):
        if isinstance(idx, slice): start, stop = idx.start, idx.stop
        elif isinstance(idx, int): start, stop = idx, idx+1
        else: 
            raise NotImplementedError
        
        sub_backbone_out = {
            'vision_features': self.store['vision_features'][start:stop],
            'vision_pos_enc': [b[start:stop] for b in self.store['vision_pos_enc']],
            'backbone_fpn': [b[start:stop] for b in self.store['backbone_fpn']],
        }
        if isinstance(idx, slice): return BackboneOut(sub_backbone_out)
        else: return sub_backbone_out 
        
    def append(self, value):
        self.store = {
            'vision_features': torch.cat([self.store['vision_features'], value['vision_features']]),
            'vision_pos_enc': [torch.cat([self.store['vision_pos_enc'][i], value['vision_pos_enc'][i]]) for i in range(len(self.store['vision_pos_enc']))],
            'backbone_fpn': [torch.cat([self.store['backbone_fpn'][i], value['backbone_fpn'][i]]) for i in range(len(self.store['backbone_fpn']))],
        }
        
    def join(self, backboneouts):
        for b in backboneouts:
            self.append(b.store)
        return self
    