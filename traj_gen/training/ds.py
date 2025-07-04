
import torch
import numpy as np
import torch.nn.functional as F
import glob
import os
from torch.utils.data import Dataset, DataLoader
import csv
import json
import av
import random
from utils.tool import get_frame_indices


import decord
decord.bridge.set_bridge("torch")

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
    

def divide_into_sublists(lst, total=10):
    # Calculate the size of each chunk
    n = len(lst)
    chunk_size = n // total
    remainder = n % total
    
    sublists = []
    start = 0
    
    # Divide the list into 10 sublists
    for i in range(total):
        end = start + chunk_size + (1 if remainder > 0 else 0)
        sublists.append(lst[start:end])
        start = end
        remainder -= 1

    return sublists


class RegularVideoData(Dataset):
    def __init__(self, 
        base_path=None,
        anno_file=None,
        num_frames = 16,
        sample='middle',
        image_size=512,
        total_split=10,
        split=-1,
        version_ext=''
    ):
        self.video_base_path = base_path  # required by json file
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.sample = sample
        self.num_frames = num_frames
        self.split = split
        self.version_ext = version_ext
        data = self.read_data(anno_file)
        
        if self.split >= 0:
            print(f"Split dataset into {total_split} subsets!")
            chunks = divide_into_sublists(data, total=total_split)
            data = chunks[split]
            
        self.data = self.filter_data(data)
        print("original data length", len(data), "filtered data length", len(self.data))
        
            
    def filter_data(self, data):
        filtered_data = []
        for datum in data:
            if self.video_base_path: datum['video'] = os.path.join(self.video_base_path, datum['video'])
            if "_short" in datum['video']: continue
            video_name, out_dir = self._get_video_name_and_out_dir(datum)
            short_video_path = os.path.join(out_dir, video_name+".mp4")
            graph_path = os.path.join(out_dir, video_name.replace("_short", "")+f"_graph{self.version_ext}.npz")
            if not os.path.exists(short_video_path) or not os.path.exists(graph_path):
                filtered_data.append(datum)
        return filtered_data
    
    
    def _get_out_video_name(self, video_path, videoid):
        subpath = os.path.splitext(os.path.basename(video_path))[0]
        if videoid>=0: subpath = subpath + f"_{videoid}"
        if self.num_frames == 16:
            return subpath + "_short"
        else:
            return subpath + "_short" + f"_{self.num_frames}"
    
    def _get_video_name_and_out_dir(self, datum):
        videoid = datum.get('id', -1)
        video_name = self._get_out_video_name(datum['video'], videoid)
        out_dir = os.path.dirname(datum['video'])
        return video_name, out_dir
        
    def read_data(self, annofile):
        ext = os.path.splitext(annofile)[1]
        if ext == '.csv':
            file_paths = []
            with open(annofile, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    path = row[0].split(',')[0]
                    file_paths.append(path)
            return file_paths
        elif ext == '.json':
            with open(annofile, 'r') as f: jsondata = json.load(f)
            return jsondata   
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path, num_frames, start, end):
        try:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            vlen = len(video_reader)
            frame_indices = get_frame_indices(
                num_frames, vlen, sample=self.sample, input_fps=video_reader.get_avg_fps(), start=start, end=end
            )
            frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
            return frames
        except:
            container = av.open(video_path)
            video_frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            video_arr =  np.stack(video_frames)
            vlen = video_arr.shape[0]
            if self.num_frames < vlen:
                sample_indices = get_frame_indices(num_frames=self.num_frames, sample=self.sample, vlen=vlen)
                arr = torch.from_numpy(video_arr[sample_indices])
            else:
                arr = torch.from_numpy(video_arr)
            return arr
                    
    
    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video_path = self.data[idx]['video']
                
                if "timing" in self.data[idx].keys():
                    start, end = self.data[idx]['timing']
                else:
                    start, end = None, None
                arr = self._load_video(video_path, num_frames=self.num_frames, start=start, end=end)
                video_name, out_dir = self._get_video_name_and_out_dir(self.data[idx])
                success = True
            except Exception as e:
                print("getitem Exception", e)
                idx = random.randint(0,len(self.data)-1)
                continue

        return_arr = arr
        arr = arr.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

        # Resize and normalize for mp4
        arr = F.interpolate(arr, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        mean = torch.tensor(self.image_mean).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
        std = torch.tensor(self.image_std).view(1, 3, 1, 1)    # Shape (1, C, 1, 1)
        arr = arr.float() / 255.0  # Scale to [0, 1]
        arr = (arr - mean) / std  # Normalize

        return arr, video_name, out_dir, return_arr




class VideoLoader():
    def __init__(self, basepath=None, jsonfile=None, image_size=1024, num_frames=16):
        self.basepath = basepath
        self.jsonfile = jsonfile
        self.image_size = image_size
        self.num_frames = num_frames

    def generate_batch(self, batch_size, num_workers):
        """ yield data format:
            return_dict = {
                "video_name": str, 
                "video": [T,C,H,W], torch.tensor
                "save_video": [T,H,W,C], np.array,
                "subfolder_name": str,
                "caption": str
            }
        """
        raise NotImplementedError



from tqdm import tqdm 
class RegularVideoLoader(VideoLoader):
    def __init__(self, basepath=None, jsonfile=None, image_size=1024, num_frames=16, split=-1, total_split=10, version_ext=''):
        super().__init__(basepath, jsonfile, image_size, num_frames)
        self.num_frames = num_frames
        self.dataset = RegularVideoData(
            base_path=basepath,
            anno_file=jsonfile,
            num_frames=num_frames,
            image_size=image_size,
            split=split,
            total_split=total_split,
            version_ext=version_ext
        )
        
    def generate_batch(self, batch_size, num_workers):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers, pin_memory=True, )
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            arr, video_name, out_dir, return_arr = batch
            
            yield [{
                    "video_name":  video_name[i],
                    "video": arr[i].squeeze(),
                    "save_video": return_arr[i].squeeze().numpy(),
                    "write_folder": out_dir[i],
                } for i in range(batch_size)]