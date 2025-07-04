import cv2
import io
import zlib
import av
import torch
import numpy as np

    
# Helper function to decompress segmentation mask from PNG
def decompress_segmentation_mask(png_bytes):
    masks = []
    for i in range(len(png_bytes)):
        mask_np = np.frombuffer(png_bytes[i], np.uint8)  # Read PNG bytes
        mask_decoded = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)  # Decode PNG
        masks.append(mask_decoded)
    return torch.tensor(masks, dtype=torch.int32)


# Helper function to decompress graph using zlib
def decompress_graph(compressed_graph,M):
    graph_bytes = zlib.decompress(compressed_graph)
    graph = np.frombuffer(graph_bytes, dtype=np.int32).reshape(M,-1)
    
    return torch.tensor(graph, dtype=torch.int32)




        


# # Example tensors (replace with actual data)
# t = 30  # Number of frames
# w, h = 640, 360  # Width and height
# M = 10  # Size of graph

# # Simulate a video (t, 3, w, h), mask (t, w, h), and graph (M, t)
# video = torch.randint(0, 256, (t, 3, w, h), dtype=torch.uint8)  # Video data
# mask = torch.randint(0, 100, (t, w, h), dtype=torch.int32)  # Segmentation mask with object IDs
# graph = torch.randint(0, 2, (M, t), dtype=torch.int32)  # Metadata matrix

# # Writing to TFRecord file
# output_tfrecord_file = 'compressed_video_data.tfrecord'

# with tf.io.TFRecordWriter(output_tfrecord_file) as writer:
#     write_to_tfrecord(video, mask, graph, M, 'hi', 'hello', 'world', writer, )

# print(f"Data successfully written to {output_tfrecord_file}")

# load_and_verify_tfrecord('/gscratch/krishna/chenhaoz/videotok/datasets/video_images/gcp/panda70m_dense_caption/split_1/00000_withGraph.tfrecord')


