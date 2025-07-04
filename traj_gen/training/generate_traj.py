import os
import torch
import torch.nn.functional as F
import imageio
import argparse
from pipeline import TrajGenPipeline
from ds import RegularVideoLoader




def process_one_video(data_dict: dict, traj_gen: TrajGenPipeline,  args, verbose=False):
    video_name = data_dict["video_name"]
    video = data_dict['video']
    save_video = data_dict["save_video"]
    write_folder = data_dict['write_folder']
    
    os.makedirs(write_folder, exist_ok=True)
        
    # save the shortened video
    video_path = os.path.join(write_folder, f"{video_name}.mp4")
    extension = os.path.splitext(os.path.basename(video_path))[1]
    if not os.path.exists(video_path): imageio.mimsave(video_path, save_video, fps=5, codec="libx264",)
    
    if os.path.exists(video_path.replace(extension, f"_mask{args.version_ext}.npz")): 
        return
    mask, graph = traj_gen.traj_generation(video=video, video_path=video_path, use_key_frame_selection=args.use_key_frame, save_to_disk=True, version_ext=args.version_ext, verbose=verbose)

    if args.visualize:
        traj_gen.visualize(video, video_graph=graph, return_masks=mask, 
            visualize_dir=args.visualize_dir, video_name=video_name, )
    
    
    
def main(args):
    traj_gen = TrajGenPipeline(
        sam2_config=args.sam2_cfg,
        frame_num=16
    )
    
    dataset = RegularVideoLoader(
        basepath=args.video_dir, jsonfile=args.json_path, image_size=args.image_size, num_frames=args.num_frames, split=args.split, total_split=args.total_split, version_ext=args.version_ext)
    
    streams = [torch.cuda.Stream() for _ in range(args.batch_size)]
    generator = dataset.generate_batch(batch_size=args.batch_size, num_workers=args.num_workers)
    
    verbose = True
    for batch in generator:
        for i in range(args.batch_size):
            stream = streams[i]
            with torch.cuda.stream(stream):
                try: 
                    process_one_video(batch[i], traj_gen, args, verbose=verbose)
                    verbose=False
                except Exception as e:
                    print("!Exception:", batch[i]['video_name'], e)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    
    # Arguments for model configurations
    parser.add_argument("--sam2_cfg", type=str, default='sam2/configs/sam2.1_training/sam2.1_hiera_s_MOSE_finetune.yaml', help="Path to SAM2 config file.")

    # Dataset and processing arguments
    parser.add_argument("--json_path", type=str, required=True, help="path to dataset json")
    parser.add_argument("--video_dir", type=str, required=True, help="path to dataset json")
    parser.add_argument("--version_ext", type=str, default="")
    parser.add_argument("--visualize_dir", type=str, default="")
    parser.add_argument("--split", type=int, default=0, help="Dataset split number.")
    parser.add_argument("--split_ten", type=int, default=0, help="split number in ten digit")
    parser.add_argument("--total_split", type=int, default=1, help="split number in ten digit")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video.")
    parser.add_argument("--image_size", type=int, default=512, help="Size of the input image for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker threads for data loading.")
    
    # Optional flags
    parser.add_argument("--use_key_frame", action="store_true", help="Whether to use key frame selection.")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results.")

    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    main(args)