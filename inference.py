import argparse
import torch
import numpy as np
from os.path import join
import os
import cv2
from tqdm import tqdm

from model import model as model_arch
from data_loader.data_loader import InferenceDataLoader
from utils.util import torch2cv2, append_timestamp, setup_output_folder

from parse_config import ConfigParser

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    model_info['num_bins'] = config['arch']['args']['config']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def main(args, model):
    dataset_kwargs = {'transforms': {'LegacyNorm': {}},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'voxel_method': {'method': 'between_frames'},
                      'calibrate': args.calibrate
                      }


    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)

    ts_fname = setup_output_folder(args.output_folder)
    
    frames = []
    model_states = None
    for i, item in enumerate(tqdm(data_loader)):
        voxel = item['events'].to(device)

        output = model(voxel, model_states)
        model_states = output['state']

        # save sample images, or do something with output here
        image = torch2cv2(output['image'])
        frames.append(image)
        fname = 'frame_{:010d}.png'.format(i)
        cv2.imwrite(join(args.output_folder, fname), image)
        append_timestamp(ts_fname, fname, item['timestamp'].item())
    
    # Get the height and width from the first frame
    height, width = frames[0].shape[:2]
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
    video_path = os.path.join(args.output_folder, "video.mp4")
    fps = 20
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Ensure the frame is in the correct format (BGR)
        if len(frame.shape) == 2:  # if the frame is grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved to {video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Calibrate the hot and cold pixels in the camera')

    args = parser.parse_args()
    
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    model = load_model(checkpoint)
    main(args, model)