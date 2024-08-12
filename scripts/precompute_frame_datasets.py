# scripts/precompute_frame_datasets.py

r"""Prepare datasets as we need multiple frames from each video."""

import os
import matplotlib.pyplot as plt
import imageio
import hydra
from src.datasets.coco import COCODataset

original_freq = 25
new_freq = 2
frames_to_extract = 6

@hydra.main(config_path='../config', config_name='config')
def extract_frames(args):
    
    # TRAIN
    print("\n---------- Train set ----------")
    train_dataset = COCODataset(args.dataDir, args.trainAnnFile, args.numClass)
    train_video_dir = args.trainVideoFrames
    print(train_dataset.__len__())

    video_dir = args.videoDir

    # create dir if not exists
    train_frame_path = os.path.join(args.dataDir, args.trainVideoFrames)
    os.makedirs(train_frame_path, exist_ok=True)
    print(train_frame_path)
    
    '''
    for i in range(train_dataset.__len__()):
        imgID = train_dataset.ids[i]
        file_name = train_dataset.coco.imgs[imgID]['file_name']
        f, date, clip_num, frame_num = file_name.split('.')[0].split('/')
        video_path = os.path.join(video_dir, date, clip_num + '.mp4')

        vid = imageio.get_reader(video_path, 'ffmpeg')
        
        frame_num = int(frame_num.split('_')[-1]) + 1
        #print(frame_num, frame_num * original_freq)
        current_frame_num = frame_num * original_freq
        
        # check if the directory exists, if not create it if yes, skip
        file_dir = file_name.split('/')[:-1]
        file_dir = os.path.join(train_frame_path, *file_dir)

        # check if the directory exists and contains a file starting with the same name
        if not os.path.exists(file_dir):
            print(file_dir)
            os.makedirs(file_dir, exist_ok=True)

        files = os.listdir(file_dir)
        if any([file_name.split('.')[0] in file for file in files]):
            continue
        else:
            for j in range(1, frames_to_extract + 1):
                new_frame_num = current_frame_num - j * new_freq
                new_frame = vid.get_data(new_frame_num)
                new_frame_name = file_name.split('.')[0] + f"_frame_{-j}.jpg"
                new_frame_path = os.path.join(train_frame_path, new_frame_name)
                imageio.imwrite(new_frame_path, new_frame)
            
    '''   
    
    # VAL
    print("\n---------- Val set ----------")
    val_dataset = COCODataset(args.dataDir, args.valAnnFile, args.numClass)
    val_video_dir = args.valVideoFrames
    print(val_dataset.__len__())

    # create dir if not exists
    val_frame_path = os.path.join(args.dataDir, args.valVideoFrames)
    os.makedirs(val_frame_path, exist_ok=True)

    for i in range(val_dataset.__len__()):
        imgID = val_dataset.ids[i]
        file_name = val_dataset.coco.imgs[imgID]['file_name']
        f, date, clip_num, frame_num = file_name.split('.')[0].split('/')
        video_path = os.path.join(video_dir, date, clip_num + '.mp4')

        vid = imageio.get_reader(video_path, 'ffmpeg')
        
        frame_num = int(frame_num.split('_')[-1]) + 1
        #print(frame_num, frame_num * original_freq)
        current_frame_num = frame_num * original_freq
        
        # check if the directory exists, if not create it
        file_dir = file_name.split('/')[:-1]
        file_dir = os.path.join(val_frame_path, *file_dir)

        if not os.path.exists(file_dir):
            print(file_dir)
            os.makedirs(file_dir, exist_ok=True)

        files = os.listdir(file_dir)
        if any([file_name.split('.')[0] in file for file in files]):
            continue
        else:
            for j in range(1, frames_to_extract + 1):
                new_frame_num = current_frame_num - j * new_freq
                new_frame = vid.get_data(new_frame_num)
                new_frame_name = file_name.split('.')[0] + f"_frame_{-j}.jpg"
                new_frame_path = os.path.join(val_frame_path, new_frame_name)
                imageio.imwrite(new_frame_path, new_frame)



    # TEST
    print("\n---------- Test set ----------")
    test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass)
    test_video_dir = args.testVideoFrames
    print(test_dataset.__len__())

    # create dir if not exists
    test_frame_path = os.path.join(args.dataDir, args.testVideoFrames)
    os.makedirs(test_frame_path, exist_ok=True)

    for i in range(test_dataset.__len__()):
        imgID = test_dataset.ids[i]
        file_name = test_dataset.coco.imgs[imgID]['file_name']
        f, date, clip_num, frame_num = file_name.split('.')[0].split('/')
        video_path = os.path.join(video_dir, date, clip_num + '.mp4')

        vid = imageio.get_reader(video_path, 'ffmpeg')
        
        frame_num = int(frame_num.split('_')[-1]) + 1
        print(frame_num, frame_num * original_freq)
        current_frame_num = frame_num * original_freq
        
        # check if the directory exists, if not create it
        file_dir = file_name.split('/')[:-1]
        file_dir = os.path.join(test_frame_path, *file_dir)

        if not os.path.exists(file_dir):
            print(file_dir)
            os.makedirs(file_dir, exist_ok=True)

        files = os.listdir(file_dir)
        if any([file_name.split('.')[0] in file for file in files]):
            continue
        else:
            for j in range(1, frames_to_extract + 1):
                new_frame_num = current_frame_num - j * new_freq
                new_frame = vid.get_data(new_frame_num)
                new_frame_name = file_name.split('.')[0] + f"_frame_{-j}.jpg"
                new_frame_path = os.path.join(test_frame_path, new_frame_name)
                imageio.imwrite(new_frame_path, new_frame)
        
            


if __name__ == '__main__':
    extract_frames()