from matplotlib import pyplot as plt
import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import src.utils.transforms as T
import hydra
from collections import defaultdict
import scipy.stats



class TmpDataset(Dataset):
    #def __init__(self, root: str, annotation: str, targetHeight: int, targetWidth: int, numClass: int):
    def __init__(self, root: str, annotation: str, numClass: int, scaling_thresholds: Dict[str, Tuple[float, float]] = None, removeBackground: bool = True):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.scaling_thresholds = scaling_thresholds
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('RGB')

        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])
        #metadata = self.loadMetadata(imgID)
        metadata = None

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        image, targets = self.transforms(image, targets)
        return image, metadata, targets


    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # convert from [x1, y1, x2, y2] to [x, y, w, h]
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)

    
    def get_weather_info(self, idx: int) -> Dict[str, float]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]
        metadata = imgInfo['meta']
        timestamp = imgInfo['date_captured']
        metadata['Hour'] = int(timestamp.split('T')[1].split(':')[0])
        metadata['Month'] = int(timestamp.split('T')[0].split('-')[1])
        return metadata


    def loadMetadata(self, imgID: int) -> np.ndarray:
        imgInfo = self.coco.imgs[imgID]
        metadata = imgInfo['meta']
        timestamp = imgInfo['date_captured']
        metadata['Hour'] = int(timestamp.split('T')[1].split(':')[0])
        metadata['Month'] = int(timestamp.split('T')[0].split('-')[1])
        metadata = self.scale_harborfront_metadata(metadata)
        metadata = torch.as_tensor(list(metadata.values()))
        return metadata

    def scale_harborfront_metadata(self, metadata: Dict[str, float]) -> Dict[str, float]:
        metadata = metadata.copy()
        for key, (min_val, max_val) in self.scaling_thresholds.items():
            metadata[key] = (metadata[key] - min_val) / (max_val - min_val)
            # shift the range to [-2, 2]
            #metadata[key] = (metadata[key] - 0.5) * 4
        return metadata



@hydra.main(config_path="../../config", config_name="config")
def main(args):
    dataset = TmpDataset(args.dataDir, args.trainAnnFile, args.numClass, removeBackground=args.cropBackground)
    print(dataset.__len__())

    #weather_info = dataset.get_weather_info(0)
    #print(weather_info)

    weather_per_hour = defaultdict(list)
    observations_per_hour = defaultdict(list)

    # create a list containing 12 lists, each representing the number of observations per hour for a specific month
    observations_per_month_and_hour = {i: defaultdict(list) for i in range(1, 13)}


    for i in range(dataset.__len__()):
        weather_info = dataset.get_weather_info(i)
        img, metadata, targets = dataset.__getitem__(i)
        #weather_per_hour[weather_info['Hour']].append(weather_info['Temperature'])
        observations_per_hour[weather_info['Hour']].append(targets['labels'].shape[0])
        observations_per_month_and_hour[weather_info['Month']][weather_info['Hour']].append(targets['labels'].shape[0])
        

    #print(weather_per_hour)
    '''
    # plot the weather distribution over time (hours in a day) with confidence intervals using plt.plot
    hours = list(weather_per_hour.keys())
    temperatures = [np.mean(weather_per_hour[hour]) for hour in hours]
    stds = [np.std(weather_per_hour[hour]) for hour in hours]
    hours, temperatures, stds = zip(*sorted(zip(hours, temperatures, stds)))
    plt.plot(hours, temperatures, 'o-')
    plt.fill_between(hours, np.array(temperatures) - np.array(stds), np.array(temperatures) + np.array(stds), alpha=0.2)
    plt.xlabel('Hour')
    plt.ylabel('Temperature')
    plt.show()
    '''

    font_size = 11
    
    months = list(observations_per_month_and_hour.keys())
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(10.7, 3.8))
    plt.ylim(0, 25)
    #set font size
    plt.rc('font', size=font_size)

    for month in months:
        sample = observations_per_month_and_hour[month]
        if not sample:
            continue
        # sort the dictionary by keys (hours)
        sample = dict(sorted(sample.items()))
        hours = list(sample.keys())
        observations = [np.mean(sample[hour]) for hour in hours]
        stds = [np.std(sample[hour])*.5 for hour in hours] 
        #hours, observations, stds = zip(*sorted(zip(hours, observations, stds)))

        plt.plot(hours, observations, 'o-', label=month_names[month-1], alpha=0.7, markersize=5)
        plt.fill_between(hours, np.array(observations) - np.array(stds), np.array(observations) + np.array(stds), alpha=0.15)


    plt.xlabel('Time of the day', fontsize=font_size)
    plt.ylabel('Number of observations', fontsize=font_size)
    plt.legend(ncol=5, loc='upper left')
    plt.show()

    '''
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 40)
    for month in months:
        sample = observations_per_month_and_hour[month]
        # sort the dictionary by keys (hours)
        sample = dict(sorted(sample.items()))
        hours = list(sample.keys())
        observations = [np.mean(sample[hour]) for hour in hours]
        stds = [np.std(sample[hour])*.5 for hour in hours] 
        
        # upsample and make the data smoother using scipy
        if hours and observations:
            size = 100
            from scipy.interpolate import make_interp_spline
            hours_new = np.linspace(0, 23, size)
            observations_smooth = make_interp_spline(hours, observations, k=3)(hours_new)
            stds_smooth = make_interp_spline(hours, stds, k=3)(hours_new)

            plt.plot(hours_new, observations_smooth, '-', label=month_names[month-1])
            plt.fill_between(hours_new, np.array(observations_smooth) - np.array(stds_smooth), np.array(observations_smooth) + np.array(stds_smooth), alpha=0.15)


    plt.xlabel('Hour')
    plt.ylabel('Number of observations')
    plt.legend(ncol=4)
    plt.show()
    '''

    


if __name__ == "__main__":
    main()