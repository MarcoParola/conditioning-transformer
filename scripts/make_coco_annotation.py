import json
import os
from random import sample
import pandas as pd
import yaml
import datetime
import math
import csv
import argparse
from tqdm import tqdm


def load_datasplits(path):
    with open(path) as f:
        sub_sets = yaml.load(f, Loader=yaml.FullLoader)
    splits = {}
    for s in sub_sets.keys():
        splits[s] = sub_sets[s]
    return splits

def load_annotations(sample_name, img_path, ann_path):
    #Label Storage
    uids = [] 
    labels = []
    boxes = []
    areas = []  
    ocls = []

    class_list = {
    "human"      : 0,
    "bicycle"    : 1,
    "motorcycle" : 2,
    "vehicle"    : 3,}
    
    #Load annotations
    try:

        with open(ann_path, "r") as annotations:
            reader = csv.reader(annotations, delimiter=" ")
            for row in reader:
                if row != ''.join(row).strip(): #ignore empty annotations
                    #Add unique object id                
                    uids.append(int(row[0]))
                    #Add class labels
                    labels.append(int(class_list[row[1]]))
                    #Add occlusion tags
                    #ocls.append(int(row[6]))
                    #Add bounding box
                    #boxes.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])]) #Xtopleft, Ytopleft, Xbottomright, Ybottomright
                    boxes.append([int(row[2]), int(row[3]), abs(int(row[4])-int(row[2])), abs(int(row[5])-int(row[3]))]) #Xtopleft, Ytopleft, Xbottomright, Ybottomright
                    #Add areas
                    areas.append(abs(int(row[4])-int(row[2])) * abs(int(row[5])-int(row[3])))

        #Parse timestamp from samplename
        timestamp =  datetime.datetime.fromisoformat('{}-{}-{} {}:{}'.format(sample_name[:4],  #Year
                                                                            sample_name[4:6],    #Month
                                                                            sample_name[6:8],    #Day
                                                                            sample_name[-9:-7],  #hour
                                                                            sample_name[-7:-5]))  #minute
        #Save target dict
        targets = {
            "uids"       : uids,
            "boxes"     : boxes,
            "labels"    : labels,
            #"occlusions": ocls,
            "iscrowd"   : ocls, #iscrowd is used as an occlusion tag
            "areas"      : areas,
            "timestamp" : timestamp + datetime.timedelta(0, int(sample_name[-4:]))
            #sample_name example: '20200514_clip_0_1331_0001'
            #because framerate is 1fps we use framenumber as second
        }
        print(f'Loaded {sample_name}')
        return targets
    except:
        print(f'Error in {sample_name}')
        return None

def generate_unique_image_id(dateobj,clipname):
    return int( #Using Zfill to zero pad every number 
        f'{dateobj.year}'.zfill(4)+
        f'{dateobj.month}'.zfill(2)+
        f'{clipname.split("_")[1]}'.zfill(3)+
        f'{dateobj.day}'.zfill(2)+
        f'{dateobj.hour}'.zfill(2)+
        f'{dateobj.minute}'.zfill(2)+
        f'{dateobj.second}'.zfill(2))

# COCO META DATA REQUIRED FOR ANNOTATIONS
ANNO_INFO = {
                "description": "Harborfront",
                "version": "1.0",
                "year": 2024,
                "contributor": "Milestone Research Programme at Aalborg University, Visual Analysis and Perception Lab at Aalborg University",
                "date_created": "23/01/2024"
            }

ANNO_LICENSES = [
"COPYRIGHTED IMAGE - PLACEHOLDER LICENSE"
] 

ANNO_CATEGORIES = [
        {"supercategory": "background","id": 0,"name": "background"},
        {"supercategory": "person","id": 1,"name": "person"},
        {"supercategory": "bicycle","id": 2,"name": "bicycle"},
        {"supercategory": "motorcycle","id": 3,"name": "motorcycle"},
        {"supercategory": "vehicle","id": 4,"name": "vehicle"},
] 

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("Generate COCO annotations from Harborfront txt")
    
    #REQUIRED
    parser.add_argument('root', help="Path to the Harborfront Root Directory")
    
    #OPTIONAL
    parser.add_argument('--output', default="./", help="Output folder for the produced JSON annotation file")
    parser.add_argument('--img_folder', default="frames/", help="Relative path from root to image directory")
    parser.add_argument('--ann_folder', default="annotations/", help="Relative path from root to annotation directory")
    parser.add_argument('--meta_file', default="metadata.csv", help="Path to the metadata.csv from the original LTD Dataset, [None] to exclude meta data")
    parser.add_argument('--datasplit_yaml', default=None, help="Optional Dataset split file to generate several JSON annotation files. [None] will result in a complete JSON annotation file")
    parser.add_argument('--min_objects', default=1, type=int, help="Set lower object limit for sample inclusion, [<1] will result in inclusion of frames without objects")
    parser.add_argument('--meta_hist', default=0, type=int, help="Hours of metadata history to store in image annotation")
    args = parser.parse_args()

    #Load splits
    if args.datasplit_yaml is not None:
        data = load_datasplits(args.datasplit_yaml)
    else:
        tmp = {}
        for date in os.listdir(os.path.join(args.root, args.ann_folder)):
            tmp[date] = []
            for clip in os.listdir(os.path.join(args.root, args.ann_folder, date)):
                tmp[date].append(clip) 
        data = {"Annotations" : tmp}

    #Load meta Data
    if args.meta_file is not None:
        meta_df = pd.read_csv(args.meta_file)

        #Convert datetime string to datetime object
        meta_df["DateTime"] = meta_df["DateTime"].apply(pd.to_datetime)
    
    print(meta_df)

    #Frame Counter
    frame_idx = 1
    annot_idx = 1

    #Iterate over annotations
    for split in data.keys():
        print(f'Processing Datasplit: {split}')
        output_anno_file = os.path.join(args.output, split+".json")
        ANNO_INFO["partition"] = "{}-set".format(split)

        #Initialize Annotation file
        annotation_file = {
            "info": ANNO_INFO,
            "licenses": ANNO_LICENSES,
            "images": [],
            "annotations": [],
            "categories": ANNO_CATEGORIES,
        }

        #Iterate over data
        for date in tqdm(data[split].keys(), "Processsing Date"):
            clips = data[split][date]
            if split == "Train":
                clips = sample(clips, math.ceil(len(clips)*0.45))

            for clip in clips:
                if args.meta_file is not None:

                    #Get Meta Data
                    img_meta_data = meta_df.loc[(meta_df['Clip Name'] == clip) & (meta_df['Folder name'] == int(date))].to_dict("records")[0]
                    
                    his_meta_data = None
                    if args.meta_hist > 0:
                        mask = (meta_df['DateTime'] > img_meta_data["DateTime"]-datetime.timedelta(hours=args.meta_hist, minutes=5)) & (meta_df['DateTime'] <= img_meta_data["DateTime"])
                        his_meta_data = meta_df.loc[mask].to_dict("records")
                    

                    #Remove unwanted meta entries
                    for ex in ['Folder name', 'Clip Name', 'DateTime']:
                        del img_meta_data[ex]
                    
                    #Clean up historical data, if parsed
                    if args.meta_hist > 0:
                        for e in his_meta_data: #iterate through every historical sample
                            #Convert datetime object to string
                            e["DateTime"] = e["DateTime"].isoformat()

                            #Remove unwanted historical data
                            for ex in ['Folder name', 'Clip Name']:
                                del e[ex]
                
                #Iterate over frames in current clip
                for frame in os.listdir(os.path.join(args.root, args.img_folder, date, clip)):
                    
                    #Get frame number
                    frame_number = frame.rsplit('.', 1)[0].rsplit('_')[1]
                    
                    #Make file_paths
                    img_path = os.path.join(args.root, args.img_folder, date, clip, "image_{}.jpg".format(frame_number))
                    ann_path = os.path.join(args.root, args.ann_folder, date, clip, "annotations_{}.txt".format(frame_number))
                    
                    #Load annotations
                    sample_name = "{}_{}_{}".format(date, clip, frame_number)
                    annots = load_annotations(sample_name, img_path, ann_path)

                    if annots is None:
                        continue

                    #Generate frame_id from timestamp

                    #Create image and annotation entries
                    if len(annots["labels"]) >= args.min_objects:
                        img_entry = {
                            "license": 0,
                            "file_name": os.path.join(args.img_folder, date, clip, "image_{}.jpg".format(frame_number)),
                            "coco_url": "N/A",
                            "height": 288,
                            "width": 384,
                            "date_captured": annots["timestamp"].isoformat(),
                            "flickr_url": "N/A",
                            #"id": frame_idx,
                            "id": generate_unique_image_id(annots["timestamp"], clip)
                        }

                        #Append metadata
                        if args.meta_file is not None:
                            img_entry["meta"] = img_meta_data
                        if args.meta_hist > 0:
                            img_entry["meta_hist"] = his_meta_data

                        #Append annotation
                        annotation_file["images"].append(img_entry)
 
                        for i in range(len(annots["labels"])):
                            
                            #Write annotation info dict
                            ann_entry = {
                            "area": int(annots["areas"][i]),
                            #"iscrowd": int(annots["iscrowd"][i]),
                            "image_id": int(img_entry["id"]),
                            "bbox": annots["boxes"][i],
                            "category_id": int(annots["labels"][i]+1),
                            "id": annot_idx,
                            "uid": int(annots["uids"][i])
                            }
                            annotation_file["annotations"].append(ann_entry)
                            
                            #Increment annotation counter
                            annot_idx += 1

                    #Increment framecounter
                    #frame_idx += 1

        #Save Json file
        print(f'Saving annotationfile for {split} ({output_anno_file})')
        with open(output_anno_file, "w") as out_file:
            json.dump(annotation_file, out_file, indent=2)