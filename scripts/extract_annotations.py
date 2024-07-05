
## PLEASE FIRST UNZIP THE HITL ZIP FILES INTO ONE FOLDER

import os
import re
from lxml import etree
import zipfile
import json
import collections

annotation_folder = 'data/Raw HITL'
output_folder = './data/outputs/'
anno_meta_data_path = './annotation_meta.json'
verbose = True

if os.path.isfile(anno_meta_data_path):
    with open(anno_meta_data_path) as anmet:
        anno_meta_dict = json.load(anmet)
else:
    anno_meta_dict = {
        'unique_tracks': 0,
        'track_id_length': 6,
    }

#Unique Track ID
overwrite_ID = True #ONLY change this if you do NOT want unique ID's

#Iterate through the unzipped HITL folders
for f in os.listdir(annotation_folder):
    if os.path.isdir(os.path.join(annotation_folder, f)):
        #Print progress
        if verbose == True:
            print("--------- Date {} ---------".format(f[:8])) 
        
        #List Zips in folder
        for dir in os.listdir(os.path.join(annotation_folder, f)):
            print(dir)
            if verbose == True:
                print("- Annotation Zip {}".format(dir[11:-4])) 

            print(os.path.join(annotation_folder, f, dir))

            for z in os.listdir(os.path.join(annotation_folder, f, dir)):
                #Check if file is a zip


                if zipfile.is_zipfile(os.path.join(annotation_folder, f, dir, z)):
                    #Format output names
                    temp = re.search(r"\d","___2912312").start()
                    name_date = z[re.search(r"\d",z).start():(re.search(r"\d",z).start()+8)] #Get date (find the first digit and assume thats the date)
                    name_clip = z[z.find('c'):-4] #Get clip
                    name_clip = re.sub("[\(\[].*?[\)\]]", "", name_clip) #Remove Dublicate indicator "Clip_1_3_3(1).zip" - > "Clip_1_3_3"
                    
                    try: #Some Annotation files are broken, so we try to write them, except those we cant, and create a list of the ones we have to manually look at
                        #Extract annotations
                        with zipfile.ZipFile(os.path.join(annotation_folder, f, dir, z)) as anno_zip:
                            with anno_zip.open('annotations.xml') as anno_file:

                                #Use cache to batch up annotations so we write in buld (much faster IO)
                                annotation_cache = collections.defaultdict(list)

                                #Build XML Tree
                                tree = etree.parse(anno_file)
                                
                                #find all tracklets
                                tracks = tree.findall("track")
                                
                                #Write Track to annotation file
                                for track in tracks:
                                    #Track Data
                                    track_class = track.attrib["label"].lower()

                                    #Overwrite annoation ID with Unique ID
                                    if overwrite_ID:
                                        track_id = '{}'.format(anno_meta_dict['unique_tracks']).zfill(anno_meta_dict['track_id_length'])
                                        anno_meta_dict['unique_tracks'] = anno_meta_dict['unique_tracks'] + 1  
                                    else:
                                        track_id = '{}'.format(int(track.attrib['id'])).zfill(anno_meta_dict['track_id_length'])
                                    
                                    #Object Data    
                                    for node in track:
                                        #Bounding Box coordinates (rounded to nearest int)
                                        x_top_left = int(float(node.attrib['xtl']))
                                        y_top_left = int(float(node.attrib['ytl']))
                                        x_bot_right = int(float(node.attrib['xbr']))
                                        y_bot_right = int(float(node.attrib['ybr']))
                                        
                                        #Object Tags
                                        keyframe = node.attrib['keyframe']
                                        occluded = node.attrib['occluded']
                                        outside = node.attrib['outside']
                                        
                                        #Frame Number
                                        frame_num = int(float(node.attrib['frame']))

                                        #Cache node
                                        if outside == '0':
                                            annotation_cache['{0:04d}'.format(frame_num)].append('{} {} {} {} {} {}\n'.format(track_id, track_class, x_top_left, y_top_left, x_bot_right, y_bot_right))

                        #Define output file and path
                        out_path = os.path.join(output_folder, name_date, name_clip)
                        os.makedirs(out_path, exist_ok=True)
        
                        #iterate over cached annotations
                        for k, v in annotation_cache.items():
                            #Define annotation name
                            out_anno_file_name = os.path.join(out_path, 'annotations_{}.txt'.format(k))
                            #write cached lines
                            with open(out_anno_file_name, 'a+') as anno_out:
                                anno_out.writelines(v)

                        #Update meta file
                        with open(anno_meta_data_path, 'w+') as anmet:
                                json.dump(anno_meta_dict, anmet)
                    except Exception as ex:
                        print("Error with annotation file ({} - {})\n".format(name_date, name_clip))
                        with open('./errors.txt', 'a+') as elog:
                                elog.write("{} - {}: {} {}\n".format(name_date, name_clip, type(ex).__name__, ex.args))
                        
                    



