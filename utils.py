import json
import os
import torch
import random
import xml.etree.ElementTree as ET 
import torchvision.transforms.functional as FT 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ('aeroplane',' bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k: v+1 for v,k in enumerate(voc_labels)}
'''
{'sheep': 17, 'horse': 13, 'aeroplane': 1, 'cow': 10, 'sofa': 18,
 'bus': 6,' bicycle': 2, 'dog': 12, 'cat': 8,'person': 15,
 'train': 19,'diningtable': 11, 'bottle': 5, 'car': 7,'pottedplant': 16,
 'tvmonitor': 20, 'chair': 9, 'bird': 3, 'boat': 4, 'motorbike': 14}
 '''
label_map['background'] = 0
rev_label_map = {v:k for k,v in label_map.items()}

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

label_color_map = {k : distinct_colors[i] for i,k in enumerate(label_map.keys())}

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)-1
        ymin = int(bbox.find('ymin').text)-1
        xmax = int(bbox.find('xmax').text)-1
        ymax = int(bbox.find('ymax').text)-1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map(label))
        difficulties.append(difficult)

    return {'boxes':boxes, 'labels':labels, 'difficulties':difficulties}

def create_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    for path in [voc07_path, voc12_path]:
        with open(os.path.join(path,'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()
        
        for id in ids:
            objects = parse_annotation(os.)