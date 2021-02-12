import json
import os
import pandas as pd
import xml.dom.minidom
from xml.dom.minidom import parse
from tqdm import tqdm

label_map = {
    i.split('-')[-1]: int(i.split('-')[1][-3:])
    for i in os.listdir('./dataset/low-annotations/')
}

label_map['background'] = 0

rev_label_map = {v: k for k, v in label_map.items()}

distinct_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080', '#000080',
    '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#e6beff', '#808080', '#FFFFFF'
]
label_color_map = {
    k: distinct_colors[i % len(distinct_colors)]
    for i, k in enumerate(label_map.keys())
}

def parse_annotation(annotation_path):
    
    DOMTree = parse(annotation_path)
    collection = DOMTree.documentElement
    
    folder = str(collection.getElementsByTagName("folder")[0].childNodes[0].data)
#     filename = str(collection.getElementsByTagName("filename")[0].childNodes[0].data)

#     dog_type = str(collection.getElementsByTagName("name")[0].childNodes[0].data)   
#     headbndbox = collection.getElementsByTagName("headbndbox")[0]
#     head_xmin = int(headbndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
#     head_ymin = int(headbndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
#     head_xmax = int(headbndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
#     head_ymax = int(headbndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
    difficult = int(collection.getElementsByTagName("difficult")[0].childNodes[0].data)
    
    bodybndbox = collection.getElementsByTagName("bodybndbox")[0]
    body_xmin = int(bodybndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
    body_ymin = int(bodybndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
    body_xmax = int(bodybndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
    body_ymax = int(bodybndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
    
    data = {
        'boxes': [[body_xmin - 1,body_ymin - 1,body_xmax - 1,body_ymax - 1]],
        'labels':[int(folder.split('-')[1][1:].lstrip('0'))],
        'difficulties':[difficult]
    }

    return data

def create_json_data(Lst_dir,anno_dir):
    train_images = list()
    train_objects = list()

    lst_path = os.path.join(Lst_dir, 'train.lst')
    dt = pd.read_csv(lst_path,header=None)
    
    for item in tqdm(dt[0]):
        img = item.split('/')[-2] + '/' + item.split('/')[-1]
        train_images.append(img)
        anno_path = os.path.join(anno_dir,img) + '.xml'
        train_objects.append(parse_annotation(anno_path))
        
    assert len(train_objects) == len(train_images)
    
    # Save to file
    with open(os.path.join('./dataset/JsonData/', 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join('./dataset/JsonData/', 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join('./dataset/JsonData/', 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too
    
    # Validation data
    test_images = list()
    test_objects = list()
    
    lst_path = os.path.join(Lst_dir, 'validation.lst')
    dt = pd.read_csv(lst_path,header=None)
    
    for item in tqdm(dt[0]):
        img = item.split('/')[-2] + '/' + item.split('/')[-1]
        test_images.append(img)
        
        anno_path = os.path.join(anno_dir,img) + '.xml'
        test_objects.append(parse_annotation(anno_path))
        
    assert len(test_objects) == len(test_images)
    
    # save to file
    with open(os.path.join('./dataset/JsonData/', 'VALID_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join('./dataset/JsonData/', 'VALID_objects.json'), 'w') as j:
        json.dump(test_objects, j)
        
    print('Train data:{},Valid data:{}'.format(len(train_objects),len(test_objects)))
    
    
if __name__=='__main__':
    create_json_data('./dataset/TrainAndValList/','./dataset/low-annotations/')