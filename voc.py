import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

def parse_voc_annotation(ann_file):

    all_insts = []

    try:
        tree = ET.parse(ann_file)
    except Exception as e:
        print(e)
        print('Ignore this bad annotation: ' + ann_file)
    
    for elem in tree.iter():
        if 'FOD' in elem.tag or 'part' in elem.tag:
            obj = {}
            
            for attr in list(elem):
                if 'type' in attr.tag:
                    obj['type'] = attr.text
                        
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

            all_insts += [obj]
    return all_insts
