import os
from detect_white import is_almost_white_page,is_almost_single_color
from faiss_compare import return_index
from faiss_vector_gen import generate_vectors
import faiss
import subprocess
import time
from ocr import detect_text
import json

def remove_white_images(fpath, fname):
    images = []
    for root, dirs, files in os.walk(fpath):
        for file in files:
            print(file)
            if file.endswith('png'):
                if not is_almost_single_color(root  + '/'+ file):
                    images.append(root  + '/'+ file)
    json.dump(images, open(f'no_white_{fname}.json', 'w'))
    return images

def extract_text(images, extn):
    img_to_txt = {}
    no_txt_images = []
    for image in images:
        txt = detect_text(image)
        print(txt)
        if txt == '':
            no_txt_images.append(image)
            continue
        img_to_txt[image] = txt
    json.dump(no_txt_images, open(f'no_txt_images_{extn}.json', 'w'))
    json.dump(img_to_txt, open(f'ocr_{extn}.json', 'w'))
    
    return list(img_to_txt.keys())

def remove_duplicates(images, vector_index, extn):
    # print(len(images), images)
    interesting_index = []
    vector_index = faiss.read_index(vector_index)

    for image in range(len(images)):
        try:
            d_ind, i_ind = return_index(images[image], vector_index)
            distance = list(d_ind[0])
            index = list(i_ind[0])
        
            for d in range(1, len(distance)):
                # print(f'distance[{d}]', distance[d])
                if distance[d] < 0.001:
                    if index[d] not in interesting_index and index[d] > image:
                        interesting_index.append(index[d])
        except Exception as e:
            print(e)
            continue

    if len(interesting_index) > 1:
        interesting_index.sort(reverse=True)
        print('interesting_index', len(interesting_index), interesting_index)
        for i in interesting_index:
            try:
                del images[i]
            except Exception as e:
                print(e, i)
                continue
    
    json.dump(images, open(f'images_dedup_{extn}.json', 'w'))

adb_path = f'./../sample_data/adblock'
control_path = f'./../sample_data/control'

control_index = f'control.index'
adb_index = f'adblock.index'

# remove whitespaces
control_images = remove_white_images(control_path, f'control')
adb_images = remove_white_images(adb_path, f'adblock')
print('White Spaces Removed')

# OCR
# control_images = extract_text(control_images, f'control')
# time.sleep(5)
# adb_images = extract_text(adb_images, f'adblock')
# time.sleep(5)
# print('Text Extracted')

# generate vectors
generate_vectors(control_images, f'control')
time.sleep(5)
generate_vectors(adb_images, f'adblock')
time.sleep(5)
print('Vectors Generated')

# deduplicates
remove_duplicates(control_images, control_index, f'control')
time.sleep(5)
remove_duplicates(adb_images, adb_index, f'adblock')
time.sleep(5)
print('Duplicates Removed')