"""
uses a local `detectserver` instance to separate images 
"""

import requests
import os 
import json
import cv2 
import base64
import numpy as np
import argparse
import pathlib
from tqdm import tqdm
import shutil


def make_request(image_path: str):
    try:
        select_image = cv2.imread(os.path.join(image_path))
        _, im_arr = cv2.imencode(".jpeg", select_image)
        img_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(img_bytes)
        myobj = {"data":f"{im_b64.decode('ascii')}", "detect_type":"left2right"}
        resp = requests.post(
                ADDRESS,
                json=myobj
                )
        _resp = resp.json()
        extension = os.path.splitext(image_path)[1] 
        fname = pathlib.Path(image_path).stem + extension
        # print(_resp)
        # print(len(_resp['detections'])) 
        if len(_resp['detections']) != 0:
            shutil.move(image_path, os.path.join(FRONT_DIR, fname))
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":

    ADDRESS = "http://192.168.89.243:5005/cardDetectPost"
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, help="processing folder")
    opt = args.parse_args()
    
    BACK_DIR = os.path.join(opt.folder, "back") 
    FRONT_DIR = os.path.join(opt.folder, "front")
    if not os.path.exists(BACK_DIR):
        os.makedirs(BACK_DIR)
    if not os.path.exists(FRONT_DIR):
        os.makedirs(FRONT_DIR)
    
    all_images = [os.path.join(opt.folder, x) for x in os.listdir(opt.folder)]
    # print(all_images)
    [make_request(x) for x in tqdm(all_images)]
