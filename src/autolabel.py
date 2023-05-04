from models.common import DetectMultiBackend
from detect import imageDetect
from utils.torch_utils import select_device
import cv2
from pathlib import Path
import os 
import configparser
import json
import base64
#read global config file
config = configparser.ConfigParser()
config.read('auto-labelcfg.ini')


MODEL_PATH = os.path.join("detect-models",config["MODEL"]["MODEL_NAME"])
CUDA_DEVICE = select_device(str(config["DEVICE_SETTINGS"]["CUDA_DEVICE"]))
CONFIDENCE_THRESH = float(config["INFERENCE_CONFIG"]["CONFIDENCE"])
loadmodel = DetectMultiBackend(MODEL_PATH,CUDA_DEVICE , dnn=False ,data=None ,fp16=False)
ext = [".jpeg", ".jpg", ".png"]
# imgpath = Path("test/test.jpeg")
# print(os.path.basename(imgpath))
# det = imageDetect(imgpath, "0", loadmodel, 0.8)
# det.detect()

class AutoLabel:
    def __init__(self, processingFolder, confidence = 0.9, iou = 0.7):
        self.processingFolder = processingFolder
        self.confidence = confidence
        self.iou = iou

    def findAllImageInFolder(self):
        imagePaths = []
        for roots,dirs,files in os.walk(self.processingFolder):
            for file in files:
                if file.endswith(tuple(ext)):
                    fullimagepath = os.path.join(self.processingFolder,file)                
                    imagePaths.append(fullimagepath)
            return imagePaths
    

    def Label(self):
        images = self.findAllImageInFolder()
        for image in images:
            detectLabels = imageDetect(image,CUDA_DEVICE,loadmodel,CONFIDENCE_THRESH)
            result = detectLabels.detect()
            
            json_obj = json.dumps(result, indent=2)
             
            fileName = os.path.splitext(os.path.basename(image))[0] + ".json"
            jsonFileExportName = os.path.join(self.processingFolder, fileName)
            
            
            #export json with same name as image file name


            with open(jsonFileExportName, "w") as json_output:
                json_output.write(json_obj)

test = AutoLabel("test")
test.Label()
