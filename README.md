# auto-label
simple autolabeling tool , uses yolov7x model to auto label images in labelme format

## usage

### install requirements

`pip install -r requirements.txt`

### configure

create a folder called `detect-models` in `src` and place your models there
<br>
in `src/auto-labelcfg.ini`
```ini
[DEVICE_SETTINGS]
CUDA_DEVICE = 0 

[MODEL]
MODEL_NAME = yourmodelname.pt

[INFERENCE_CONFIG]
CONFIDENCE = 0.7
IOU = 0.7

```

### label

`cd src`
<br>
`python3 main.py --source path/to/image/folder`

