import torch, os
def inference(img_path:str):

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5n yolov5s - yolov5x6, custom

    # Images
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    results = model(img_path)

    # Results
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

# inference('./data/COCO/2017/000000000139.jpg')
img_list=os.listdir('./data/COCO/2017')[:1]
img_list=[os.path.join('./data/COCO/2017', img) for img in img_list]
inference(img_list)
