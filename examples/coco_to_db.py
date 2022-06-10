from typing import Tuple
import random

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as image

def randomCatIds(coco:COCO) -> Tuple[list, list]:
    catIds = coco.getCatIds()
    sample_num = random.randint(1,len(catIds))

    print(f'Get [{sample_num}] Random Sample')
    randomCatIds = random.sample(catIds, sample_num)
    return randomCatIds

def getCats(coco:COCO, catIds:list) -> list:
    cats = coco.loadCats(catIds)
    print('Categories')
    print(' '.join([cat['name'] for cat in cats]))
    return cats

def getAllImgsWithCatIds(coco:COCO, catIds:list) -> Tuple[list, list]:
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = [coco.loadImgs(imgId)[0] for imgId in imgIds ]
    
    return imgIds, imgs

def showImage(img:dict, catIds:list):
    data_dir = '/host/home/jwher/dev/mount/bucket/data/val2017/'
    _image = image.imread(data_dir + img['file_name'])
    plt.axis('off')
    plt.imshow(_image)

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, True)
    plt.show()

def convert(img:dict, ann:dict, classes:list):
    return {
        "objects": [
            {
            "type_id": classes[ann['category_id']],
            "score": 1.0,
            "bbox": ann['bbox'],
            "segment": ann['segment']
            }
        ],
        "width": img['width'],
        "height": img['height'],
        "capture": img['file_name'],
        "img_id": img['id'],
        "date_captured": img['date_captured']
    }

if __name__ == '__main__':
    data_dir = '/host/home/jwher/dev/mount/bucket/data/val2017/'
    ann_name = data_dir+'instances_val2017.json'
    coco = COCO(ann_name)

    # catIds = randomCatIds(coco)
    catIds = [1,2,3]
    cats = getCats(coco, catIds)

    _, imgs = getAllImgsWithCatIds(coco, catIds)
    # imgIds = [ img['id'] for img in imgs]

    for idx, img in enumerate(imgs):
        showImage(img, catIds)
    # annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds)
    # anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)