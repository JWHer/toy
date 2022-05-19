import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

class Viewer:
    def __init__(self, ann_name:str=None) -> None:
        self._ann_name = ann_name
        self.dir_name = os.path.dirname(ann_name)
        self._coco = None

    @property
    def ann_name(self)->str:
        return self._ann_name

    @ann_name.setter
    def ann_name(self, value:str):
        if self._ann_name is not None and self._ann_name != value:
            self._coco = None
        self.dir_name = os.path.dirname(value)
        self._ann_name = value

    @property
    def coco(self):
        if self._coco is None:
            self._coco = COCO(self.ann_name)
        return self._coco

    @property
    def images(self) -> list:
        return list(self.coco.imgs.values())

    def get_anns(self, imgIds:list=[]):
        ann_ids = self.coco.getAnnIds(imgIds)
        return self.coco.loadAnns(ann_ids)

    def show(self, image_id:str=None, is_show=True, is_save=False):
        if image_id is None:
            image_ids = self.coco.getImgIds()
        else:
            image_ids = [image_id]
        for image_id in image_ids:
            self._show(image_id)
            if is_show: plt.show()
            if is_save: plt.savefig(f"{image_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)

    def _show(self, image_id:str):
        image = self.coco.imgs[image_id]
        anns = self.get_anns([image_id])

        im = Image.open(
            os.path.join(self.dir_name, image['file_name'])
        )
        plt.axis('off')
        plt.imshow(np.asarray(im))
        self.coco.showAnns(anns, draw_bbox=True)

if __name__ == '__main__':
    print("COCO Viewer")
    ann_name = input("Annotation path: ")
    if not ann_name: ann_name='./coco_sample/annotation.json'
    v = Viewer(ann_name)

    while True:
        select = input('Select Input (i:image_ids) (s:show) (q:quit): ')
        if select=='i':
            print(v.coco.getImgIds())
        elif select=='s':
            image_id = input('Insert image id: ')
            if not image_id: image_id = None
            v.show(image_id)
        elif select=='q':
            break
        else:
            print(f"You've entered [{select}] (q to quit)")
    print("Thanks!")
