import os
import json
import urllib.request
from tqdm import tqdm
from pymongo import MongoClient

class DatasetScheme:
    objects:str = "objects"
    score:str = "score"
    track_id:str = "track_id"
    type_id:str = "type_id"

    bbox:str = "bbox"
    segment:str = "segment"

    file_name:str = "file_name"
    width: str = "width"
    height: str = "height"
    date_captured: str = "date_captured"
    img_id: str = "img_id"

    stream= "stream"
    app= "app"
    model= "model"
    uri= "uri"

class DatasetSchemeV0(DatasetScheme):
    file_name = "capture"

class DatasetLoader:
    def __init__(self):
        self.scheme = DatasetScheme

    def load(
        self, project_id:str, dataset_id:str, classes:list,
        mongo_uri="mongodb://mongo:27017",
        object_storage_uri="http://tx-server/api/v1/storage",
        save_dir=None,
        resize_factor=(1, 1)
    ):
        if not save_dir:
            # /image001.jpg
            # /...
            # /annotation.json
            save_dir=f"{project_id}/{dataset_id}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # open db session
        client = MongoClient(mongo_uri)
        database = client[project_id]
        collection = database[dataset_id]
        documents =collection.find()

        print('Start making COCO format json from DB..')
	
        coco = {
            "categories": [ {"id":idx, "name":_class } for idx, _class in enumerate(classes)],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        for idx, doc in tqdm(enumerate(documents, start=1), total=documents.count()):
            image_path = doc[self.scheme.file_name]

            # In most case, It runs in container.

            # First, check original path
            if not os.path.isfile(image_path):
                base_name = os.path.basename(image_path)
                image_path = os.path.normpath(f"{save_dir}/{base_name}")

            # Second, check local path
            if not os.path.isfile(image_path):
                # Download
                urllib.request.urlretrieve(
                    f"{object_storage_uri}/{project_id}/{dataset_id}/{base_name}",
                    image_path
                    )

            if not os.path.isfile(image_path):
                raise ValueError(f'Image not found: {image_path}')

            width = doc[self.scheme.width] * resize_factor[0]
            height = doc[self.scheme.height] * resize_factor[1]

            date_captured = "" if not self.scheme.date_captured in doc.keys()\
                else doc[self.scheme.date_captured]

            coco["images"].append({
                "id": idx,
                "file_name": image_path,
                "width": width,
                "height": height,
                "date_captured": date_captured
            })

            results = doc[self.scheme.objects]
            for result in results:
                # bbox
                xmin = result[self.scheme.bbox]['x']
                ymin = result[self.scheme.bbox]['y']
                obj_width = result[self.scheme.bbox][self.scheme.width]
                obj_height = result[self.scheme.bbox][self.scheme.height]

                # segment
                segmentation = [] if not self.scheme.segment in result else result[self.scheme.segment]

                score = float("{:.3f}".format(float(result[self.scheme.score])))
                trackid = "" if not self.scheme.track_id in result else result[self.scheme.track_id]
                category = result[self.scheme.type_id]
                if type(category) == str: category = classes.index(category)

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": category,
                    "iscrowd": 0,
                    "attribute": 0, #?
                    "bbox": [int(xmin), int(ymin), int(obj_width), int(obj_height)],
                    "area": int(obj_width * obj_height),
                    # "keypoints": [],
                    "segmentation": segmentation,
                    "score": score,
                    "trackid": trackid
                })
                annotation_id += 1

        coco_json = os.path.normpath(f"{save_dir}/annotation.json")
        with open(coco_json, 'w') as f:
            json.dump(coco, f, indent=4)

        return coco_json