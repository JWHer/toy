import os
import json
from unicodedata import name
from tqdm import tqdm
from pymongo import MongoClient


class Connection:
    def __init__(self, mongo_url):
        client = MongoClient(mongo_url)
        self.col = client["analytics"]

    def db_to_json(self, file_name, classes, capture_id, resize_factor=(1, 1)):
        collection = self.col[capture_id]
        documents = collection.find()

        print(f'Start to make COCO format json from DB... ({documents.count()})')

        json_dict = {
            "categories": [],
            "images": [],
            "annotations": []
        }

        for idx, name in enumerate(classes):
            json_dict["categories"].append({
                "id": idx,
                "name": name
            })

        image_id = 1
        annotation_id = 1
        phase_name = ""

        for data in tqdm(documents, total=documents.count()):
            image_path = data["capture"]
            width = data["width"] * resize_factor[0]
            height = data["height"] * resize_factor[1]
            date_captured = data["date_captured"] if 'date_captured' in data else ""
            json_dict["images"].append({
                "id": image_id,
                "file_name": os.path.join(phase_name, image_path),
                "width": width,
                "height": height,
                "date_captured": date_captured
            })

            results = data['objects']
            for result in results:
                xmin = result['bbox']['x'] * width
                ymin = result['bbox']['y'] * height
                obj_width = result['bbox']['width'] * width
                obj_height = result['bbox']['height'] * height
                score = float("{:.3f}".format(float(result['score'])))
                trackid = result['track_id']
                category = result['type_id']
                category = classes.index(category) if type(category) == str else result['type_id']
                json_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category,
                    "iscrowd": 0,
                    "attribute": 0,
                    "bbox": [int(xmin), int(ymin), int(obj_width), int(obj_height)],
                    "area": int(obj_width * obj_height),
                    "keypoints": [],
                    "score": score,
                    "trackid": trackid
                })
                annotation_id += 1
            image_id += 1

        with open(file_name, 'w') as f:
            json.dump(json_dict, f, indent=4)

        return file_name

if __name__ == '__main__':
    db = Connection('localhost:27017')
    id="a933db9e-1397-4152-b9b9-a404b1ef1545"
    output_name = db.db_to_json(file_name=id+'.json', classes=['wheelchair', 'blind', 'stroller', 'person'], capture_id=id)
    print(output_name)