import json
# import os.path as osp
from tqdm import tqdm

from random import randint

def dict2xml(d, root_node=None):
	wrap          =     False if None == root_node or isinstance(d, list) else True
	root          = 'objects' if None == root_node else root_node
	root_singular = root[:-1] if 's' == root[-1] and None == root_node else root
	xml           = ''
	children      = []

	if isinstance(d, dict):
		for key, value in dict.items(d):
			if isinstance(value, dict):
				children.append(dict2xml(value, key))
			elif isinstance(value, list):
				children.append(dict2xml(value, key))
			else:
				xml = xml + ' ' + key + '="' + str(value) + '"'
	else:
		for value in d:
			children.append(dict2xml(value, root_singular))

	end_tag = '>' if 0 < len(children) else '/>'

	if wrap or isinstance(d, dict):
		xml = '<' + root + xml + end_tag

	if 0 < len(children):
		for child in children:
			xml = xml + child

		if wrap or isinstance(d, dict):
			xml = xml + '</' + root + '>'
		
	return xml

if __name__ == '__main__':

    id = 'a933db9e-1397-4152-b9b9-a404b1ef1545'
    # input source
    coco_json = open(id+'.json')
    coco_dict= json.load(coco_json)

    data_dir = '' #'/label-studio/data' # path inside data storage root.
    ls_storage_path = '/data/local-files/?d=' # example path when LS set local storage root.



    # convert annotation format
    images = coco_dict['images']
    annotations = coco_dict['annotations']
    classes = coco_dict['categories']
    converted_list = [{'id': img['id'], 
                        'annotations': [{'id': img['id'],  # unique annotation id
                                        'result': [],
                        }],
                        'data': {}
                        }for img in images] 
    for ann in tqdm(annotations, total=len(annotations)):
        # 이유는 모르지만 img_id는 1부터 시작함
        img_id = ann['image_id']-1

        cls = classes[ann['category_id']]['name']
        image_name= images[img_id]['file_name']
        image_name = image_name.replace('/host', data_dir)
        image_path = ls_storage_path + image_name
        height, width = images[img_id]['height'], images[img_id]['width']
        bbox = ann['bbox']
        x = bbox[0] / width * 100
        y = bbox[1] / height* 100
        ls_width = bbox[2] / width * 100
        ls_height = bbox[3] / height* 100    
        res = {"original_width":width,
            "original_height":height, 
            "image_rotation": 0,
            "value":{
                    "x":x,
                    "y":y,
                    "width":ls_width,
                    "height":ls_height,
                    "rotation":0,
                    "rectanglelabels": [cls]
                },
            "id":ann['id'], # unique bbox id
            "from_name":"label",
            "to_name":"image",
            "type":"rectanglelabels"}
        for c in converted_list:
            if c['id'] == img_id:
                c['annotations'][0]['result'].append(res)
                c['data'] = {'image': image_path}
            
    # output annotation json
    with open(id+'_converted.json', 'w') as f:
        json.dump(converted_list, f, indent=4)
        f.close()
    # create label config xml file
    config_dict = {
        "Image":{"name":"image", "value":"$image"},
        "RectangleLabels": {"name": "label", "toName": "image"}
    }
    label = [] 
    for cls in classes:    
        cls_name = cls['name']
        color = '#%06X' % randint(0, 0xFFFFFF)
        label.append({"value": cls_name,"background":color})
    config_dict["RectangleLabels"].update({'Label': label})
    xml_str = dict2xml(config_dict, 'View')
    with open(id+'_config.xml', 'w') as f:
        f.write(xml_str)
        f.close()

    print(id+" convert done sucessfully")