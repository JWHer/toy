import os, re, json, uuid

# constants
root_dir:str = 'trt'
model_regex = re.compile(r'model-?.*.json$')
endpoint = 'rnd.snuailab.ai:15130'
output = '01_models.json'

cur_path:str = os.getcwd()
root_pos:int = cur_path.find(root_dir) + len(root_dir)
model_path: str = cur_path[:root_pos]+'/model/models'
models = os.listdir(model_path)

support_models:list = [model for model in models if '.json' not in model]

rows:list = []
stack:list = [model_path+'/'+model for model in support_models]
while stack:
    cur_name = stack.pop()
    if model_regex.search(cur_name) is not None:
        with open(cur_name, 'r') as model_file:
            model_dict = json.load(model_file)

        rel_path = os.path.dirname(cur_name)[root_pos+len('/model/models')+1:]
        name = rel_path.replace('/', ' ')
        name = name.replace('_', ' ')
        id:uuid.UUID = uuid.uuid3(uuid.NAMESPACE_DNS, name)
        rows.append({
            "id": str(id),
            "production_id": str(id),
            "name": name,
            "classes": model_dict['classes'],
            "desc": name.title(),
            "tags": ["base", name.split(' ')[0].lower()] if not 'ssl' in cur_name else ["base", name.split(' ')[0].lower(), "ssl"],
            "location": f"file:///opt/model/models/{rel_path}",
            "status": "ALIVE",
            "capacity": 1,
            "version": f"{name.replace(' ', '_').lower()}-{str(id)[-8:]}_compute86",
            "platform": "dgpu",
            "framework": "deepstream==6.0",
            "precision": "FP16"
        })

    elif os.path.isdir(cur_name):
        subdirs = os.listdir(cur_name)
        stack.extend([cur_name+'/'+subdir for subdir in subdirs])

result:dict = {
    "table": "model",
    "rows": rows
}
with open(output, 'w') as output:
    json.dump(result, output, indent=2)
