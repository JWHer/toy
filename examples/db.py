import os, os.path
from tqdm import tqdm
from pymongo import MongoClient

def create(id, path_list):
    # create db with files ###
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    col = db[id]

    meta = {
        "objects": [],
        "stream": "3fc9c921-2562-2252-7128-3dc2ec049d87",
        "app": "846cfd41-ecd4-459e-b21c-d9106f258277",
        "uri": "rtsp://admin:init123!!@192.168.0.58:7001/3fc9c921-2562-2252-7128-3dc2ec049d87",
        "model": "ca1e3eeb-35a1-4164-a36c-651ebaa73850",
        "width": 1920,
        "height": 1080,
        "capture": ""   #path
    }
    for path in path_list:
        meta['capture']=path
        col.insert_one(meta)
        del(meta['_id'])

def migrate(org_id, id=None, query:dict={}, dir=""):
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    org_col = db[org_id]
    if id: col = db[id]
    print(f"Total meta length: {org_col.find().count()}")

    meta_db = org_col.find(query)
    not_exist = []
    if dir and not os.path.exists(dir): os.mkdir(dir)

    for meta in tqdm(meta_db, total=meta_db.count()):
        # org_col.delete_one(meta)
        if os.path.isfile(meta['capture']):
            if dir: meta = change_path(meta, dir)
            if id: col.insert_one(meta)
        else:
            not_exist.append(meta['capture'])
        
    print(f"Wrong meta nums: {len(not_exist)}")
    print(not_exist)


def change_path(meta, dir, name=""):
    """
    meta:dict

    dir:str
        - root directory
    name:str
        - file name with extention
    """
    if not name:
        name = os.path.basename(meta['capture'])

    org_path = meta['capture']
    new_path = os.path.join(dir, name)
    os.replace(org_path, new_path)
    meta['capture'] = new_path

    return meta

def get_path_by_dir(file_path='/host/home/jwher/'):
    # file_path = '/host/home/jwher/phase1/unlabeled'
    data_total = os.listdir(file_path)
    data_total = [ data for data in data_total if data.split('.')[-1]=='png']
    print(f'Total data length: {len(data_total)}')
    return data_total

def get_path_by_db(id, query:dict={}, modify_path_dir=""):
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    col = db[id]

    if query:
        meta_total = list(col.find(query))
    else:
        meta_total = list(col.find())
    print(f"Total meta length: {len(meta_total)}")
    if modify_path_dir:
        meta_path = [ os.path.join(modify_path_dir, os.path.basename(meta['capture'])) for meta in meta_total ]
    else:
        meta_path = [ meta['capture'] for meta in meta_total ]
    return meta_path

def val(id, file_path):
    # list db ###
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    col = db[id]

    # col.rename('3d160a84-5792-4d05-a365-30e2144d8151')
    meta_total = list(col.find())
    print(f"Total meta length: {len(meta_total)}")
    meta_path = [ os.path.basename(meta['capture']) for meta in meta_total ]

    # list dir ###
    data_total = os.listdir(file_path)
    data_total = [ data for data in data_total if data.split('.')[-1]=='png']
    max = 2000
    print(f'Total data length: {len(data_total)}')

    # print(f'Trim to {max}')
    # data_total = data_total[:max]

    # filter ###
    wrong_meta_list = []
    for meta in meta_path:
        if meta not in data_total:
            wrong_meta_list.append(meta)
            # filter = { "capture": meta }
            # col.delete_one(filter)
    print(f'Wrong meta: {len(wrong_meta_list)}')

    wrong_data_list = []
    for data in data_total:
        if data not in meta_path:
            wrong_data_list.append(data)
            # os.remove(os.path.join(file_path, data))
            # os.replace(os.path.join(file_path, data), os.path.join(file_path+'/unlabeled', data))
    print(f'Wrong data: {len(wrong_data_list)}')


def val_with_meta(id, file_path):
    """
    validate with meta and migrate data to file_path
    """
    # list db ###
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    col = db[id]

    # col.rename('3d160a84-5792-4d05-a365-30e2144d8151')
    print(f"Total meta length: {col.find().count()}")

    not_exist = []
    for meta in col.find():
        if os.path.isfile(meta['capture']):
            dest = os.path.join(file_path, os.path.basename(meta['capture']) )
            os.replace(meta['capture'], dest)
            col.update_one(meta, {"$set": {"capture": dest}})
        else:
            not_exist.append(meta['capture'])
    print(f'Lost data: {len(not_exist)}')

    with open(file_path+'lost.txt', 'w') as f:
        f.write(str(not_exist))

def val_with_file(id, dir):
    """
    validate with file path and change capture path
    '/host/home/jwher/label-studio/'
    """

    files = get_path_by_dir(dir)

    # connect db ###
    client = MongoClient('localhost', 27017)
    db = client['analytics']
    col = db[id]

    not_exist = []
    for path in tqdm(files, total=len(files)):
        meta = col.find_one({"capture": {"$regex": path}})
        if meta:
            col.update_one(meta, {"$set": {"capture": os.path.join(dir+path)}})
        else:
            not_exist.append(path)
    print(f'Lost data: {len(not_exist)}')

    with open(dir+'lost.txt', 'w') as f:
        f.write(str(not_exist))

if __name__ == '__main__':
    # org_id = "426b0e83-dce0-48c5-b5a9-f10a6368c692"
    # #id = "ec5075ef-bf60-4511-82d7-c9a2a43eb71e"
    # id = None
    # query = {} #{ "date_captured": {"$lt": 1637661360}}
    # # query = { "$and": [{ "date_captured": {"$gt": 1637628000}} , { "date_captured": {"$lt": 1637630000} }] }
    # dir = "/host/home/jwher/label-studio/"+org_id
    # migrate(org_id, id, query, dir)

    # create()
    # val('bc11134b-f756-4b86-8b22-db7bbb5ab50e', '/host/home/jwher/phase2')
    # val_data('bc11134b-f756-4b86-8b22-db7bbb5ab50e', '/host/home/jwher/phase2')

    val_with_file("426b0e83-dce0-48c5-b5a9-f10a6368c692", '/host/home/jwher/label-studio/426b0e83-dce0-48c5-b5a9-f10a6368c692/')