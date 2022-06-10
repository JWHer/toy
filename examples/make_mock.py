import json
from faker import Faker
fake = Faker()

mx = {
    "id": fake.uuid4(),
    "name": fake.name(),
    "uri": fake.uri(),
    "user_id": fake.name(),
    "user_pw": fake.password()
}

with open("mx.json", 'w') as file:
    json.dump(mx, file)

stream = {
    "id": fake.uuid4(),
    "name": fake.name(),
    "uri": fake.uri(),
}

with open("stream.json", 'w') as file:
    json.dump(stream, file)

ax = {
    "id": fake.uuid4(),
    "ip": fake.ipv4(),
    "port": fake.port_number(),
    "name": fake.name(),
    "user_id": fake.name(),
    "user_pw": fake.password(),
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("ax.json", 'w') as file:
    json.dump(ax, file)

tx = {
    "id": fake.uuid4(),
    "name": fake.name(),
    "desc": fake.sentence(),
    "user_id": fake.name(),
    "user_pw": fake.password(),
    "license_number": fake.pyint(min_value=1, max_value=8),
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("tx.json", 'w') as file:
    json.dump(tx, file)

setting = {
    "id": fake.uuid4(),
    "tx_id": fake.uuid4(),
    "name": fake.name(),
    "setting": fake.pydict(4, allowed_types=[str]),
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("setting.json", 'w') as file:
    json.dump(setting, file)

project = {
    "id": fake.uuid4(),
    "app_id": fake.uuid4(),
    "experiment_id": fake.pyint(min_value=1, max_value=8),
    "name": fake.name(),
    "desc": fake.sentence(),
    "tags": fake.pylist(4, allowed_types=[str]),
    "data_uri": fake.uri(),
    "meta_uri": fake.uri(),
    "status": ["ALIVE", "DELETED"][fake.pyint(max_value=1)],
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("project.json", 'w') as file:
    json.dump(project, file)

app = {
    "id": fake.uuid4(),
    "name": fake.name(),
    "desc": fake.sentence(),
    "models": [ fake.uuid4() for _ in range(fake.pyint(min_value=1, max_value=4))],
    "properties": fake.pydict(4, allowed_types=[str]),
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("app.json", 'w') as file:
    json.dump(app, file)

model = {
    "id": fake.uuid4(),
    "production_id": fake.uuid4(),
    "name": fake.name(),
    "classes": fake.pylist(4, True, [str]),
    "desc": fake.sentence(),
    "location": fake.url(),
    "status": ["CREATED", "ALIVE", "FAILED", "DELETED"][fake.pyint(max_value=3)],
    "capacity": fake.pyint(1,8),
    "version": "person-net_1.0-d7d3aa59_compute86",
    "platform": ["dgpu", "jetson", "trition"][fake.pyint(max_value=2)],
    "framework": ["deepstream==6.0", "deepstream==5.1"][fake.pyint(max_value=1)],
    "precision": ["FP16", "INT8"][fake.pyint(max_value=1)]
}

with open("model.json", 'w') as file:
    json.dump(model, file)

collector = {
    "id": fake.uuid4(),
    "project_id": fake.uuid4(),
    "name": fake.name(),
    "stream_name": fake.name(),
    "stream_uri": fake.uri(),
    "stream_start": fake.iso8601(),
    "stream_end": fake.iso8601(),
    "assigned": fake.iso8601(),
    "dismissed": fake.iso8601(),
    "setting": {"interval": 2},
    "status": ["RESERVED", "RUNNING", "DONE", "FAILED"][fake.pyint(max_value=3)],
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("collector.json", 'w') as file:
    json.dump(collector, file)

dataset = {
    "id": fake.uuid4(),
    "project_id": fake.uuid4(),
    "name": fake.name(),
    "size": fake.pyint(max_value=1024),
    "source": { fake.uuid4(): fake.pydict(4, allowed_types=[str]) for _ in range(fake.pyint(max_value=4))},
    "tags": fake.pylist(4, True, [str]),
    "status": ["ALIVE", "DELETED"][fake.pyint(max_value=1)],
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("dataset.json", 'w') as file:
    json.dump(dataset, file)

label = {
    "id": fake.uuid4(),
    "dataset_id": fake.uuid4(),
    "source": {
        "label_studio": {
            "project_id":fake.pyint(min_value=1, max_value=8),
            "storage_ids": [ i for i in range(fake.pyint(min_value=1, max_value=4)) ]
        } 
    },
    "tags": fake.pylist(4, True, [str]),
    "status": ["ALIVE", "DELETED"][fake.pyint(max_value=1)],
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("label.json", 'w') as file:
    json.dump(label, file)

train = {
    "id": fake.uuid4(),
    "project_id": fake.uuid4(),
    "name": fake.name(),
    "uri": fake.uri(),
    "entry_point": 'tools/train.py',
    "version": fake.sentence(),
    "params": fake.pydict(4, allowed_types=[str]),
    "metric": fake.pydict(4, allowed_types=[str]),
    "tags": fake.pylist(4, True, [str]),
    "artifact_uri": fake.uri(),
    "status": ["running", "scheduled", "finished", "failed", "killed"][fake.pyint(max_value=4)],
    "assigned": fake.iso8601(),
    "dismissed": fake.iso8601(),
    "updated": fake.iso8601(),
    "created": fake.iso8601()
}

with open("train.json", 'w') as file:
    json.dump(train, file)
