from nats.aio.client import Client as NATS
import asyncio, json

d = {"stream":"3645c7ee-ca91-e579-e753-1d85af1fd08c","name":"TestCameraLive2","service":"a7a72476-82d6-9ae5-c47a-afe095d12a25","app":"144682da-f96b-41e9-bb1f-cb1013c4276c","app_name":"capture","models":["66c10733-f5ad-4f28-bf7a-ee8ac97344d8","debc195c-8ee0-486d-8bb1-e81f657967b1"],"uri":"RTSP://127.0.0.1:7777","settings":{"capture":{"enable":"true","supported_storage_system":{"type":"database"},"interval":60,"when":{"operator":"and","expression":{"confidence":{"condition":"larger than","value":0.5},"object_number":{"condition":"larger than","value":0}}}}}}
INFERENCE_FILTER = 'inference.*'
INFERENCE_START = 'inference.start'
INFERENCE_UPDATE = 'inference.update'
INFERENCE_STOP = 'inference.stop'
INFERENCE_STATE_CHANGED = 'inference.state.changed'

ANALYTICS_FILTER = 'analytics.*'
ASSIGN = 'analytics.assign'
UPDATE = 'analytics.update'
STOP = 'analytics.stop'
INSPECT = 'analytics.inspect'

NATS_ENDPOINT = '127.0.0.1:4222'

async def run(loop):
    nc = NATS()

    await nc.connect(NATS_ENDPOINT, loop=loop)

    async def work(msg):
        print(f"subject: {msg.subject} \n\
                reply: {msg.reply} \n\
                data: {msg.data.decode()}")

        await asyncio.sleep(2)                
        if msg.data.decode() == 'pong':
            await nc.publish('analytics.assign', b'ping')
        else:
            await nc.publish('analytics.assign', b'pong')

    await nc.subscribe('analytics.assign', 'worker', work)
    await nc.publish('analytics.assign', json.dumps(d).encode())


async def debug(loop):
    nc = NATS()
    await nc.connect(NATS_ENDPOINT, loop=loop)
    print("connected")

    async def debuger(msg):
        print(f"subject: {msg.subject} \n\
            reply: {msg.reply} \n\
            data: {msg.data.decode()}")

    await nc.subscribe(INFERENCE_FILTER, 'debug', debuger)


async def analytics(loop):
    nc = NATS()
    await nc.connect(NATS_ENDPOINT, loop=loop)
    print("connected")

    d = {
        "id": "f94e6bba787a490eb24d469cd6077d51",
        "inference_id": "01c44839-0d3f-43cb-9c61-976633944f2a",
        "stream": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
        "model": "ca1e3eeb-35a1-4164-a36c-651ebaa73850",
        "uri": "rtsp://admin:init123!!@rnd.snuailab.ai:32348/4d3f64a0-2ce3-16e9-d731-69df5f5cb78f",
        "app": "ca1e3eeb-35a1-4164-a36c-651ebaa73850",
        "app_name": "capture",
        "settings": {
            "capture": {
                "enable": True,
                "storage_system": "database",
                "address": "mongodb://mongo:27017/b1251e13-238d-438c-adc5-789b418adce3/01c44839-0d3f-43cb-9c61-976633944f2a",
                "extra": {
                        "buffer_commit_path": "file:///b1251e13-238d-438c-adc5-789b418adce3/01c44839-0d3f-43cb-9c61-976633944f2a",
                        "buffer_commit_filename": "%(uuid).png"
                },
                "skip_time": 2
            }
        }
    }
    async def debuger(msg):
        print(f"subject: {msg.subject} \n\
            reply: {msg.reply} \n\
            data: {msg.data.decode()}")

    await nc.subscribe(ANALYTICS_FILTER, 'debug', debuger)
    await nc.publish(ASSIGN, json.dumps(d).encode())


async def stop(loop):
    nc = NATS()
    await nc.connect(NATS_ENDPOINT, loop=loop)
    print("connected")

    d = {
        "id": "f94e6bba787a490eb24d469cd6077d51",
        "inference_id": -1
    }
    async def debuger(msg):
        print(f"subject: {msg.subject} \n\
            reply: {msg.reply} \n\
            data: {msg.data.decode()}")

    await nc.subscribe(ANALYTICS_FILTER, 'debug', debuger)
    await nc.publish(STOP, json.dumps(d).encode())


async def inspect(loop):
    nc = NATS()
    await nc.connect(NATS_ENDPOINT, loop=loop)
    print("connected")

    d = {
        "id": "f94e6bba787a490eb24d469cd6077d51",
        "inference_id": 1,
        "stream": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
        "uri": "rtsp://192.168.0.58:7001/3645c7ee-ca91-e579-e753-1d85af1fd08c",
        "app": "846cfd41-ecd4-459e-b21c-d9106f258277",
        "settings": '{"capture": {"enable": "true", "storage_system": "database", "address": "mongodb://localhost:27017/"}}'
    }
    async def debuger(msg):
        print(f"subject: {msg.subject} \n\
            reply: {msg.reply} \n\
            data: {msg.data.decode()}")

    await nc.subscribe(ANALYTICS_FILTER, 'debug', debuger)
    await nc.publish(INSPECT, json.dumps(d).encode())

if __name__ == '__main__':
    cmd = input("select: ")
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(run(loop))
    if cmd.lower()=="stop":
        loop.run_until_complete(stop(loop))
    if cmd.lower()=="ann":
        loop.run_until_complete(analytics(loop))
    loop.run_forever()
    loop.close()
