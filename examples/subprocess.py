import asyncio

class Worker:
    def __init__(self, loop=None):
        self._tasks = {}
        self._loop = loop
        self._queue = None

    # @property
    # def loop(self):
    #     if not self._loop:
    #         self._loop = asyncio.get_running_loop()
    #     return self._loop
        
    @property
    def queue(self):
        # initialize queue
        if not self._queue:
            loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue(loop=loop)    
        return self._queue

    def register(self, name, func):
        """
        register function
        """
        self._tasks[name] = func
        print(f'{name} registered')
    
    def run_task(self, name, args=None, sync=False):
        """
        run task with args
        args is single object
        """
        load = {}
        load['name'] = name
        load['args'] = args
        load['sync'] = sync

        self.queue.put_nowait(load)

    async def loop(self):      
        # task loop
        while True:
            load = await self.queue.get()
            func = self._tasks[load['name']]
            task = asyncio.create_task(func(load['args']))
            print(f'exec : {load["name"]}')
            if load['sync']: await task

async def run(cmd):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()
    print(f'{cmd!r} exited with {proc.returncode}')
    if stdout:
        print(f'[stdout] : {stdout.decode()}')
    if stderr:
        print(f'[stderr] : {stderr.decode()}')

worker = Worker()

async def main():
    worker_test = asyncio.create_task(worker.loop())
    
    worker.register('t1', run)
    print('t1 assinged')
    worker.run_task('t1', '/home/jwher/build/autocare_tx/examples/test.sh', True)
    print('t1 run')

    worker.register('t2', run)
    print('t2 assinged')
    worker.run_task('t2', '/home/jwher/build/autocare_tx/examples/test.sh')
    print('t2 run')

    worker.register('t3', run)
    print('t3 assinged')
    worker.run_task('t3', '/home/jwher/build/autocare_tx/examples/test.sh')
    print('t3 run')

    await worker_test

if __name__ == '__main__':
    asyncio.run(main())
    # worker.register('t1', run)
    # print('t1 assinged')
    # worker.run_task('t1', ['test.sh'])
    # print('t1 run')
