import asyncio, logging

logger = logging.getLogger(__name__)

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
            loop = self._loop if self._loop else asyncio.get_running_loop()
            self._queue = asyncio.Queue(loop=loop)    
        return self._queue

    def register(self, name, func):
        """
        register function
        """
        self._tasks[name] = func
        logger.info(f'{name} registered')
    
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
            task = asyncio.create_task(func(*load['args']))
            logger.info(f'exec : {load["name"]}')
            if load['sync']: await task

# Create for each gpu
worker = Worker()
