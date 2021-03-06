import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # 검사 간격 (초)
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        
monitor = Monitor(5)  # 5초 간격으로 검사 결과를 출력   

# monitor.stop()  #  모니터링 종료
