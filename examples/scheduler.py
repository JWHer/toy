from datetime import datetime, timedelta
import os, sys, asyncio, time

from apscheduler.schedulers.asyncio import AsyncIOScheduler


def tick():
    print('Tick! The time is: %s' % datetime.now())

def set_jobstore(scheduler):
    url = sys.argv[1] if len(sys.argv) > 1 else 'sqlite:///example.db'
    scheduler.add_jobstore('sqlalchemy', url=url)
    return scheduler

def add_job(scheduler):
    scheduler.add_job(tick, 'interval', seconds=3)
    return scheduler

def run_job(scheduler):
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    # Execution will block here until Ctrl+C (Ctrl+Break on Windows) is pressed.
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass

def get_jobs(scheduler):
    return scheduler.get_jobs()

def get_job(scheduler):
    return scheduler.get_job()

async def async_job(*args):
    await asyncio.sleep(3)
    print("slept!")
    print(args)


if __name__ == '__main__':
    scheduler = AsyncIOScheduler()
    # scheduler = set_jobstore(scheduler)
    # scheduler.add_job(tick, 'interval', seconds=3)
    scheduler.add_job(async_job, 'date', run_date=(datetime.now() + timedelta(seconds=0)), args=['func', 123, datetime.now()] )
    run_job(scheduler)
    jobs = scheduler.get_jobs()
    print(jobs)

    # time.sleep(6)
    # jobs = scheduler.get_jobs()
    # print(jobs)