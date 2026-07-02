import time

from app.backend.services.scheduler import MonitorScheduler


class FakeMonitor:
    def __init__(self):
        self.calls = 0
    def scan_all(self):
        self.calls += 1


def test_scheduler_runs_job():
    mon = FakeMonitor()
    sched = MonitorScheduler(mon, interval_sec=1)
    sched.start()
    try:
        time.sleep(2.5)
        assert mon.calls >= 1
    finally:
        sched.shutdown()


def test_scheduler_shutdown_is_idempotent():
    sched = MonitorScheduler(FakeMonitor(), interval_sec=1)
    sched.start()
    sched.shutdown()
    sched.shutdown()  # must not raise
