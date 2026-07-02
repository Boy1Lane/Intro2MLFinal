from apscheduler.schedulers.background import BackgroundScheduler


class MonitorScheduler:
    """Runs MonitorService.scan_all on a fixed interval in a worker thread."""

    def __init__(self, monitor, interval_sec: int):
        self._monitor = monitor
        self._interval = max(int(interval_sec), 1)
        self._scheduler = BackgroundScheduler()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.add_job(
            self._monitor.scan_all, "interval", seconds=self._interval,
            id="monitor_scan_all", max_instances=1, coalesce=True,
        )
        self._scheduler.start()
        self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False
