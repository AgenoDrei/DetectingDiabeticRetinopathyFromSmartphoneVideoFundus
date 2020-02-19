import gpustat
import notify_run as nr
import threading
import time
import sys


def run():
    print(f'{time.strftime("%H:%M:%S")}> Getting GPU info..')
    notify = nr.Notify()
    try:
        query = gpustat.GPUStatCollection.new_query()
    except Exception:
        raise Exception('Check your gpustat and cuda installation!')
    try:
        endpoint = notify.info()
    except nr.NotConfigured:
        print(notify.register())

    free_gpus = get_free_gpus(query)
    if len(free_gpus) > 0:
        send_gpu_notification(free_gpus, notify)


def get_free_gpus(query):
    free_gpus = []
    for gpu in query.gpus:
        usage = gpu['memory.used'] / gpu['memory.total']
        procs = gpu['processes']
        if usage < 0.5 and (procs is None or len(procs) <= 1):
            free_gpus.append((gpu.index, f'{usage:0.2f}'))

    return free_gpus


def send_gpu_notification(free_gpus, notify_handler):
    notification_str = f'Free GPUs available ({len(free_gpus)}):\n'
    for fg in free_gpus:
        notification_str += f'{fg[0]}: {fg[1]}/1.0\n'
    notify_handler.send(notification_str)
    print(f'{time.strftime("%H:%M:%S")}> {notification_str}')


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


if __name__ == '__main__':
    print(f'{time.strftime("%H:%M:%S")}> Gpu observer starting...')
    
    run()
    rt = RepeatedTimer(60 * 5, run)
    try:
        time.sleep(60 * 60 * 24 * 7)  # Run for one week
    finally:
        rt.stop()
        sys.exit(0)
