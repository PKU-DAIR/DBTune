import time
import psutil
import subprocess
import multiprocessing as mp
from multiprocessing import Manager


class ResourceMonitor:

    def __init__(self, interval, t):
        self.interval = interval
        self.t = t
        self.ticks = int(self.t / self.interval)
        self.cpu_usage_seq = Manager().list()
        self.mem_usage_seq = Manager().list()
        self.io_read_seq, self.io_write_seq = Manager().list(), Manager().list()
        self.dirty_pages_pct_seq = Manager().list()
        self.processes = []

    def run(self, disk_name, user, sock):
#        self.cpu_usage_seq.clear()
#        self.mem_usage_seq.clear()
#        self.io_read_seq.clear()
#        self.io_write_seq.clear()
#        self.dirty_pages_pct_seq.clear()
#        self.processes.clear()
        p1 = mp.Process(target=self.monitor_cpu_usage, args=())
        self.processes.append(p1)
        p2 = mp.Process(target=self.monitor_mem_usage, args=())
        self.processes.append(p2)
        p3 = mp.Process(target=self.monitor_io_usage, args=(disk_name,))
        self.processes.append(p3)
        p4 = mp.Process(target=self.monitor_dirty_pages, args=(user, sock))
        self.processes.append(p4)
        [proc.start() for proc in self.processes]

    def run_only_cpu(self):
        p1 = mp.Process(target=self.monitor_cpu_usage, args=())
        self.processes.append(p1)
        [proc.start() for proc in self.processes]

    def get_monitor_data(self):
        [proc.join() for proc in self.processes]
        return {
            'cpu': list(self.cpu_usage_seq),
            'mem': list(self.mem_usage_seq),
            'io_read': list(self.io_read_seq),
            'io_write': list(self.io_write_seq),
            'dirty_pages_pct': list(self.dirty_pages_pct_seq)
        }

    def monitor_cpu_usage(self):
        count = 0
        while count < self.ticks:
            cpu_pct = psutil.cpu_percent(interval=self.interval)
            self.cpu_usage_seq.append(cpu_pct)
            count += 1

    def monitor_mem_usage(self):
        count = 0
        while count < self.ticks:
            mem_pct = psutil.virtual_memory()[2]
            self.mem_usage_seq.append(mem_pct)
            time.sleep(self.interval)
            count += 1

    def monitor_io_usage(self, disk_name):
        count = 0
        while count < self.ticks:
            sp1 = psutil.disk_io_counters(perdisk=True, nowrap=True)[disk_name]
            time.sleep(self.interval)
            sp2 = psutil.disk_io_counters(perdisk=True)[disk_name]
            self.io_read_seq.append(sp2[2]-sp1[2])
            self.io_write_seq.append(sp2[3]-sp1[3])
            count += 1

    def monitor_dirty_pages(self, user, sock):
        get_status_cmdfmt = 'mysqladmin -u{} -S {} ext | grep {}'
        r = subprocess.check_output(get_status_cmdfmt.format(user,
                                                             sock,
                                                             'Innodb_buffer_pool_pages_total'),
                                    shell=True)
        total_pages = int(r.decode('utf-8').strip().split()[3])
        count = 0
        while count < self.ticks:
            r = subprocess.check_output(get_status_cmdfmt.format(user,
                                                                 sock,
                                                                 'Innodb_buffer_pool_pages_dirty'), shell=True)
            dirty_pages = int(r.decode('utf-8').strip().split()[3])
            self.dirty_pages_pct_seq.append(100 * dirty_pages / total_pages)
            time.sleep(self.interval)
            count += 1

