import sys
import time
import psutil
from autotune.resource_monitor import ResourceMonitor
from multiprocessing.connection import Listener

if __name__ == '__main__':
    # !!! args to be set
    wt = 0      # warmup time
    rt = 15     # run time
    address = ('100.81.249.186', 6001)

    # listening address
    listener = Listener(address, authkey=b'DBTuner')
    print('Remote Resource Monitor Process is running now!')

    while True:
        conn = listener.accept()
        print("[{}] Tuning Manager is connected to the clientDB!".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        ##### before benchmark runing
        msg = conn.recv()
        if msg == 'done':
            break
        pid = int(msg)
        print("[{}] Monitoring {}!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), pid))
        # start Resource Monitor
        p = psutil.Process(pid)
        p.cpu_percent()
        rm = ResourceMonitor(pid, 1, wt, rt)
        rm.run()

        ##### after benchmark runing
        # block wait for Benchmark-Finish msg
        sig = conn.recv()
        # terminate Resource Monitor
        cpu = p.cpu_percent() / len(p.cpu_affinity())
        rm.terminate()
        # Send back Monitor Data

        print("[{}] Sending clientDB resource data to Tuning Manager!".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory = rm.get_monitor_data_avg()
        conn.send([cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory])
        conn.close()

    # close connection
    listener.close()

