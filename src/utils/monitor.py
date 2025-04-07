import datetime
import subprocess
from multiprocessing import Process
from time import time

import psutil

from utils.logger import FedLogger


class Monitor:
    def __init__(self, id: str, pid: int) -> None:
        self.id: str = id
        self.pid: int = pid
        self.logger = FedLogger(id=self.id, loggername="UTIL_MONITOR")
        self.nvidia_smi_available = self.is_nvidia_smi_available()
        self.monitor_process = Process(target=self.monitor_all)
        self.monitor_process.start()

    def set_session(self, session_id):
        self.logger.update_id(session_id)

    def reset_session(self):
        self.logger.update_id("0")

    def is_nvidia_smi_available(self):
        try:
            subprocess.check_output(["nvidia-smi"])
            return True
        except FileNotFoundError:
            return False
        except subprocess.CalledProcessError:
            return False

    def get_gpu_usage(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            gpu_usage = [int(x) for x in output.decode("utf-8").strip().split("\n")]
            return gpu_usage
        except Exception as e:
            print(f"Error in monitor: {e}")
            return None

    def get_memory_from_free(self):
        try:
            output = subprocess.check_output(
                [
                    "free",
                    "-w",
                ]
            )
            memory_usage = [
                i.split() for i in output.decode("utf8").strip().split("\n")
            ]
            memory_usage_ram = memory_usage[1][2]
            memory_usage_swap = memory_usage[2][2]
            return memory_usage_ram, memory_usage_swap
        except Exception as e:
            print(f"free not available: {e}")
            return None, None

    def monitor_all(self):
        if psutil.pid_exists(self.pid):
            process = psutil.Process(self.pid)

        create_time = datetime.datetime.fromtimestamp(process.create_time()).strftime(
            "%Y%m%d - %H:%M:%S"
        )
        self.logger.info("CREATE_TIME", create_time)

        io_0 = psutil.net_io_counters()
        bs, br = io_0.bytes_sent, io_0.bytes_recv
        o_s, o_r = bs, br

        io_disk_0 = process.io_counters()
        dw, dr = io_disk_0.write_bytes, io_disk_0.read_bytes
        d_w, d_r = dw, dr

        while psutil.pid_exists(self.pid):
            if not process:
                break
            try:
                cpu_perc = process.cpu_percent(interval=1)
                monitor_time = time()
                self.logger.info("CPU_USAGE", f"{cpu_perc}")

                try:
                    memory_rss = process.memory_full_info().rss
                    self.logger.info("MEMORY_USAGE_PHYSICAL", f"{memory_rss}")
                    memory_rss_swap = process.memory_full_info().swap
                    self.logger.info("MEMORY_USAGE_SWAP", f"{memory_rss_swap}")
                except Exception as e:
                    self.logger.info("CPU_USAGE", f"{cpu_perc}")
                    memory_rss = process.memory_info().rss

                memory_vms = process.memory_info().vms
                self.logger.info("MEMORY_USAGE_VMS", f"{memory_vms}")
                memory_data_f = process.memory_info().data
                self.logger.info("DATA_SECTION_MEMORY", f"{memory_data_f}")
                memory_ram, memory_swap = self.get_memory_from_free()
                self.logger.info("MEMORY_USAGE_LINUX", f"{memory_ram},{memory_swap}")

                io = psutil.net_io_counters()
                network_bytes_sent_s = round((io.bytes_sent - bs), 2)
                network_bytes_recv_s = round((io.bytes_recv - br), 2)
                self.logger.info(
                    "NETWORK_I/O", f"{network_bytes_sent_s},{network_bytes_recv_s}"
                )
                bs, br = io.bytes_sent, io.bytes_recv

                io_disk = process.io_counters()
                disk_bytes_write_s = round((io_disk.write_bytes - dw), 2)
                disk_bytes_read_s = round((io_disk.read_bytes - dr), 2)
                dw, dr = io_disk.write_bytes, io_disk.read_bytes
                self.logger.info(
                    "DISK_I/O", f"{disk_bytes_write_s},{disk_bytes_read_s}"
                )

                thread_count = process.num_threads()
                self.logger.info("CURRENT_THREADS", f"{thread_count}")

                if self.nvidia_smi_available:
                    gpu_perc = self.get_gpu_usage()
                    self.logger.info("GPU_USAGE", f"{gpu_perc}")

            except psutil.NoSuchProcess as e:
                self.logger.error(f"{e}", "")
            finally:
                o_s, o_r = round((bs - o_s) / (1024 * 1024), 2), round(
                    (br - o_r) / (1024 * 1024), 2
                )
                self.logger.info("TOTAL_NETWORK_IO", f"{o_s},{o_r}")
                d_w, d_r = round((dw - d_w) / (1024 * 1024), 2), round(
                    (dr - d_r) / (1024 * 1024), 2
                )
                self.logger.info("TOTAL_DISK_IO", f"{d_w},{d_r}")
                self.logger.info("MONITOR_TIME", time() - monitor_time)
