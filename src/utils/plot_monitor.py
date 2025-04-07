import argparse

import matplotlib.pyplot as plt

from log_parser import parse_log_file

parser = argparse.ArgumentParser()
parser.add_argument("f")
args = parser.parse_args()

filename = args.f

df = parse_log_file(filename)
hack = df["thread_id"].unique()[-1]

if "48" in filename:
    plot_name = "PI_8GB"
elif "server" in filename:
    plot_name = "Server"
else:
    plot_name = "PI_2GB"

monitor = df[df["thread_id"] == hack]
monitor.reset_index()

"""NETWORK_I/O"""
NETWORK = monitor[monitor["message"] == "NETWORK_I/O"]

sent = []
recv = []
for i, j in enumerate(NETWORK["values"]):
    if i % 1 == 0:
        sent.append(float(j[0]) / (1024))
        recv.append(float(j[1]) / (1024))

fig, ax1 = plt.subplots()
ax1.grid()
x = list(range(len(recv[:500])))
plt.ylim()
plt.xlim(0, 500)
plt.title(f"NETWORK I/O {plot_name}", fontsize=20)
plt.xlabel("Seconds", fontsize=15)
plt.ylabel("Kilobytes", fontsize=15)
plt.plot(x, recv[:500], color="red", label="Bytes Recv")
plt.plot(x, sent[:500], color="blue", label="Bytes Sent")
plt.legend(bbox_to_anchor=(1.3, 1.0))
fig.savefig(f"network_{plot_name}.jpg", bbox_inches="tight")

"""DISK_I/O"""
DISK = monitor[monitor["message"] == "DISK_I/O"]

d_sent = []
d_recv = []
for i, j in enumerate(DISK["values"]):
    if i % 1 == 0:
        d_sent.append(float(j[0]) / (1024))
        d_recv.append(float(j[1]) / (1024))

fig, ax1 = plt.subplots()
ax1.grid()
plt.xlim(0, 500)
x = list(range(len(recv[:500])))
plt.title(f"DISK I/O {plot_name}", fontsize=20)
plt.xlabel("Seconds", fontsize=15)
plt.ylabel("Kilobytes", fontsize=15)
plt.plot(x, d_recv[:500], color="red", label="Bytes Written")
plt.plot(x, d_sent[:500], color="blue", label="Bytes Read")
plt.legend(bbox_to_anchor=(1.35, 1.0))
fig.savefig(f"disk_{plot_name}.jpg", bbox_inches="tight")

"""CPU_USAGE"""
CPU = monitor[monitor["message"] == "CPU_USAGE"]

cpu = []
for i, j in enumerate(CPU["values"]):
    if i % 1 == 0:
        cpu.append(float(j[0]))

fig, ax1 = plt.subplots()
ax1.grid()
x = list(range(len(recv)))
plt.ylim(0, 100)
plt.xlim(0, 500)
plt.title(f"CPU_USAGE {plot_name}", fontsize=20)
plt.xlabel("Seconds", fontsize=15)
plt.ylabel("Percentage %", fontsize=15)
plt.plot(x, cpu, color="red", label="CPU Utilization %")
plt.legend(bbox_to_anchor=(1.4, 1.0))
fig.savefig(f"cpu_{plot_name}.jpg", bbox_inches="tight")

"""CPU_USAGE vs NETWORK I/O"""
CPU = monitor[monitor["message"] == "CPU_USAGE"]
NETWORK = monitor[monitor["message"] == "NETWORK_I/O"]

cpu = []
for i, j in enumerate(CPU["values"]):
    if i % 1 == 0:
        cpu.append(float(j[0]))

sent = []
recv = []
for i, j in enumerate(NETWORK["values"]):
    if i % 1 == 0:
        sent.append(float(j[0]) / (1024))
        recv.append(float(j[1]) / (1024))

fig, ax1 = plt.subplots()
ax1.grid()
x = list(range(len(recv)))
plt.ylim(0, 100)
plt.xlim(0, 500)
plt.title(f"CPU_USAGE vs NETWORK I/O {plot_name}", fontsize=15)
plt.xlabel("Seconds", fontsize=15)
ax1.set_ylabel("CPU Percentage %", fontsize=15)
ax1.plot(x, cpu, color="green", label="CPU Utilization %")
plt.legend(bbox_to_anchor=(1.5, 1.0))

ax2 = ax1.twinx()
ax2.plot(x, recv, color="red", label="Bytes Recv")
ax2.plot(x, sent, color="blue", label="Bytes Sent")
ax2.set_ylabel("Kilobytes", fontsize=15)
ax2.set_ylim(
    0,
)
plt.legend(bbox_to_anchor=(1.415, 0.91))
fig.savefig(f"cpu_vs_net_{plot_name}.jpg", bbox_inches="tight")
