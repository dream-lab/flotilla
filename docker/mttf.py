import time
import os
import random

mttf = 36
control_interval = 5
random.seed(1122001)

how_many_died = 0
while True:
    # time.sleep(control_interval)
    how_many_to_kill = 0
    for i in range(16):
        decider = random.randint(0, 1000000)
        print(decider)
        if decider < int(control_interval / mttf * 1000000):
            how_many_to_kill += 1
            how_many_died += how_many_to_kill

    print(f"Kill:{how_many_to_kill}")
    # if how_many_to_kill > 0:
    #     os.popen(f"docker rm -f $(docker ps -aq | head -{how_many_to_kill})")

    if how_many_died > 16:
        print("Killed All")
        break
