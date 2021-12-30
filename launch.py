import sys
import os
import importlib
import time
from abpy import management

if __name__ == "__main__":
    task = int(sys.argv[2])-1
    no_tasks = int(sys.argv[3])
    config_file = sys.argv[1]

    c = importlib.import_module("config." + config_file)

    print(time.strftime("%I:%M:%S", time.localtime()) + " LAUNCH: " + config_file + ", "+ str(task+1) + "/" + str(no_tasks), flush=True)

    time.sleep(task)

    a = time.time()
    em = management.ExecutionManager(c.config, 1, True)
    em.generate_runs()
    em.execute_share(task, no_tasks, shuffle=True)
    b=time.time()

    print(time.strftime("%I:%M:%S", time.localtime()) + " TERMINATED: " + config_file + ", "+ str(task+1) + "/" + str(no_tasks) + " [" + str(int(b-a)) + "s]", flush=True)




