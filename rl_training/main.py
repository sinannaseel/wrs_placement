#!/usr/bin/python3

import time
import numpy as np

from dqrobotics import *
from dqrobotics.interfaces.coppeliasim import DQ_CoppeliaSimInterfaceZMQ

def main():
    vi = DQ_CoppeliaSimInterfaceZMQ()
    try:
        if not vi.connect("localhost", 23000, 500, 1):
            raise RuntimeError("Unable to connect to CoppeliaSim.")

        vi.set_synchronous(True)
        vi.start_simulation()
        time.sleep(0.2)




    except (Exception, KeyboardInterrupt) as e:
        print(e)
        pass

if __name__ == '__main__':
    main()