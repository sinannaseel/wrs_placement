from wrs.robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from enum import Enum

class Choice(Enum):
    HOME = 0
    OPEN_GRIPPER = 1
    CLOSE_GRIPPER = 2

SELECTED_ACTION = Choice(0)

def main():
    robot = XArmLite6X("192.168.1.232", has_gripper=True)
    try:
        if SELECTED_ACTION == Choice.HOME:
            robot.homeconf()
        elif SELECTED_ACTION == Choice.OPEN_GRIPPER:
            robot.open_gripper()
        elif SELECTED_ACTION == Choice.CLOSE_GRIPPER:
            robot.close_gripper()
        else:
            print("Selected action is not supported")

    except (Exception,KeyboardInterrupt) as e:
        print(e)
        pass

if __name__ == "__main__":
    main()