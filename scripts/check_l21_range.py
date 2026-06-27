#!/usr/bin/env python3
"""
测试：几个roll关节的情况
"""

import time
import sys

sys.path.insert(0, '/home/ub/TransHandR/linkerhand_20_lite_test')

from linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

def main():
    print("初始化...")
    hand = LinkerHandApi(hand_joint="L21", hand_type="right", can="can0")
    hand.set_speed(speed=[60, 220, 220, 220, 220])
    hand.set_torque(torque=[255] * 5)
    print("初始化完成")
    
    pose = [255] * 25 
    pose[6] = 255   
    pose[7] = 255  
    pose[8] = 128   
    pose[9] = 0    
    input("\n按 Enter ")
    print("发送...")
    hand.finger_move(pose=pose)
    print("完成")

if __name__ == '__main__':
    main()