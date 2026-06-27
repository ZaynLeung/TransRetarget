from inspire_hand import InspireHand

hand = InspireHand('/dev/ttyUSB0')
hand.open()

# 可选：设置速度
hand.set_all_finger_speeds(200)

targets = [1000, 1000, 1000, 1000, 1000, 0]
 # 小拇指 无名指 中指 食指 大拇指 弯曲都是1000直 0弯曲
 # 最后一个是大拇指侧摆，1000摆出去，0摆进来

for finger_id, angle in enumerate(targets):
    hand.set_finger_angle(finger_id, angle)

hand.close()