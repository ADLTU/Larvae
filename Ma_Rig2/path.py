import cv2
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def get_angle(c, b, a):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


cap = cv2.VideoCapture('/Users/hadi/Applications/tmp/G.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('/Users/hadi/Applications/tmp/Out1.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

positions = pd.read_csv("/Users/hadi/Applications/tmp/G2.csv")

f1x_coo = np.array(list(map(float, positions.values[3:, 1])))
f1y_coo = np.array(list(map(float, positions.values[3:, 2])))
f1c_coo = np.array(list(map(float, positions.values[3:, 3])))
m1x_coo = np.array(list(map(float, positions.values[3:, 4])))
m1y_coo = np.array(list(map(float, positions.values[3:, 5])))
m1c_coo = np.array(list(map(float, positions.values[3:, 6])))
b1x_coo = np.array(list(map(float, positions.values[3:, 7])))
b1y_coo = np.array(list(map(float, positions.values[3:, 8])))
b1c_coo = np.array(list(map(float, positions.values[3:, 9])))

f2x_coo = np.array(list(map(float, positions.values[3:, 10])))
f2y_coo = np.array(list(map(float, positions.values[3:, 11])))
f2c_coo = np.array(list(map(float, positions.values[3:, 12])))
m2x_coo = np.array(list(map(float, positions.values[3:, 13])))
m2y_coo = np.array(list(map(float, positions.values[3:, 14])))
m2c_coo = np.array(list(map(float, positions.values[3:, 15])))
b2x_coo = np.array(list(map(float, positions.values[3:, 16])))
b2y_coo = np.array(list(map(float, positions.values[3:, 17])))
b2c_coo = np.array(list(map(float, positions.values[3:, 18])))

f3x_coo = np.array(list(map(float, positions.values[3:, 19])))
f3y_coo = np.array(list(map(float, positions.values[3:, 20])))
f3c_coo = np.array(list(map(float, positions.values[3:, 21])))
m3x_coo = np.array(list(map(float, positions.values[3:, 22])))
m3y_coo = np.array(list(map(float, positions.values[3:, 23])))
m3c_coo = np.array(list(map(float, positions.values[3:, 24])))
b3x_coo = np.array(list(map(float, positions.values[3:, 25])))
b3y_coo = np.array(list(map(float, positions.values[3:, 26])))
b3c_coo = np.array(list(map(float, positions.values[3:, 27])))

f4x_coo = np.array(list(map(float, positions.values[3:, 28])))
f4y_coo = np.array(list(map(float, positions.values[3:, 29])))
f4c_coo = np.array(list(map(float, positions.values[3:, 30])))
m4x_coo = np.array(list(map(float, positions.values[3:, 31])))
m4y_coo = np.array(list(map(float, positions.values[3:, 32])))
m4c_coo = np.array(list(map(float, positions.values[3:, 33])))
b4x_coo = np.array(list(map(float, positions.values[3:, 34])))
b4y_coo = np.array(list(map(float, positions.values[3:, 35])))
b4c_coo = np.array(list(map(float, positions.values[3:, 36])))

f5x_coo = np.array(list(map(float, positions.values[3:, 37])))
f5y_coo = np.array(list(map(float, positions.values[3:, 38])))
f5c_coo = np.array(list(map(float, positions.values[3:, 39])))
m5x_coo = np.array(list(map(float, positions.values[3:, 40])))
m5y_coo = np.array(list(map(float, positions.values[3:, 41])))
m5c_coo = np.array(list(map(float, positions.values[3:, 42])))
b5x_coo = np.array(list(map(float, positions.values[3:, 43])))
b5y_coo = np.array(list(map(float, positions.values[3:, 44])))
b5c_coo = np.array(list(map(float, positions.values[3:, 45])))


# time_s = []
# sec_count = 0
# got_to = 1
# angle1 = []
# angle2 = []
# angle3 = []
# angle4 = []
# angle5 = []
# to_secs1 = []
# to_secs2 = []
# to_secs3 = []
# to_secs4 = []
# to_secs5 = []
#
# for j in range(0, 908):
#     if got_to == 10:
#         to_secs1.append(sum(angle1) / len(angle1))
#         to_secs2.append(sum(angle2) / len(angle2))
#         to_secs3.append(sum(angle3) / len(angle3))
#         to_secs4.append(sum(angle4) / len(angle4))
#         to_secs5.append(sum(angle5) / len(angle5))
#         got_to = 1
#         angle = []
#         # sec_count += 1
#
#     if (m1c_coo[j] > 0.9) and (f1c_coo[j] > 0.9) and (b1c_coo[j] > 0.9):
#         A1 = get_angle((f1x_coo[j], f1y_coo[j]), (m1x_coo[j], m1y_coo[j]), (b1x_coo[j], b1y_coo[j]))
#     if (m2c_coo[j] > 0.9) and (f2c_coo[j] > 0.9) and (b2c_coo[j] > 0.9):
#         A2 = get_angle((f2x_coo[j], f2y_coo[j]), (m2x_coo[j], m2y_coo[j]), (b2x_coo[j], b2y_coo[j]))
#         angle2.append(A2)
#     if (m3c_coo[j] > 0.9) and (f3c_coo[j] > 0.9) and (b3c_coo[j] > 0.9):
#         A3 = get_angle((f3x_coo[j], f3y_coo[j]), (m3x_coo[j], m3y_coo[j]), (b3x_coo[j], b3y_coo[j]))
#         angle3.append(A3)
#     if (m4c_coo[j] > 0.9) and (f4c_coo[j] > 0.9) and (b4c_coo[j] > 0.9):
#         A4 = get_angle((f4x_coo[j], f4y_coo[j]), (m4x_coo[j], m4y_coo[j]), (b4x_coo[j], b4y_coo[j]))
#         angle4.append(A4)
#         angle1.append(A4)
#         time_s.append(j)
#         print(j, A4-180)
#     else:
#         print(j)
#     if (m5c_coo[j] > 0.9) and (f5c_coo[j] > 0.9) and (b5c_coo[j] > 0.9):
#         A5 = get_angle((f5x_coo[j], f5y_coo[j]), (m5x_coo[j], m5y_coo[j]), (b5x_coo[j], b5y_coo[j]))
#         angle5.append(A5)
#
#
#
#
#
#     got_to += 1
    # else:
    #     angle.append(0)

# fig = plt.figure()
# plt.figure(figsize=(50, 10))
# plt.plot(time_s, angle1, c="#ffff00")
# plt.plot(time_s, angle1, c="#f58742")
# plt.plot(time_s, angle3, c="#80de54")
# plt.plot(time_s, angle4, c="#d638af")
# plt.plot(time_s, angle5, c="#2929f0")
# plt.title('Angle Plot (H)', fontsize=202)
# plt.xlabel('Time (s)', fontsize=14)
# plt.ylabel('Angle (°)', fontsize=14)
# plt.close(fig)
# plt.show()

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    # if i > 50:
    #     print(i)
    # j = i
    for j in range(0, i):
        cv2.circle(frame, center=(int(m1x_coo[j]), int(m1y_coo[j])), thickness=2, radius=1, color=(0, 255, 255))
        # cv2.circle(frame, center=(int(b1x_coo[j]), int(b1y_coo[j])), thickness=2, radius=3, color=(0, 255, 255))
        # cv2.circle(frame, center=(int(f1x_coo[j]), int(f1y_coo[j])), thickness=2, radius=3, color=(0, 255, 255))
        #
        cv2.circle(frame, center=(int(m2x_coo[j]), int(m2y_coo[j])), thickness=2, radius=1, color=(66, 135, 245))
        # cv2.circle(frame, center=(int(b2x_coo[i]), int(b2y_coo[j])), thickness=2, radius=3, color=(66, 135, 245))
        # cv2.circle(frame, center=(int(f2x_coo[i]), int(f2y_coo[j])), thickness=2, radius=3, color=(66, 135, 245))
        #
        cv2.circle(frame, center=(int(m3x_coo[j]), int(m3y_coo[j])), thickness=2, radius=1, color=(84, 222, 128))
        # cv2.circle(frame, center=(int(b3x_coo[i]), int(b3y_coo[i])), thickness=2, radius=3, color=(84, 222, 128))
        # cv2.circle(frame, center=(int(f3x_coo[i]), int(f3y_coo[j])), thickness=2, radius=3, color=(84, 222, 128))
        #
        cv2.circle(frame, center=(int(m4x_coo[j]), int(m4y_coo[j])), thickness=2, radius=1, color=(175, 56, 214))
        # cv2.circle(frame, center=(int(b4x_coo[i]), int(b4y_coo[i])), thickness=2, radius=3, color=(175, 56, 214))
        # cv2.circle(frame, center=(int(f4x_coo[i]), int(f4y_coo[j])), thickness=2, radius=3, color=(175, 56, 214))
        #
        cv2.circle(frame, center=(int(m5x_coo[j]), int(m5y_coo[j])), thickness=2, radius=1, color=(240, 41, 41))
        # cv2.circle(frame, center=(int(b5x_coo[i]), int(b5y_coo[i])), thickness=2, radius=3, color=(240, 41, 41))
        # cv2.circle(frame, center=(int(f5x_coo[i]), int(f5y_coo[j])), thickness=2, radius=3, color=(240, 41, 41))


    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    output.write(frame)

    i += 1


# #
#
#
# every_ten = 1
# distances1 = []
# distances2 = []
# distances3 = []
# distances4 = []
# distances5 = []
#
# t_distances1 = []
# t_distances2 = []
# t_distances3 = []
# t_distances4 = []
# t_distances5 = []
# time_b = []
# tempo = 10
# for i in range(0, 908):
#     if every_ten > tempo:
#         time_b.append(i/tempo)
#         every_ten = 1
#         distances1.append(0.12 * (math.hypot(m1x_coo[i] - m1x_coo[i-tempo], m1y_coo[i] - m1y_coo[i-tempo])))
#         distances2.append(0.12 * (math.hypot(m2x_coo[i] - m2x_coo[i-tempo], m2y_coo[i] - m2y_coo[i-tempo])))
#         distances3.append(0.12 * (math.hypot(m3x_coo[i] - m3x_coo[i-tempo], m3y_coo[i] - m3y_coo[i-tempo])))
#         distances4.append(0.12 * (math.hypot(m4x_coo[i] - m4x_coo[i-tempo], m4y_coo[i] - m4y_coo[i-tempo])))
#         distances5.append(0.12 * (math.hypot(m5x_coo[i] - m5x_coo[i-tempo], m5y_coo[i] - m5y_coo[i-tempo])))
#         t_distances1.append(sum(distances1))
#         t_distances2.append(sum(distances2))
#         t_distances3.append(sum(distances3))
#         t_distances4.append(sum(distances4))
#         t_distances5.append(sum(distances5))
#
#
#     every_ten += 1
#
# fig = plt.figure()
# plt.figure(figsize=(10, 10))
# plt.plot(time_b, t_distances1, c="#ffff00")
# plt.plot(time_b, t_distances2, c="#f58742")
# plt.plot(time_b, t_distances3, c="#80de54")
# plt.plot(time_b, t_distances4, c="#d638af")
# plt.plot(time_b, t_distances5, c="#2929f0")
# plt.title('Distance Traveled (G)', fontsize=20)
# plt.xlabel('Time (3s)', fontsize=14)
# plt.ylabel('Distance (mm)', fontsize=14)
# plt.close(fig)
# plt.show()
#
# print(sum(distances1))
# print(sum(distances2))
# print(sum(distances3))
# print(sum(distances4))
# print(sum(distances5))
