# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.spatial as spatial
# import cv2
#
# # positions = pd.read_csv("/Users/hadi/Applications/Data_Analysis/AB.csv")
# # cap = cv2.VideoCapture("/Users/hadi/Applications/Larv-Hadi-2020-10-19/JD_Black_BG/videos/AB.mp4")
#
# positions = pd.read_csv("/Users/hadi/Applications/Data_Analysis/Last_min.csv")
# cap = cv2.VideoCapture('/Users/hadi/Applications/Larv-Hadi-2020-10-19/Black_ws-Spine/videos/Las_min.mp4')
#
# print(cap.get(cv2.CAP_PROP_FPS))
# exit()
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output = cv2.VideoWriter('/Users/hadi/Applications/Data_Analysis/Las_minG2.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
#
#
# x3_coo = np.array(list(map(float, positions.values[2:, 7])))
# x4_coo = np.array(list(map(float, positions.values[2:, 10])))
# x5_coo = np.array(list(map(float, positions.values[2:, 13])))
#
# xc_coo = (x3_coo + x4_coo + x5_coo) / 3
#
# y3_coo = np.array(list(map(float, positions.values[2:, 8])))
# y4_coo = np.array(list(map(float, positions.values[2:, 11])))
# y5_coo = np.array(list(map(float, positions.values[2:, 14])))
#
# yc_coo = (y3_coo + y4_coo + y5_coo) / 3
#
#
# x_values = []
# y_values = []
# for i in range(len(xc_coo)-1):
#     x_values.append(abs(xc_coo[i] - xc_coo[i+1]))
#     y_values.append(abs(yc_coo[i] - yc_coo[i+1]))
#
# x_v = sum(x_values) / len(x_values)
# y_v = sum(y_values) / len(y_values)
#
# t_v = (y_v + x_v)/2
#
# zc_coo = [i*t_v for i in range(len(xc_coo))]
#
#
# c_values = np.stack((xc_coo, yc_coo, zc_coo), axis=1)
#
# c_point_tree = spatial.cKDTree(c_values)
#
#
#
# ppc = []
# for i, v in enumerate(c_values):
#     # How many points are within X units of the v
#     ppc.append(len(c_point_tree.data[c_point_tree.query_ball_point(v, 10)]))  #10
#
# new_cv = np.stack((xc_coo, yc_coo, zc_coo, ppc), axis=1)
# final_cv = np.array([v.tolist() for v in new_cv if v[3] > max(ppc)-2])
#
# waves = []
# for i in range(len(final_cv)):
#     try:
#         if final_cv[i][2]+4 < final_cv[i+1][2]:
#             waves.append([int(final_cv[i][2]), int(final_cv[i+1][2])])
#     except IndexError:
#         print("")
#
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#     # print(frame_n, "*(", waves[0][0], waves[0][1], "*)")
#     if (frame_n >= waves[0][0]) and (frame_n <= waves[0][1]):
#         cv2.circle(frame, center=(600, 20), radius=5, color=(0, 255, 255))
#     if frame_n == waves[0][1]:
#         waves.pop(0)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
#

import pandas as pd
import numpy as np
import scipy.spatial as spatial
import cv2


cap = cv2.VideoCapture('/Users/hadi/Applications/Larv-Hadi-2020-10-19/JD_Black_BG/videos/JD.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('/Users/hadi/Applications/Larv-Hadi-2020-10-19/JD_Black_BG/videos/JDG1.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))


i = 0
while cap.isOpened() and i < 750:
    ret, frame = cap.read()

    # # ------------------------ # #
    # # FROM HERE BEGINS OVERLAY # #
    # # ------------------------ # #
    foreground = cv2.imread('/Users/hadi/Applications/Data_Analysis/Graph/jy'+str(i)+'.png')

    scale_percent = 30  # percent of original size
    width = int(foreground.shape[1] * scale_percent / 100)
    height = int(foreground.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    foreground = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)

    rows, cols, channels = foreground.shape
    bg_rows, bg_cols, bg_channels = frame.shape

    # roi = frame[0:rows, 0:cols]
    roi = frame[rows:rows*2, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(foreground, foreground, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    # frame[0:rows, 0:cols] = dst
    frame[rows:rows*2, 0:cols] = dst

    # # ------------------------ # #
    # #     HERE ENDS OVERLAY    # #
    # # ------------------------ # #

    # cv2.imshow('frame', frame)

    output.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1


cap.release()
cv2.destroyAllWindows()