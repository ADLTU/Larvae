import pandas as pd
import numpy as np
import scipy.spatial as spatial
import cv2


def get_angle(x1, y1, x2, y2,  x3, y3):
    d12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    d13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    rad = np.arccos((d12**2 + d13**2 - d23**2) / (2 * d12 * d13))
    return int(np.rad2deg(rad))


positions = pd.read_csv("/Users/hadi/Applications/Data_Analysis/Last_min.csv")
cap = cv2.VideoCapture('/Users/hadi/Applications/Larv-Hadi-2020-10-19/Black_ws-Spine/videos/Las_minG3.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('/Users/hadi/Applications/Larv-Hadi-2020-10-19/Black_ws-Spine/videos/Las_minG4.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

x1_coo = np.array(list(map(float, positions.values[2:,1])))
x2_coo = np.array(list(map(float, positions.values[2:,4])))
x3_coo = np.array(list(map(float, positions.values[2:,7])))
x4_coo = np.array(list(map(float, positions.values[2:,10])))
x5_coo = np.array(list(map(float, positions.values[2:,13])))
x6_coo = np.array(list(map(float, positions.values[2:,16])))
sx1_coo = np.array(list(map(float, positions.values[2:,22])))
sx2_coo = np.array(list(map(float, positions.values[2:,25])))
sx3_coo = np.array(list(map(float, positions.values[2:,28])))
sx4_coo = np.array(list(map(float, positions.values[2:,31])))
sx5_coo = np.array(list(map(float, positions.values[2:,34])))
sx6_coo = np.array(list(map(float, positions.values[2:,37])))

xc_coo = (x3_coo + x4_coo + x5_coo + sx6_coo) / 4

y1_coo = np.array(list(map(float, positions.values[2:,2])))
y2_coo = np.array(list(map(float, positions.values[2:,5])))
y3_coo = np.array(list(map(float, positions.values[2:,8])))
y4_coo = np.array(list(map(float, positions.values[2:,11])))
y5_coo = np.array(list(map(float, positions.values[2:,14])))
y6_coo = np.array(list(map(float, positions.values[2:,17])))
sy1_coo = np.array(list(map(float, positions.values[2:,23])))
sy2_coo = np.array(list(map(float, positions.values[2:,26])))
sy3_coo = np.array(list(map(float, positions.values[2:,29])))
sy4_coo = np.array(list(map(float, positions.values[2:,32])))
sy5_coo = np.array(list(map(float, positions.values[2:,35])))
sy6_coo = np.array(list(map(float, positions.values[2:,38])))

yc_coo = (y3_coo + y4_coo + y5_coo + sy6_coo) / 4

x_values = []
y_values = []
for i in range(len(xc_coo)-1):
    x_values.append(abs(xc_coo[i] - xc_coo[i+1]))
    y_values.append(abs(yc_coo[i] - yc_coo[i+1]))

x_v = sum(x_values) / len(x_values)
y_v = sum(y_values) / len(y_values)

t_v = (y_v + x_v)/2

zc_coo = [i*t_v for i in range(len(xc_coo))]
tc_coo = [i for i in range(len(xc_coo))]

cvalues = np.stack((xc_coo, yc_coo, zc_coo), axis=1)

cpoint_tree = spatial.cKDTree(cvalues)

ppc = []
for i, v in enumerate(cvalues):
    # How many points are within X units of the v
    ppc.append(len(cpoint_tree.data[cpoint_tree.query_ball_point(v, 11)]))  #11

# print((ppc))
new_cv = np.stack((xc_coo, yc_coo, tc_coo, ppc), axis=1)
final_cv = np.array([v.tolist() for v in new_cv if v[3] > max(ppc)-2])
final_cam = np.array([v.tolist() for v in new_cv if v[3] > min(ppc)+3])

step_count = 0
for i in range(len(final_cv)):
    try:
        if final_cv[i][2]+4 < final_cv[i+1][2]:
            step_count += 1
    except:
        print("Done counting Steps")

cam_count = 0
for i in range(len(final_cam)):
    try:
        if final_cam[i][2]+4 < final_cam[i+1][2]:
            cam_count += 1
    except:
        print("Done counting cam moves")

head_angles = []
body_angle = []
for t in tc_coo:
    head_angle = get_angle(x1_coo[t], y1_coo[t], sx1_coo[t], sy1_coo[t], sx3_coo[t], sy3_coo[t])

    h = get_angle(x1_coo[t], y1_coo[t], sx1_coo[t], sy1_coo[t], sx2_coo[t], sy2_coo[t])
    b1 = get_angle(sx1_coo[t], sy1_coo[t], sx2_coo[t], sy2_coo[t], sx3_coo[t], sy3_coo[t])
    b2 = get_angle(sx2_coo[t], sy2_coo[t], sx3_coo[t], sy3_coo[t], sx4_coo[t], sy4_coo[t])
    b3 = get_angle(sx3_coo[t], sy3_coo[t], sx4_coo[t], sy4_coo[t], sx5_coo[t], sy5_coo[t])
    b4 = get_angle(sx4_coo[t], sy4_coo[t], sx5_coo[t], sy5_coo[t], sx6_coo[t], sy6_coo[t])
    b5 = get_angle(sx1_coo[t], sy1_coo[t], sx3_coo[t], sy3_coo[t], sx6_coo[t], sy6_coo[t])


    head_angles.append(head_angle)

    body_angle.append(sum([b1, b2, b3, b4, b5]) // 5)

search_time = 0
turn_indexes = []
turn_count = 0

i = 0
while cap.isOpened() and i < 1519:
    ret, frame = cap.read()

    # # # ------------------------ # #
    # # # FROM HERE BEGINS OVERLAY # #
    # # # ------------------------ # #
    # foreground = cv2.imread('/Users/hadi/Applications/Data_Analysis/Graph/x'+str(i)+'.png')
    #
    # scale_percent = 50  # percent of original size
    # width = int(foreground.shape[1] * scale_percent / 100)
    # height = int(foreground.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # # resize image
    # foreground = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)
    #
    # rows, cols, channels = foreground.shape
    # bg_rows, bg_cols, bg_channels = frame.shape
    #
    # roi = frame[0:rows, 0:cols]
    # # roi = frame[rows:rows*2, 0:cols]
    #
    # # Now create a mask of logo and create its inverse mask also
    # img2gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    #
    # # Now black-out the area of logo in ROI
    # img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    #
    # # Take only region of logo from logo image.
    # img2_fg = cv2.bitwise_and(foreground, foreground, mask=mask)
    #
    # # Put logo in ROI and modify the main image
    # dst = cv2.add(img1_bg, img2_fg)
    # frame[0:rows, 0:cols] = dst
    # # frame[rows:rows*2, 0:cols] = dst
    #
    # # # ------------------------ # #
    # # #     HERE ENDS OVERLAY    # #
    # # # ------------------------ # #

    cv2.putText(frame, "Head Angle: " + str(head_angles[i]),
                (450, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (230, 160, 80), 2,
                cv2.LINE_AA)

    cv2.putText(frame, "Body Angle: " + str(body_angle[i]),
                (450, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (230, 160, 80), 2,
                cv2.LINE_AA)

    if body_angle[i] > 5:
        turn_indexes.append(i)
        cv2.putText(frame, "Body Angle: " + str(body_angle[i]) + " Turning!!",
                    (450, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)
    elif head_angles[i] > 10:
        search_time += 1
        cv2.putText(frame, "Head Angle: " + str(head_angles[i]) + " Searching!!",
                    (450, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)

    if i == 1518:
        cv2.putText(frame, "Number of Steps = " + str(step_count),
                    (450, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)

        cv2.putText(frame, "Time spent Searching = " + str(search_time/cap.get(cv2.CAP_PROP_FPS)) + " Seconds",
                    (450, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)

        for tc in range(len(turn_indexes)):
            try:
                if turn_indexes[tc+1] - turn_indexes[tc] > 11:
                    turn_count += 1
            except:
                print("Done counting turns")

        cv2.putText(frame, "Number of turns = " + str(turn_count),
                    (450, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Number of times the camera moved = " + str(cam_count),
                    (450, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (230, 160, 80), 2,
                    cv2.LINE_AA)
    # cv2.imshow('frame', frame)

    output.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1


cap.release()
cv2.destroyAllWindows()