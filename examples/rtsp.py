import cv2

cap = cv2.VideoCapture('rtsp://admin:init123!!@192.168.0.58:7001/3fc9c921-2562-2252-7128-3dc2ec049d87?pos=2021-11-16T15:11:00')
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: "+str(fps))

# https://docs.opencv.org/3.3.1/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while(cap.isOpened() and len(calc_timestamps)<300):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break

cap.release()

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print('Frame %d difference:'%i, abs(ts - cts))