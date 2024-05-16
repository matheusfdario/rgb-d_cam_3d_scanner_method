import cv2  # state of the art computer vision algorithms library
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import numpy as np
import os

# path
bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240516_142546.bag"
video_name = '/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/VID/video.mp4'

# var

cir_size = 3
lines = True

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 20),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Setup:/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240426_143549.bag
# Set the playback so it's not done in real-time: https://github.com/IntelRealSense/librealsense/issues/3682#issuecomment-642344385

pipe = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(config)

#Needed so frames don't get dropped during processing:
#profile.get_device().as_playback().set_real_time(False)

playback = profile.get_device().as_playback()
playback.set_real_time(False)

frame_number = 0

while True:
    print(frame_number)
    try:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        playback.pause()
        color_frame = frameset.get_color_frame()
        color_frame_npy = np.asanyarray(color_frame.get_data())
        if(frame_number==0):
            # to LK
            old_frame = color_frame_npy
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

            # video
            height, width, layers = old_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, 10.0, (width, height))


            #width = old_frame.shape[1]
            #height = old_frame.shape[0]
            mask = np.zeros_like(old_gray, dtype=np.uint8)
            mask0 = np.zeros_like(old_frame)
            # Get the coordinates and dimensions of the detect_box
            lim_perc = 0.25
            x = round(width * lim_perc)
            y = round(height * lim_perc)
            w = width - 2 * x
            h = height - 2 * y
            # Set the selected rectangle within the mask to white
            mask[y:y + h, x:x + w] = 255

            p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

            for p in p0:
                image = cv2.circle(old_frame, (int(p[0,0]), int(p[0,1])), radius=cir_size, color=(0, 0, 255), thickness=-1)
            video.write(image)
        else:
            new_frame = color_frame_npy
            frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            detect_corners = False
            for p in good_new:
                if (mask[int(p[1]), int(p[0])] == 0):
                    detect_corners = True
            mask0 = np.zeros_like(new_frame)
            if(detect_corners):
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask0 = cv2.line(mask0, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
                    image = cv2.circle(new_frame, (int(a), int(b)), radius=cir_size, color=(0, 0, 255), thickness=-1)
                lines = False
            else:
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    if(lines):
                        mask0 = cv2.line(mask0, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
                    image = cv2.circle(new_frame, (int(a), int(b)), radius=cir_size, color=(255, 0, 0), thickness=-1)
                lines = True
                image = cv2.add(image, mask0)
            video.write(image)
        frame_number += 1  # get number of frames
        playback.resume()
    except:
        break
# Cleanup:
cv2.destroyAllWindows()
video.release()
pipe.stop()
print("Video Genereted")
