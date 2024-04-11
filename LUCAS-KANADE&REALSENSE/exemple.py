import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())

#import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/media/matheusfdario/HD/REALSENSE/test/20240314_143232.bag")
profile = pipe.start(cfg)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Skip 5 first frames to give the Auto-Exposure time to adjust
#for x in range(5):
#    pipe.wait_for_frames()

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

print('a')
# Cleanup:
#pipe.stop()
#print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
#plt.rcParams["axes.grid"] = False
#plt.rcParams['figure.figsize'] = [12, 6]
#plt.imshow(color)
#plt.show()


colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
#plt.imshow(colorized_depth)
#plt.show()

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
aligned_color_frame = frameset.get_color_frame()
aligned_color = np.asanyarray(aligned_color_frame.get_data())
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

old_frame = aligned_color
# Take first frame and find corners in it
#ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
list_x = []
list_y = []

# Show the two frames together:
#images = np.hstack((color, colorized_depth))

#cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('RealSense', images)
#cv2.waitKey(1)

while True:
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    #print('a')
    # Cleanup:
    #pipe.stop()
    #print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    #plt.rcParams["axes.grid"] = False
    #plt.rcParams['figure.figsize'] = [12, 6]
    #plt.imshow(color)
    #plt.show()


    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    #plt.imshow(colorized_depth)
    #plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    aligned_color_frame = frameset.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    aligned_color = np.asanyarray(aligned_color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    if not depth_frame or not color_frame:
        continue

    # Take first frame and find corners in it
    #ret, old_frame = cap.read()
    frame = aligned_color
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        #if(a <frame.shape[1]):
        #    if(a <frame.shape[0]):
                # Get the  depth value
        #        depth = depth_frame.get_distance(a, b)
        #        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
        #        print(i,a,b,depth_point)
        #    else:
        #        print(i, 'OUT: B')
        #else:
        #    print(i, 'OUT: A')
        try:
            depth = depth_frame.get_distance(a, b)
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
            print(i,a,b,depth_point)
        except:
            print(i, a, b, 'OUT')
    list_x.append(a)
    list_y.append(b)



    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    #out.write(img)  # Write frame to output video

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    '''
    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    # Show the two frames together:
    images = np.hstack((color, colorized_depth))

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)
    '''

#cv2.imshow("Depth Stream", images)
#plt.imshow(images)
#plt.show()
'''
    key = cv2.waitKey(1)
    # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
'''
# Cleanup:
pipe.stop()
print("Frames Captured")