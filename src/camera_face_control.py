#lilianpicard21@gmail.com ----- 17/04/2023
#This code is the intel real sense part for Face_Control.py
#Import realsense part
import pyrealsense2 as rs
import numpy as np
#Import CV2
import cv2

class IntelRealSense():
    #Initialize the camera
    def __init__(self):
        realsense_ctx = rs.context()
        connected_devices = [] # List of serial numbers for present cameras
        # Get device product line for setting a supporting resolution
        for i in range(len(realsense_ctx.devices)):
            detected_camera =   realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            print(f"{detected_camera}")
            connected_devices.append(detected_camera)

        # Configure depth and color streams
        device = connected_devices[0]
        self.pipeline = rs.pipeline()
        config = rs.config()
        #Enable Streams
        self.xDim = 1280
        self.yDim = 720
        config.enable_stream(rs.stream.color, self.xDim, self.yDim, rs.format.bgr8, 30)
        #Start Streaming
        self.pipeline.start(config)

    def Render_cam(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        self.color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        self.color_image = np.asanyarray(self.color_frame.get_data())

    def get_frame(self):
        if not (self.color_image).any() :
            return False, None, None
        return True, self.color_image
    
    def release(self):
        self.pipeline.stop()

