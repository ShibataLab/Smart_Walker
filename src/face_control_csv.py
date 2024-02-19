#lilianpicard21@gmail.com ----- 17/04/2023
#This code is the control system for the walker (Face orientation control)
import csv
import os
import numpy as np
#Import Hebi part
import hebi
from time import sleep, time
#Import realsense part
from camera_face_control import IntelRealSense 
#Import open cv
import cv2
#Import mediapipe
import mediapipe as mp
import time

#writing csv file to have the diferent feedback informations
def Writing_csv_file():
  with open ('command_feedback.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    outputdata = [
		  totalTime,   
			group_feedback.velocity[0],
			group_feedback.velocity[1],
			group_feedback.velocity_command[0],
			group_feedback.velocity_command[1],
			group_feedback.effort[0],
			group_feedback.effort[1],
			group_feedback.effort_command[0],
			group_feedback.effort_command[1],
			group_feedback.position[0],
			group_feedback.position[1],
			group_command.position[0],
			group_command.position[1],
            text,
			]
    writer.writerow(outputdata)

# ====== Mediapipe ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#Getting the two actuators
lookup = hebi.Lookup()

# Wait 2 seconds for the module list to populate
sleep(2.0)
group = lookup.get_group_from_names(["X5-1"], ["X-01146", "X-01079"])
group_command  = hebi.GroupCommand(group.size)
group_feedback = hebi.GroupFeedback(group.size)

RealSenseD435 = IntelRealSense()

#Variables for the speed of the walker
#Change the speed (m/s) to adjust the forward speed of the walker according to the patient speed walk
speed = 0.6
#Turning speed (m/s)can also be adjusted
turn_speed = 0.5
#base torque for commanding the wheels
torque = 0.4
#Variable used to lock the wheels
lock = 0
#Torque variables that command actuators (reset each time the walker stops)
#Create a new torque variable that will be used to control the forward speed of the walker
torque_forward = torque
#Create a new torque variable that will be used to control the turning speed of the walker
torque_turn = torque

#Start Time
start = time.time()

#Csv File
text = None
with open('command_feedback.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time","fbkV_R","fbkV_L","comV_R","comV_L","fbkE_R","fbkE_L","comE_R","comE_L","fbkP_R","fbkP_L","comP_R","comP_L","tick_D","state"])

###CODE###
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:    
    while True:
        #Get the feedback from actuators
        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)

        #For the camera
        RealSenseD435.Render_cam()
        ret, color_image = RealSenseD435.get_frame()

        # ====== FaceOrientation ======

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        color_image = cv2.cvtColor(cv2.flip(color_image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance
        color_image.flags.writeable = False
        # Get the result
        results = face_mesh.process(color_image)
        # To improve performance
        color_image.flags.writeable = True
        # Convert the color space from RGB to BGR
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = color_image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        # Get the 2D Coordinates
                        face_2d.append([x, y])
                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z]) 

                #Get eyes distance
                dX = 60
                dx = np.sqrt(abs((face_2d[4][1]-face_2d[1][1])**2+(face_2d[4][0]-face_2d[1][0])**2))
                #D435 Norm focal
                Norm_fx = 1.88
                fx = min(img_w, img_h) * Norm_fx 
                #Getdistance of the face from the camera in cm
                Dist_z = (fx * (dX /dx))/10
                Dist_z = round(Dist_z, 2)
                print(Dist_z)
                #Distance from which the walker should stop (may not adapt to taller people)
                treshold = 250

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)
                # The camera matrix
                focal_length = 1.88 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                # See where the user's head tilting
                #for 640*480: yt around 10
                #for 1920*1080: yt around 4
                #for 1280*720:
                yt = 5.5
                #print(y)
                if y < -yt:
                    text = "Looking Left"
                elif y > yt:
                    text = "Looking Right"
                else:
                    text = "Forward"
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=face_landmarks,
                        #connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            
        #Condition for the walker to work 
        if results.multi_face_landmarks:
            #Turning right
            if y > yt: 
                #The walker  will be locked if the depth measurment is null
                if Dist_z >= treshold :
                    if lock == 0:
                        #before stoping the walker or applying brake we decrease the effort in steps of 0.01 to avoid jerk
                        if group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        #Decrease speed in case we go from turning left intent to turning right
                        elif group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce Torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [-Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        #Decrease speed in case we goo from going straight to turning right
                        if group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        pos1 = group_feedback.position[0]
                        pos2 = group_feedback.position[1]
                    lock = 1
                    #Create a new torque variable that will be used to control the forward speed of the walker
                    torque_forward = torque
                    #Create a new torque variable that will be used to control the turning speed of the walker
                    torque_turn = torque
                    group_command.position = [pos1, pos2]
                    group.send_command(group_command)
                else:
                    lock = 0
                    if group_feedback.velocity[0]*0.075 < turn_speed and  group_feedback.velocity[1]*0.075 < turn_speed:
                        torque_turn = torque_turn + 0.01
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        group_command.effort = [torque_turn, torque_turn]
                        group_command.position = [np.nan , np.nan]
                        group.send_command(group_command)
                    else:
                        torque_turn = torque_turn - 0.01
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        group_command.effort = [torque_turn, torque_turn]
                        group_command.position = [np.nan , np.nan]    
                        group.send_command(group_command)                   

            #Turning left
            elif y < -yt:
                #The walker  will be locked if the depth measurment is null
                if Dist_z >= treshold:
                    if lock == 0:
                        #before stoping the walker or applying brake we decrease the effort in steps of 0.01 to avoid jerk
                        if group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce Torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [-Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        elif group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                        #Decrease speed in case we go from turning right intent to turning left
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        #Decrease speed in case we go from going straight to intent to turning left
                        elif group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        pos1 = group_feedback.position[0]
                        pos2 = group_feedback.position[1]
                    lock = 1
                    #Create a new torque variable that will be used to control the forward speed of the walker
                    torque_forward = torque
                    #Create a new torque variable that will be used to control the turning speed of the walker
                    torque_turn = torque
                    group_command.position = [pos1, pos2]
                    group.send_command(group_command)
                else:
                    lock = 0
                    if group_feedback.velocity[0]*0.075 > -turn_speed and  group_feedback.velocity[1]*0.075 > -turn_speed:
                        torque_turn = torque_turn + 0.01
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        group_command.effort = [-torque_turn, -torque_turn]
                        group_command.position = [np.nan , np.nan]
                        group.send_command(group_command)
                    else:
                        torque_turn = torque_turn - 0.01
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        group_command.effort = [-torque_turn, -torque_turn]
                        group_command.position = [np.nan , np.nan]     
                        group.send_command(group_command)     
            #Go straight
            else :
                #The walker  will be locked if the depth measurment is null
                if Dist_z >= treshold :
                    if lock == 0:
                        #before stoping the walker or applying brake we decrease the effort in steps of 0.01 to avoid jerk
                        #If the walker is moving forward
                        if group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)                
                        #Decrease speed in case we go from turning left intent to going straight       
                        elif group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                            while group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce Torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [-Walker_Torque, -Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        elif group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                        #Decrease speed in case we go from turning right intent to going straight
                            while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                                #Get feedback
                                group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                                #Get torque
                                Walker_Torque = np.abs(group_feedback.effort[0])
                                #reduce torque
                                Walker_Torque = Walker_Torque - 0.01
                                group_command.effort = [Walker_Torque, Walker_Torque] 
                                group_command.position = [np.nan , np.nan]
                                group.send_command(group_command)
                        pos1 = group_feedback.position[0]
                        pos2 = group_feedback.position[1]
                    lock = 1
                    #Create a new torque variable that will be used to control the forward speed of the walker
                    torque_forward = torque
                    #Create a new torque variable that will be used to control the turning speed of the walker
                    torque_turn = torque
                    group_command.position = [pos1, pos2]
                    group.send_command(group_command)
                #The speed is adjusted to the set speed
                else :
                    lock = 0
                    if group_feedback.velocity[0] > 0.1 and group_feedback.velocity[1] < -0.1:
                        if group_feedback.velocity[0]*0.075 < speed and  group_feedback.velocity[1]*0.075 > -speed:
                            torque_forward = torque_forward + 0.01
                            #Get feedback
                            group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                            group_command.effort = [torque_forward, -torque_forward] 
                            group_command.position = [np.nan , np.nan]
                            group.send_command(group_command)
                            #print(group_feedback.velocity[0]*0.075)
                        else :
                            torque_forward = torque_forward - 0.01
                            #Get ::àààààààààààààààààààààààààààààààààààààààààokiuiolàpm^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^q
                            group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                            group_command.effort = [torque_forward, -torque_forward] 
                            group_command.position = [np.nan , np.nan]
                            group.send_command(group_command)
                            #print(group_feedback.velocity[0]*0.075)

        #When powered on and without any face detected, the walker is locked
        else:
            if lock == 0:
                #This is safety in case there is no face detection
                #before stoping the walker or applying brake we decrease the effort in steps of 0.01 to avoid jerk
                #If walker is turning right
                if group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                    while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] > 0.09:
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        #Get torque
                        Walker_Torque = np.abs(group_feedback.effort[0])
                        #reduce torque
                        Walker_Torque = Walker_Torque - 0.01
                        group_command.effort = [Walker_Torque, Walker_Torque] 
                        group_command.position = [np.nan , np.nan]
                        group.send_command(group_command)
                #If walker is turning left
                elif group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                    while group_feedback.velocity[0] < -0.09 and group_feedback.velocity[1] < -0.09:
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        #Get torque
                        Walker_Torque = np.abs(group_feedback.effort[0])
                        #reduce Torque
                        Walker_Torque = Walker_Torque - 0.01
                        group_command.effort = [-Walker_Torque, -Walker_Torque] 
                        group_command.position = [np.nan , np.nan]
                        group.send_command(group_command)
                #If walker is going straight
                elif group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                    #Get torque
                    Walker_Torque = np.abs(group_feedback.effort[0])
                    while group_feedback.velocity[0] > 0.09 and group_feedback.velocity[1] < -0.09:
                        #Get feedback
                        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
                        #reduce torque
                        Walker_Torque = Walker_Torque - 0.01
                        group_command.effort = [Walker_Torque, -Walker_Torque] 
                        group_command.position = [np.nan , np.nan]
                        group.send_command(group_command)
                pos1 = group_feedback.position[0]
                pos2 = group_feedback.position[1]
            lock = 1
            #Create a new torque variable that will be used to control the forward speed of the walker
            torque_forward = torque
            #Create a new torque variable that will be used to control the turning speed of the walker
            torque_turn = torque
            group_command.position = [pos1, pos2]
            group.send_command(group_command)
        
        if results.multi_face_landmarks:
            # Show images
            cv2.line(color_image, p1, p2, (255, 0, 0), 3)
            # Add the text on the image
            cv2.putText(color_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(color_image, "dist: " + str(Dist_z), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if Dist_z >= treshold:
                cv2.putText(color_image, "Stop", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(color_image, "Functionning", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
            cv2.putText(color_image, "Set_Speed" + str(speed), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
            #cv2.putText(color_image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(color_image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(color_image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
            #cv2.imshow('Depth_Camera', depth_colormap)
            #print("FPS: ", fps)
            #cv2.putText(color_image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Head Pose Estimation', color_image)

        end = time.time()
        totalTime = end - start
        try:
            fps = 1 / totalTime
        except ZeroDivisionError :
            fps= 0

        #Register Csvfile
        Writing_csv_file()

        sleep(0.1)

        if cv2.waitKey(1) & 0xFF == 27:
            break