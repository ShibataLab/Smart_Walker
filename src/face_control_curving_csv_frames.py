#lilianpicard21@gmail.com ----- 17/04/2023
#Edited by Kakeru Yamasaki ----27/8/2023
#This code is the control system for the walker (Face orientation control)
import csv
import os
import numpy as np
import hebi
import cv2
import mediapipe as mp
from time import sleep, time
from camera_face_control import IntelRealSense
from save_frames_from_camera import save_frames_from_camera
import time

from frame_capture import FrameCapture

CSV_HEADER = ["time","fbkV_R","fbkV_L","comV_R","comV_L","fbkE_R","fbkE_L","comE_R","comE_L","fbkP_R","fbkP_L","comP_R","comP_L","tick_D","state"]

#writing csv file to have the diferent feedback informations

def write_csv_entry(totalTime, group_feedback, group_command, text):
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
    with open ('command_feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(outputdata)


def adjust_velocity(group, group_feedback, group_command, direction=1):
    while (abs(direction * group_feedback.velocity[0]) > 0.09) and (abs(direction * group_feedback.velocity[1]) > 0.09):
        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
        Walker_Torque = np.abs(group_feedback.effort[0]) - 0.01
        group_command.effort = [direction * Walker_Torque, direction * Walker_Torque]
        group_command.position = [np.nan, np.nan]
        group.send_command(group_command)
    return group_feedback, group_command

def adjust_velocity_mixed(group, group_feedback, group_command, direction1=1, direction2=-1):
    while (direction1 * group_feedback.velocity[0] > 0.09) and (direction2 * group_feedback.velocity[1] < -0.09):
        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
        Walker_Torque = np.abs(group_feedback.effort[0]) - 0.01
        group_command.effort = [direction1 * Walker_Torque, direction2 * Walker_Torque]
        group_command.position = [np.nan, np.nan]
        group.send_command(group_command)
    return group_feedback, group_command

def handle_lock(lock_status, group, torque_value, group_feedback, group_command):
    if lock_status == 0:
        group_feedback, group_command = adjust_velocity(group, group_feedback, group_command)
        group_feedback, group_command = adjust_velocity(group, group_feedback, group_command, -1)
        group_feedback, group_command = adjust_velocity_mixed(group, group_feedback, group_command)
        pos1, pos2 = group_feedback.position
    lock_status = 1
    group_command.position = [pos1, pos2]
    group.send_command(group_command)
    return lock_status, group_feedback, group_command

def adjust_torque(lock_state, group, group_feedback, group_command, turn_speed, torque_turn, turn_direction=1):
    if group_feedback.velocity[0]*0.075*turn_direction < turn_speed:
        torque_turn += 0.01
    else:
        torque_turn -= 0.01
    group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
    group_command.effort = [torque_turn * turn_direction, torque_turn * turn_direction]
    group_command.position = [np.nan , np.nan]
    group.send_command(group_command)
    return lock_state, group_feedback, group_command

def adjust_forward_torque(lock_state, group, group_feedback, group_command, speed, torque_forward):
    if group_feedback.velocity[0] > 0.1 and group_feedback.velocity[1] < -0.1:
        if group_feedback.velocity[0]*0.075 < speed and group_feedback.velocity[1]*0.075 > -speed:
            torque_forward += 0.01
        else:
            torque_forward -= 0.01
        group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
        group_command.effort = [torque_forward, -torque_forward] 
        group_command.position = [np.nan , np.nan]
        group.send_command(group_command)
    return lock_state, group_feedback, group_command

def adjust_forward_turn_torque(lock_state, group, group_feedback, group_command, turn_speed, torque_turn, turn_direction=1):
    radius = 0.075 #[m]
    vel_ratio = 0.1 #左右のトルクの比率

    vel_1 = group_feedback.velocity[0] * radius
    vel_2 = group_feedback.velocity[1] * radius
    
    if ((vel_1 + vel_2)/2) < turn_speed:
        torque_turn += 0.01
    else:
        torque_turn -= 0.01
    
    group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)

    #左右どちらの速度にvel_ratioをかけて減速させるか
    if turn_direction == 1:
        group_command.effort = [torque_turn * turn_direction, torque_turn * turn_direction * 0.5 * vel_ratio]
    else:
        group_command.effort = [torque_turn * turn_direction * vel_ratio, torque_turn * turn_direction * 0.5]

    group_command.position = [np.nan , np.nan]
    group.send_command(group_command)
    return lock_state, group_feedback, group_command

def handle_movement(y, Dist_z, yt, treshold, lock, group, torque, turn_speed, speed, torque_turn, torque_forward, group_feedback, group_command):
    # 鍵をかける条件
    if Dist_z >= treshold:
        return handle_lock(lock, group, torque, group_feedback, group_command)

    # 鍵をかける条件でない場合
    if y > yt:
        # return adjust_torque(group, group_feedback, group_command, turn_speed, torque_turn, 1)
        return adjust_forward_turn_torque(lock, group, group_feedback, group_command, turn_speed, torque_turn, 1)
    elif y < -yt:
        # return adjust_torque(group, group_feedback, group_command, turn_speed, torque_turn, -1)
        return adjust_forward_turn_torque(lock, group, group_feedback, group_command, turn_speed, torque_turn, -1)

    else:
        return adjust_forward_torque(lock, group, group_feedback, group_command, speed, torque_forward)

def setup_mediapipe():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    return mp_face_mesh, mp_drawing, drawing_spec

def get_hebi_group(names=["X5-1"], ids=["X-01146", "X-01079"], wait_time=2.0):
    """Returns a HEBI group given module names and ids"""
    
    # Getting the actuators
    lookup = hebi.Lookup()
    
    # Wait for the module list to populate
    sleep(wait_time)
    
    # Get group and group command/feedback based on names and ids
    group = lookup.get_group_from_names(names, ids)
    group_command  = hebi.GroupCommand(group.size)
    group_feedback = hebi.GroupFeedback(group.size)

    return group, group_command, group_feedback

def initialize_walker_parameters(speed_val=0.6, turn_speed_val=0.5, torque_val=0.4, lock_val=0):
    """
    Initializes and returns walker parameters.
    
    Args:
    - speed_val: forward speed of the walker (default: 0.6 m/s)
    - turn_speed_val: turning speed of the walker (default: 0.5 m/s)
    - torque_val: base torque for the wheels (default: 0.4)
    - lock_val: variable used to lock the wheels (default: 0)
    
    Returns:
    - Dictionary containing the initialized parameters
    """

    parameters = {
        "speed": speed_val,
        "turn_speed": turn_speed_val,
        "torque": torque_val,
        "lock": lock_val,
        "torque_forward": torque_val,
        "torque_turn": torque_val,
        "start": time.time()
    }
    
    return parameters

def process_face_landmarks(face_landmarks, yt, img_w, img_h):
    face_2d = []
    face_3d = []
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
    return y, Dist_z, text, p1, p2, face_landmarks

def initialize_csv():
    with open('command_feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

def main():
    # Mediapipe のセットアップ
    mp_face_mesh, mp_drawing, drawing_spec = setup_mediapipe()
    group, group_command, group_feedback = get_hebi_group()
    RealSenseD435 = IntelRealSense()
    walker_params = initialize_walker_parameters(speed_val=0.7, turn_speed_val=0.6)
    initialize_csv()

    text = None
    ###CODE###
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:    
        yt = 5.5
        treshold = 250

        # frame numbers
        frame_count = 0
        output_dir = "./Smart_Walker_Code/Control_Face_Orientation_Mediapipe/frames"

        while True:
            #Get the feedback from actuators
            group_feedback = group.get_next_feedback(reuse_fbk=group_feedback)
            #For the camera
            RealSenseD435.Render_cam()
            ret, color_image = RealSenseD435.get_frame()

            frame_count = save_frames_from_camera(output_dir, frame_count, color_image)

            color_image = cv2.cvtColor(cv2.flip(color_image, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(color_image)
            # To improve performance
            color_image.flags.writeable = True
            # Convert the color space from RGB to BGR
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            img_h, img_w, img_c = color_image.shape
            

            if results.multi_face_landmarks:
                y, Dist_z, text, p1, p2, face_landmarks = process_face_landmarks(results.multi_face_landmarks[0],yt, img_w, img_h)
                mp_drawing.draw_landmarks(
                    image=color_image,
                    landmark_list=face_landmarks,
                    #connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # 変数 speed, torque_turn, torque_forward, torque, turn_speed, lock, group_feedback, group_command が適切にセットされていることを想定しています。
                walker_params["lock"], group_feedback, group_command = handle_movement(y, Dist_z, yt, treshold, walker_params["lock"], group, walker_params["torque"], walker_params["turn_speed"], walker_params["speed"], walker_params["torque_turn"], walker_params["torque_forward"], group_feedback, group_command)

                # Show images
                cv2.line(color_image, p1, p2, (255, 0, 0), 3)
                # Add the text on the image
                cv2.putText(color_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(color_image, "dist: " + str(Dist_z), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if Dist_z >= treshold:
                    cv2.putText(color_image, "Stop", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(color_image, "Functionning", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                cv2.putText(color_image, "Set_Speed" + str(walker_params["speed"]), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                #cv2.putText(color_image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(color_image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(color_image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
                #cv2.imshow('Depth_Camera', depth_colormap)
                #print("FPS: ", fps)
                #cv2.putText(color_image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            else:
                lock, group_feedback, group_command = handle_lock(walker_params["lock"], group, walker_params["torque"], group_feedback, group_command)

            cv2.imshow('Head Pose Estimation', color_image)

            end = time.time()
            totalTime = end -  walker_params["start"]
            try:
                fps = 1 / totalTime
            except ZeroDivisionError :
                fps= 0

            #Register Csvfile
            write_csv_entry(totalTime, group_feedback, group_command, text)

            sleep(0.1)

            if cv2.waitKey(1) & 0xFF == 27:
                break

if __name__ == "__main__":
    main()
    
