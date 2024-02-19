import cv2

def save_frames_from_camera(output_dir, frame_count, frame):
    # Open the camera
    # cap = cv2.VideoCapture(0)  # 0 for default camera
    
    # Check if the camera opened successfully
    #if not cap.isOpened():
    #    print("Error: Unable to open camera")
    #    return
    
    # Initialize frame counter
    #frame_count = 0
    
    # Read and save frames continuously
    # while True:
        # Read a frame from the camera
    #    ret, frame = cap.read()
        
        # Check if frame was read successfully
    #    if not ret:
    #        print("Error: Unable to read frame from camera")
    #        break
        
        # Construct output file path
    output_path = f"{output_dir}/frame_{frame_count}.jpg"
        
        # Save the frame as an image
    cv2.imwrite(output_path, frame)
        
        # Increment frame counter
    frame_count += 1
    
    # Release the camera
    # cap.release()
    print(f"Frames saved successfully: {frame_count}")

    return frame_count