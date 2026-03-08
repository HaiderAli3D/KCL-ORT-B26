import cv2
import mediapipe as mp
import socket
import json
from collections import deque
from mediapipe_cheats import (
    mp_hands, mp_pose, 
    process_frame, process_pose_frame,
    draw_landmarks, draw_pose_landmarks,
    calculate_wrist_rotation,
    calculate_elbow_angle,
    calculate_finger_palm_angle
)

# Configuration for frame smoothing
SMOOTHING_FRAMES = 6  # Number of frames to average (easily configurable)

def main():
    """
    Main program to track both arms and hands using MediaPipe
    """
    # Initialize socket for sending data to Blender
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blender_ip = "127.0.0.1"  # localhost
    blender_port = 9999  # Changed from 5005 to avoid conflicts
    
    # Initialize frame buffers for smoothing
    elbow_buffer = deque(maxlen=SMOOTHING_FRAMES)
    wrist_buffer = deque(maxlen=SMOOTHING_FRAMES)
    fingers_buffer = deque(maxlen=SMOOTHING_FRAMES)
    
    # Initialize MediaPipe solutions
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Arm and Hand Tracker Started!")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame through pose detection
        pose_results = process_pose_frame(pose, frame)
        
        # Process frame through hand detection
        hand_results = process_frame(hands, frame)
        
        # Initialize angle data dictionary
        angle_data = {
            "elbow": None,
            "wrist": None,
            "fingers": None
        }
        
        # Calculate elbow angle from pose (RIGHT ARM ONLY)
        if pose_results.pose_landmarks:
            draw_pose_landmarks(frame, pose_results.pose_landmarks)
            
            # Calculate RIGHT elbow angle specifically
            elbow_angle = calculate_elbow_angle(pose_results.pose_landmarks, side='right')
            if elbow_angle is not None:
                # Add to smoothing buffer
                elbow_buffer.append(elbow_angle)
                # Calculate smoothed angle (average of past frames)
                smoothed_elbow = sum(elbow_buffer) / len(elbow_buffer)
                angle_data["elbow"] = smoothed_elbow
                
                # Display elbow angle on screen
                cv2.putText(
                    frame,
                    f"Right Elbow: {smoothed_elbow:.1f}deg",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        
        # Calculate wrist rotation and finger-palm angles from RIGHT HAND ONLY
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Get hand type (Left or Right)
                handedness = hand_results.multi_handedness[idx].classification[0].label
                
                # ONLY process RIGHT hand
                if handedness != "Right":
                    continue  # Skip left hand
                
                draw_landmarks(frame, hand_landmarks)
                
                # Calculate wrist rotation angle
                wrist_angle = calculate_wrist_rotation(hand_landmarks)
                # Add to smoothing buffer
                wrist_buffer.append(wrist_angle)
                smoothed_wrist = sum(wrist_buffer) / len(wrist_buffer)
                angle_data["wrist"] = smoothed_wrist
                
                # Calculate finger-palm angle
                finger_angle = calculate_finger_palm_angle(hand_landmarks)
                # Add to smoothing buffer
                fingers_buffer.append(finger_angle)
                smoothed_fingers = sum(fingers_buffer) / len(fingers_buffer)
                angle_data["fingers"] = smoothed_fingers
                
                # Display wrist rotation
                if smoothed_wrist > 30:
                    rotation_label = "Supination"
                elif smoothed_wrist < -30:
                    rotation_label = "Pronation"
                else:
                    rotation_label = "Neutral"
                
                cv2.putText(
                    frame,
                    f"Right Wrist: {smoothed_wrist:+.1f}deg ({rotation_label})",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 100, 0),
                    2
                )
                
                # Display finger-palm angle
                cv2.putText(
                    frame,
                    f"Right Fingers: {smoothed_fingers:.1f}deg",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 255, 255),
                    2
                )
        
        # Send angle data to Blender via UDP socket
        try:
            message = json.dumps(angle_data)
            sock.sendto(message.encode(), (blender_ip, blender_port))
        except Exception as e:
            # Silently handle socket errors to not interrupt tracking
            pass
        
        # Display status text
        cv2.putText(
            frame, 
            "Arm & Hand Tracking - Press 'q' to quit", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame
        cv2.imshow('Arm and Hand Tracker', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    sock.close()
    
    print("Tracker stopped")

if __name__ == "__main__":
    main()
