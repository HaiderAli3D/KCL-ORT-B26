import cv2
import mediapipe as mp
from mediapipe_cheats import (
    mp_hands, mp_pose, 
    process_frame, process_pose_frame,
    draw_landmarks, draw_pose_landmarks
)

def main():
    """
    Main program to track both arms and hands using MediaPipe
    """
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
        
        # Draw pose landmarks (arms, shoulders, etc.)
        if pose_results.pose_landmarks:
            draw_pose_landmarks(frame, pose_results.pose_landmarks)
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
        
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
    
    print("Tracker stopped")

if __name__ == "__main__":
    main()
