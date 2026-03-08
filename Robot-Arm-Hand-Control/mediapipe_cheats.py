import mediapipe as mp
import numpy as np
import cv2

# Initialize MediaPipe hands solution for use in utilities
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_frame(hands, frame):
    """
    Process a frame through MediaPipe hands detection
    
    Args:
        hands: MediaPipe Hands solution instance
        frame: BGR image/frame to process
    
    Returns:
        MediaPipe hand detection results
    """
    # Convert BGR to RGB since MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)

def draw_landmarks(frame, hand_landmarks):
    """
    Draw hand landmarks and connections on frame
    
    Args:
        frame: Image/frame to draw on
        hand_landmarks: MediaPipe hand landmarks for one hand
    """
    mp_drawing.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

def get_hand_type(handedness):
    """
    Get whether the hand is left or right
    
    Args:
        handedness: MediaPipe handedness classification result
    
    Returns:
        str: "Left" or "Right"
    """
    return handedness.classification[0].label

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two landmarks
    
    Args:
        point1: First landmark point
        point2: Second landmark point
    
    Returns:
        float: Distance between points
    """
    return np.sqrt(
        (point1.x - point2.x)**2 + 
        (point1.y - point2.y)**2
    )

def detect_pinch(hand_landmarks, threshold=0.05):
    """
    Detect pinch gesture between thumb and index finger
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        threshold: Maximum distance to consider as a pinch
    
    Returns:
        bool: True if pinch detected, False otherwise
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return calculate_distance(thumb_tip, index_tip) < threshold

def is_finger_raised(hand_landmarks, finger_name):
    """
    Check if a specific finger is raised
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        finger_name: String name of finger ('INDEX', 'MIDDLE', 'RING', 'PINKY')
    
    Returns:
        bool: True if finger is raised, False if lowered
    """
    # Define landmark mappings for each finger
    finger_landmarks = {
        'THUMB': (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_MCP),
        'INDEX': (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        'MIDDLE': (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        'RING': (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        'PINKY': (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    }
    
    if finger_name not in finger_landmarks:
        raise ValueError(f"Unknown finger name: {finger_name}")
        
    tip_landmark, base_landmark = finger_landmarks[finger_name]
    tip = hand_landmarks.landmark[tip_landmark]
    base = hand_landmarks.landmark[base_landmark]
    
    # For thumb, check x-position instead of y
    if finger_name == 'THUMB':
        return tip.x < base.x

    if calculate_distance(tip, base) < 0.1:
        return False
    else:
        return True

    # if abs(tip.y - base.y) < 0.2 and abs(tip.x - base.x) < 0.2:
    #     return False
    # else:
    #     return True
    # Return True if finger tip is above base (lower y value)
    #return tip.y < base.y

def get_hand_center(hand_landmarks):
    """
    Calculate the center point of the hand
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        dict: Contains x, y coordinates of hand center
    """
    # Use multiple points to get a stable center
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # Average the positions
    x = (wrist.x + middle_base.x + index_base.x + ring_base.x) / 4
    y = (wrist.y + middle_base.y + index_base.y + ring_base.y) / 4
    
    return x, y
    #return {'x': x, 'y': y}

def calculate_hand_distance(left_landmarks, right_landmarks):
    """
    Calculate the distance between the centers of two hands using tuple coordinates
    
    Args:d
        left_landmarks: MediaPipe landmarks for left hand
        right_landmarks: MediaPipe landmarks for right hand
        
    Returns:
        float: Distance between hand centers
    """
    # Get the center points of each hand
    left_x, left_y = get_hand_center(left_landmarks)
    right_x, right_y = get_hand_center(right_landmarks)
    
    # Calculate Euclidean distance using the x,y coordinates
    return np.sqrt(
        (left_x - right_x)**2 + 
        (left_y - right_y)**2
    )

def get_hand_gesture(hand_landmarks):
    """
    Detect common hand gestures
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        str: Name of detected gesture
    """
    # Check for pinch
    if detect_pinch(hand_landmarks):
        return "PINCH"
    
    # Get finger tips and bases
    tips = {
        'INDEX': hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        'MIDDLE': hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        'RING': hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        'PINKY': hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    }
    
    bases = {
        'INDEX': hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        'MIDDLE': hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        'RING': hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        'PINKY': hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    }
    
    # Count raised fingers
    raised_fingers = sum(1 for finger in ['INDEX', 'MIDDLE', 'RING', 'PINKY'] 
                        if is_finger_raised(hand_landmarks, finger))
    
    # Check for fist - all fingers must be curled (tips below bases)
    # fingers_curled = all(
    #     tips[finger].y > bases[finger].y + 0.08  # Tips must be significantly below bases
    #     for finger in ['INDEX', 'MIDDLE', 'RING', 'PINKY']
    # )
    
    # Identify common gestures
    if raised_fingers == 0:# and fingers_curled
        return "FIST"
    elif raised_fingers == 1 and is_finger_raised(hand_landmarks, 'INDEX'):
        return "POINTING"
    elif raised_fingers == 2 and is_finger_raised(hand_landmarks, 'INDEX') and is_finger_raised(hand_landmarks, 'MIDDLE'):
        return "PEACE"
    elif raised_fingers <= 4:
        return "OPEN_PALM"
        
    return "UNKNOWN"

def detect_hand_direction(hand_landmarks):
    """
    Detect if a hand is pointing up, down, left, or right
    
    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        str: Direction the hand is pointing ('UP', 'DOWN', 'LEFT', 'RIGHT')
        float: Angle of the hand in degrees
    """
    # Get the key points we'll use for direction detection
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Calculate the angle between the wrist and middle finger
    # We use the middle finger as it's usually the most stable for direction detection
    dx = middle_finger_tip.y - wrist.y
    dy = middle_finger_tip.x - wrist.x
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(dx, dy))
    
    # Normalize angle to 0-360 range
    angle = (angle + 360) % 360
    
    # Determine direction based on angle
    # We use 45-degree sectors for each direction
    if angle < 45 or angle >= 315:
        direction = "RIGHT"
    elif angle < 135:
        direction = "DOWN"
    elif angle < 225:
        direction = "LEFT"
    else:
        direction = "UP"
        
    return direction, angle

def get_hand_landmarks(results):

    """
    Organize hand landmarks by handedness (left/right)
    
    Returns:
        dict: Contains landmarks for each hand
            {
                'left': landmarks or None if no left hand,
                'right': landmarks or None if no right hand
            }
    """
    hands = {'left': None, 'right': None}
    
    # Only process if we detected any hands
    if results.multi_hand_landmarks:
        # Look at each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get whether this is a left or right hand
            handedness = results.multi_handedness[idx].classification[0].label.lower()
            # Store the landmarks in the appropriate category
            hands[handedness] = hand_landmarks
    
    return hands

def detect_thumbs_up(hand_landmarks):
    """
    Detect if the hand is making a thumbs up gesture.
    
    A thumbs up gesture is characterized by:
    1. Thumb extended upward (lower y position than base)
    2. All other fingers closed (curled inward)
    3. Thumb should be significantly above other finger tips
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        bool: True if thumbs up detected, False otherwise
    """
    # Get thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Get other finger tips
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get finger bases (MCP joints)
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if thumb is extended upward
    thumb_extended = thumb_tip.y < thumb_mcp.y
    
    # Check if thumb is significantly above other finger tips
    thumb_highest = all(
        thumb_tip.y < tip.y - 0.12  # Thumb should be significantly higher
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    )
    
    # Check if other fingers are closed by comparing tips to their bases
    fingers_closed = all(
        tip.y > base.y  # Tip should be below base for closed fingers
        for tip, base in [
            (index_tip, index_mcp),
            (middle_tip, middle_mcp),
            (ring_tip, ring_mcp),
            (pinky_tip, pinky_mcp)
        ]
    )
    
    return thumb_extended and thumb_highest and fingers_closed

def detect_thumbs_gesture(hand_landmarks, handedness, THRESHOLD = 30):
    """
    Detect thumbs up/down gesture using angle thresholds.
    
    The gesture is detected based on:
    1. Fingers being curled inward
    2. Thumb extension direction (left for right hand, right for left hand)
    3. Thumb angle relative to vertical (within 30 degrees of up/down)
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: String indicating "Left" or "Right" hand
        
    Returns:
        str: 'UP' for thumbs up, 'DOWN' for thumbs down, 'NONE' for no thumb gesture
    """
    # Get thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Get finger landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    base_list = [index_base, middle_base, ring_base, pinky_base]

    # Check if fingers are curled inward
    # fingers_curled = all(
    #     abs(tip.x - wrist.x) < abs(pip.x - wrist.x)
    #     for tip, pip in [
    #         (index_tip, index_pip),
    #         (middle_tip, middle_pip),
    #         (ring_tip, ring_pip),
    #         (pinky_tip, pinky_pip)
    #     ]
    # )

    if handedness == "Right":
        fingers_curled = all(
            tip.x > index_base.x + 0.02
            for tip, base in [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)   
            ]
        ) 

    else:
        fingers_curled = all(
            tip.x < index_base.x - 0.02 
            for tip, base in [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)   
            ]
        )
    
    correct_rotation = all(
        base.x < thumb_tip.x
        for base in base_list
    )

    if not fingers_curled:
        return "NONE"
    
    if not correct_rotation:
        return "NONE"

    # Calculate the angle between the thumb and the vertical axis
    # First, get the vector from MCP to tip
    dx = thumb_tip.x - thumb_mcp.x
    dy = thumb_tip.y - thumb_mcp.y
    
    # Calculate angle in degrees using arctangent
    # atan2 returns angle in radians in range (-π, π)
    angle = np.degrees(np.arctan2(dx, -dy))  # Negative dy because y increases downward
    
    # Normalize angle to be positive for both hands
    if handedness == "Left":
        angle = -angle
    
    # Determine gesture based on angle
    if abs(angle) <= THRESHOLD:
        return "UP"
    elif abs(angle) >= 180 - THRESHOLD:
        return "DOWN"
    else:
        return "NONE"

# Pose tracking functions
def process_pose_frame(pose, frame):
    """
    Process a frame through MediaPipe pose detection
    
    Args:
        pose: MediaPipe Pose solution instance
        frame: BGR image/frame to process
    
    Returns:
        MediaPipe pose detection results
    """
    # Convert BGR to RGB since MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(rgb_frame)

def draw_pose_landmarks(frame, pose_landmarks):
    """
    Draw pose landmarks and connections on frame
    
    Args:
        frame: Image/frame to draw on
        pose_landmarks: MediaPipe pose landmarks
    """
    mp_drawing.draw_landmarks(
        frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
    )

def calculate_wrist_rotation(hand_landmarks):
    """
    Calculate pronation/supination angle of the forearm
    
    This function determines the rotation of the hand around the wrist axis
    (pronation/supination - when radius crosses over ulna) by analyzing the
    3D orientation of the hand using z-coordinates and landmark positions.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        float: Pronation/supination angle in degrees
               Negative = Pronation (palm down)
               Positive = Supination (palm up)
               0 = Neutral (palm facing sideways)
    """
    # Get key landmarks for calculating hand plane orientation
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Create vectors to define the hand plane
    # Vector from wrist to middle finger base (along hand)
    v1 = np.array([
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z
    ])
    
    # Vector from pinky side to thumb side (across hand)
    v2 = np.array([
        index_mcp.x - pinky_mcp.x,
        index_mcp.y - pinky_mcp.y,
        index_mcp.z - pinky_mcp.z
    ])
    
    # Calculate normal vector to hand plane using cross product
    # This normal points in the direction the palm is facing
    normal = np.cross(v1, v2)
    
    # Normalize the vector
    normal_length = np.linalg.norm(normal)
    if normal_length > 0:
        normal = normal / normal_length
    
    # The z-component of the normal indicates pronation/supination
    # When palm faces camera (supination), normal.z is positive
    # When palm faces away (pronation), normal.z is negative
    # Calculate angle from the z-component
    # Clamp to [-1, 1] to avoid arcsin domain errors
    z_component = np.clip(normal[2], -1.0, 1.0)
    
    # Calculate angle in degrees
    # arcsin gives us angle from -90 (pronation) to +90 (supination)
    angle = np.degrees(np.arcsin(z_component))
    
    return angle

def calculate_elbow_angle(pose_landmarks, side='right'):
    """
    Calculate the angle at the elbow joint
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
        side: 'right' or 'left' to specify which arm
        
    Returns:
        float: Elbow angle in degrees (0 = fully extended, 180 = fully bent)
               Returns None if landmarks are not visible
    """
    # Get the appropriate landmarks based on side
    if side.lower() == 'right':
        shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    else:
        shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    
    # Check if landmarks are visible (visibility threshold)
    if shoulder.visibility < 0.5 or elbow.visibility < 0.5 or wrist.visibility < 0.5:
        return None
    
    # Create vectors for the two arm segments
    # Vector A: from shoulder to elbow
    v1 = np.array([
        elbow.x - shoulder.x,
        elbow.y - shoulder.y,
        elbow.z - shoulder.z
    ])
    
    # Vector B: from elbow to wrist
    v2 = np.array([
        wrist.x - elbow.x,
        wrist.y - elbow.y,
        wrist.z - elbow.z
    ])
    
    # Calculate the angle using dot product
    # cos(θ) = (v1 · v2) / (|v1| * |v2|)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None
    
    # Calculate cosine and clamp to valid range
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def calculate_finger_palm_angle(hand_landmarks):
    """
    Calculate the angle between fingers and palm
    
    This measures how much the fingers are bent relative to the palm plane.
    Useful for gripper control or hand state detection.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        float: Angle in degrees (0 = fingers flat/extended, 90 = fingers perpendicular to palm)
    """
    # Get wrist and MCP (knuckle) landmarks to define palm plane
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Get finger tip landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate average MCP position (center of knuckles)
    avg_mcp = np.array([
        (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4,
        (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4,
        (index_mcp.z + middle_mcp.z + ring_mcp.z + pinky_mcp.z) / 4
    ])
    
    # Calculate average finger tip position
    avg_tip = np.array([
        (index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 4,
        (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4,
        (index_tip.z + middle_tip.z + ring_tip.z + pinky_tip.z) / 4
    ])
    
    # Vector from wrist to knuckles (palm direction)
    palm_vector = avg_mcp - np.array([wrist.x, wrist.y, wrist.z])
    
    # Vector from knuckles to fingertips (finger direction)
    finger_vector = avg_tip - avg_mcp
    
    # Calculate angle between palm and fingers
    dot_product = np.dot(palm_vector, finger_vector)
    magnitude_palm = np.linalg.norm(palm_vector)
    magnitude_finger = np.linalg.norm(finger_vector)
    
    if magnitude_palm == 0 or magnitude_finger == 0:
        return 0.0
    
    # Calculate cosine and clamp
    cos_angle = dot_product / (magnitude_palm * magnitude_finger)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    # Return the supplementary angle so that 0 = extended, 90 = perpendicular
    angle = 180 - np.degrees(np.arccos(cos_angle))
    
    return angle
