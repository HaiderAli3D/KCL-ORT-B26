"""
Blender Receiver Script for Arm Tracking Data

This script runs inside Blender and receives angle data from the arm tracking program
via UDP socket, then applies the rotations to the three cubes in your rigged model.

Instructions:
1. Open your Blender file (arm.blend)
2. Go to the Scripting workspace
3. Open this script or paste it into a new text block
4. Run the arm_hand_tracker.py script first
5. Then run this script in Blender (Alt+P or click "Run Script")
6. To stop, press ESC in the Blender viewport

Make sure your three cubes are named:
- "BottomCube" (controlled by elbow angle, rotates on local X)
- "MiddleCube" (controlled by wrist rotation, rotates on local Y)
- "TopCube" (controlled by finger-palm angle, rotates on local X)
"""

import bpy
import socket
import json
import math

# Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 9999  # Changed from 5005 to avoid conflicts

# Cube names in your Blender file (matching your Blender scene)
BOTTOM_CUBE = "BOTTOM_CUBE"
MIDDLE_CUBE = "MIDDLE_CUBE"
TOP_CUBE = "TOP_CUBE"

# Global socket variable
sock = None

# Keep track of previous angles for smoothing (optional)
prev_angles = {
    "elbow": 0.0,
    "wrist": 0.0,
    "fingers": 0.0
}

# Smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
SMOOTHING = 0.3


def smooth_angle(new_angle, prev_angle, factor):
    """Apply exponential smoothing to reduce jitter"""
    if new_angle is None:
        return prev_angle
    return prev_angle * factor + new_angle * (1.0 - factor)


def setup_socket():
    """Set up the UDP socket with proper error handling"""
    global sock
    
    # Close existing socket if it exists
    if sock is not None:
        try:
            sock.close()
        except:
            pass
    
    # Create new socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Allow socket reuse
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False)  # Non-blocking socket
        print(f"Successfully bound to {UDP_IP}:{UDP_PORT}")
        return True
    except OSError as e:
        print(f"Error binding socket: {e}")
        print("Try restarting Blender or changing the UDP_PORT in both scripts.")
        return False


def apply_rotations():
    """Receive data and apply rotations to cubes"""
    global sock
    
    if sock is None:
        return None  # Stop the timer
    
    try:
        # Try to receive data (non-blocking)
        data, addr = sock.recvfrom(1024)
        message = data.decode()
        angle_data = json.loads(message)
        
        # Get angle values with smoothing
        if angle_data.get("elbow") is not None:
            prev_angles["elbow"] = smooth_angle(
                angle_data["elbow"], 
                prev_angles["elbow"], 
                SMOOTHING
            )
        
        if angle_data.get("wrist") is not None:
            prev_angles["wrist"] = smooth_angle(
                angle_data["wrist"], 
                prev_angles["wrist"], 
                SMOOTHING
            )
        
        if angle_data.get("fingers") is not None:
            prev_angles["fingers"] = smooth_angle(
                angle_data["fingers"], 
                prev_angles["fingers"], 
                SMOOTHING
            )
        
        # Apply rotations to cubes
        # Bottom cube: Local X rotation based on elbow angle
        if BOTTOM_CUBE in bpy.data.objects:
            obj = bpy.data.objects[BOTTOM_CUBE]
            # Convert degrees to radians and map elbow angle (0-180) to rotation
            # Elbow angle of 180 (fully bent) -> 0 rotation
            # Elbow angle of 0 (fully extended) -> 180 degrees rotation
            elbow_rad = math.radians(180 - prev_angles["elbow"])
            obj.rotation_euler[0] = elbow_rad  # X-axis rotation
        
        # Middle cube: Local Y rotation based on wrist rotation
        if MIDDLE_CUBE in bpy.data.objects:
            obj = bpy.data.objects[MIDDLE_CUBE]
            # Wrist rotation is already in degrees (-90 to +90)
            wrist_rad = math.radians(prev_angles["wrist"])
            obj.rotation_euler[1] = wrist_rad  # Y-axis rotation
        
        # Top cube: Local X rotation based on finger-palm angle
        if TOP_CUBE in bpy.data.objects:
            obj = bpy.data.objects[TOP_CUBE]
            # Finger angle: 0 = extended/open palm (no rotation), 90 = closed/bent (90 deg rotation)
            finger_rad = math.radians(prev_angles["fingers"])
            obj.rotation_euler[0] = finger_rad  # X-axis rotation
        
        # Force viewport update
        bpy.context.view_layer.update()
        
    except BlockingIOError:
        # No data available, this is normal for non-blocking socket
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    # Continue running
    return 0.01  # Update interval in seconds (100 FPS)


def main():
    """Main function to start the receiver"""
    global sock
    
    # Clean up any existing timer first
    if bpy.app.timers.is_registered(apply_rotations):
        bpy.app.timers.unregister(apply_rotations)
    
    print("=" * 50)
    print("Blender Arm Tracking Receiver Started")
    print("=" * 50)
    
    # Set up socket
    if not setup_socket():
        print("Failed to set up socket. Exiting.")
        return
    
    print(f"Make sure arm_hand_tracker.py is running!")
    print()
    print("Cube mappings:")
    print(f"  - {BOTTOM_CUBE}: Elbow angle (Local X)")
    print(f"  - {MIDDLE_CUBE}: Wrist rotation (Local Y)")
    print(f"  - {TOP_CUBE}: Finger-palm angle (Local X)")
    print()
    print("Press ESC or run cleanup() to stop")
    print("=" * 50)
    
    # Verify cubes exist
    missing_cubes = []
    for cube_name in [BOTTOM_CUBE, MIDDLE_CUBE, TOP_CUBE]:
        if cube_name not in bpy.data.objects:
            missing_cubes.append(cube_name)
    
    if missing_cubes:
        print(f"WARNING: The following cubes were not found in the scene:")
        for cube in missing_cubes:
            print(f"  - {cube}")
        print("Please update the cube names in this script to match your scene.")
        print()
    
    # Register the timer to run apply_rotations repeatedly
    bpy.app.timers.register(apply_rotations)


# Cleanup function to stop the receiver
def cleanup():
    """Clean up resources when stopping"""
    global sock
    
    if bpy.app.timers.is_registered(apply_rotations):
        bpy.app.timers.unregister(apply_rotations)
    
    if sock is not None:
        try:
            sock.close()
        except:
            pass
        sock = None
    
    print("Receiver stopped")

# Run the main function
if __name__ == "__main__":
    main()
