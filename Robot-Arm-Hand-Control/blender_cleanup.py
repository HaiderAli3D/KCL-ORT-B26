"""
Blender Cleanup Script

This script stops the arm tracking receiver and cleans up resources.
Run this script in Blender (Alt+P) to stop the receiver gracefully.

Instructions:
1. In Blender's Scripting workspace, open this script
2. Press Alt+P or click "Run Script"
3. The receiver will stop and the socket will be closed
"""

import bpy

# Import the functions from blender_receiver if they exist
try:
    from blender_receiver import apply_rotations, cleanup
    print("Imported cleanup functions from blender_receiver module")
    use_module = True
except:
    use_module = False
    print("Running standalone cleanup")

def stop_receiver():
    """Stop the receiver timer and close socket"""
    
    # Try to unregister the timer
    # We need to check for all possible timer function names
    timer_stopped = False
    
    # Check if there's a timer registered
    if bpy.app.timers.is_registered:
        # Try common timer function names
        timer_functions = ['apply_rotations']
        
        for timer_name in timer_functions:
            try:
                # Get the function from globals
                timer_func = None
                if use_module:
                    # If we imported from module, use that function
                    timer_func = apply_rotations
                else:
                    # Otherwise try to find it in bpy namespace
                    if hasattr(bpy, timer_name):
                        timer_func = getattr(bpy, timer_name)
                
                if timer_func and bpy.app.timers.is_registered(timer_func):
                    bpy.app.timers.unregister(timer_func)
                    print(f"Successfully stopped timer: {timer_name}")
                    timer_stopped = True
                    break
            except Exception as e:
                print(f"Could not stop timer {timer_name}: {e}")
    
    if not timer_stopped:
        print("No active timer found (receiver may already be stopped)")
    
    # If we have the cleanup function from the module, use it
    if use_module:
        try:
            cleanup()
            print("Cleanup function executed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    print("=" * 50)
    print("Receiver cleanup complete!")
    print("=" * 50)

# Run the cleanup
if __name__ == "__main__":
    stop_receiver()