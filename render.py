import time
import argparse
import numpy as np

from envs.robots.jvrc_walk import JvrcWalkEnv 
from envs.robots.talos_walk import TalosWalkEnv 

def test_rendering(robot_name):
    # Initialize the environment based on the selected robot
    if robot_name.lower() == "talos":
        env = TalosWalkEnv()
    elif robot_name.lower() == "jvrc":
        env = JvrcWalkEnv()
    else:
        raise ValueError("Invalid robot choice! Please use 'talos' or 'jvrc'.")

    # Reset the environment to get the initial state
    obs = env.reset_model()

    print(f"Starting rendering for {robot_name} environment...")

    # Number of steps for rendering
    num_steps = 500

    for step in range(num_steps):
        # Example action: Zero actions for all actuators
        action = np.zeros(env.action_space.shape)

        # Perform the action
        obs, reward, done, info = env.step(action)

        # Get the MuJoCo viewer (if using mujoco_py)
        viewer = env.viewer

        # Set custom camera position
        if viewer is not None:
            viewer.cam.azimuth = 180  # Rotates around the robot (0 = front, 180 = back)
            viewer.cam.elevation = -30  # Camera tilt (-90 = top-down, 0 = side view)
            viewer.cam.distance = 10.0  # Camera zoom level
            viewer.cam.lookat[:] = [0, 0, 1]  # Center the camera on the robot


        # Render the environment
        env.render()

        # Optional: Slow down rendering for better visibility
        time.sleep(0.03)

        # Reset environment if episode is done
        if done:
            print("Episode finished. Resetting environment...")
            obs = env.reset_model()

        # Wait for user input before proceeding to the next step
        input("Next step (Press Enter to continue...)")

    # Close the environment after rendering
    env.close()
    print("Rendering finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the robot environment for rendering (Talos or JVRC)")
    parser.add_argument("--robot", type=str, choices=["talos", "jvrc"], required=True,
                        help="Choose the robot: 'talos' or 'jvrc'")
    args = parser.parse_args()

    test_rendering(args.robot)
