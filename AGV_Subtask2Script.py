import ai2thor.controller
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def compute_optical_flow(prev_frame, curr_frame):
    # Ensure frames are valid
    if prev_frame is None or curr_frame is None:
        print("Error: One of the frames is None!")
        return np.array([]), np.array([])

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect features using Shi-Tomasi corner detector
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    if features is None:
        print("No features detected, skipping optical flow calculation.")
        return np.array([]), np.array([])

    # Define Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Compute optical flow
    new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None, **lk_params)

    # Filter out valid points
    if new_features is not None and status is not None:
        prev_points = features[status.flatten() == 1]
        curr_points = new_features[status.flatten() == 1]
        return prev_points, curr_points
    else:
        print("Optical flow failed to compute.")
        return np.array([]), np.array([])


def detect_foe(flow_vectors, curr_points):
    if flow_vectors is None or flow_vectors.size == 0:
        print("Warning: No flow vectors detected. Returning default FOE.")
        return np.array([0, 0])  # Default FOE if no motion detected

    try:
        flow_vectors = np.asarray(flow_vectors).reshape(-1, 2)
        curr_points = np.asarray(curr_points).reshape(-1, 2)
        A = np.column_stack((flow_vectors[:, 1], -flow_vectors[:, 0]))
        b = curr_points[:, 0] * flow_vectors[:, 1] - curr_points[:, 1] * flow_vectors[:, 0]

        FOE = np.linalg.lstsq(A, b, rcond=None)[0]
        return FOE
    except np.linalg.LinAlgError as e:
        print(f"Error in FOE computation: {e}. Returning default FOE.")
        return np.array([0, 0])


def compute_potential_field(flow_vectors, foe, target_vector, agent_position, gamma=1.0, alpha=1.0, lambda_x=0.5, lambda_y=0.5, psi=0, obstacle_map=None, sigma=5):
    """
    Computes the potential field for navigation using optical flow.
    Includes Time to Contact (TTC) computation and attraction-repulsion forces.
    Now integrates obstacle detection using Gaussian smoothing and gradient computation.
    """
    flow_vectors = np.squeeze(flow_vectors, axis=1)  # Removes axis=1 if it's of size 1
    x_foe, y_foe = foe
    v_x = flow_vectors[:, 0]
    v_y = flow_vectors[:, 1]
    x_pixels = agent_position[0]
    y_pixels = agent_position[1]
    
    # Compute TTC using the provided equation
    distance_to_foe = np.sqrt((x_pixels - x_foe) ** 2 + (y_pixels - y_foe) ** 2)
    velocity_magnitude = np.sqrt(v_x ** 2 + v_y ** 2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        TTC = np.where(velocity_magnitude > 0, distance_to_foe / velocity_magnitude, np.inf)
    
    # Compute obstacle potential field
    valid_ttc = TTC < np.inf  # Filter valid TTC values
    
    # Apply Gaussian smoothing to the obstacle map
    if obstacle_map is not None:
        smoothed_obstacles = gaussian_filter(obstacle_map.astype(float), sigma=sigma)
        
        # Compute gradients
        g_x, g_y = np.gradient(smoothed_obstacles)
    else:
        g_x = np.gradient(flow_vectors[:, 0])
        g_y = np.gradient(flow_vectors[:, 1])
    
    rep_x = gamma * np.sum(g_x[valid_ttc] / TTC[valid_ttc])
    rep_y = gamma * np.sum(g_y[valid_ttc] / TTC[valid_ttc])
    
    # Compute target potential field
    x_goal, y_goal = target_vector
    att_x = 0.5 * alpha * (x_goal - agent_position[0])*(x_goal - agent_position[0])
    att_y = 0.5 * alpha * (y_goal - agent_position[1])*(y_goal - agent_position[1])
    
    # Compute net force in image plane
    F_xt = att_x - rep_x - lambda_x * rep_x
    F_yt = att_y - rep_y - lambda_y * rep_y
    
    # Transform to global coordinates using yaw angle psi
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    F_x = cos_psi * F_xt + sin_psi * F_yt
    F_y = -sin_psi * F_xt + cos_psi * F_yt
    
    return np.array([F_x, F_y])


def move_ego(controller, agent_state, force_vector, dt=0.1):
    """
    Moves the AI2-THOR agent based on the computed force vector.

    Parameters:
    - controller: AI2-THOR controller instance.
    - agent_state: Dictionary containing the agent's state.
    - force_vector: (Fx, Fz) from potential field.
    - dt: Time step for updating movement.
    """
    # Extract current agent state
    x, y, yaw, v, delta_f = (
        agent_state["x"], agent_state["y"], agent_state["yaw"],
        agent_state["v"], agent_state["delta_f"]
    )

    #Intial Coordinates
    xi = x
    yi = y
    
    # Convert force vector into desired orientation
    Fx, Fz = force_vector
    psi_d = np.arctan2(Fz, Fx)  # Desired yaw from force field

    # Sliding mode control for yaw
    c_r = 0.5  # Tunable gain
    yaw_error = yaw - psi_d
    s_r = c_r * yaw_error  # Rotational sliding surface
    delta_f = -0.1 * np.sign(s_r)  # Steering control (bounded)

    # Longitudinal control (velocity)
    c_v = 0.3  # Tunable gain
    v_d = np.linalg.norm([Fx, Fz])  # Desired speed
    s_l = c_v * (v - v_d)  # Longitudinal sliding surface
    a = -0.2 * np.sign(s_l)  # Acceleration

    # Apply kinematic bicycle model equations
    Lf, Lr = 0.5, 0.5  # Front and rear axle distances
    beta = np.arctan((Lr * np.tan(delta_f)) / (Lf + Lr))  # Slip angle
    v += a * dt  # Update velocity

    # Compute new position
    x += v * np.cos(yaw + beta) * dt
    y += v * np.sin(yaw + beta) * dt
    yaw += (v / (Lf + Lr)) * np.cos(beta) * np.tan(delta_f) * dt  # Update yaw

    # Update agent state
    agent_state["x"], agent_state["y"], agent_state["yaw"], agent_state["v"], agent_state["delta_f"] = x, y, yaw, v, delta_f

    # **Move AI2-THOR agent**
    # Convert yaw (radians) to degrees
    yaw_degrees = np.rad2deg(yaw)

    # Rotate to the target yaw
    if yaw_degrees > 0:
        controller.step(action="RotateRight", degrees=yaw_degrees)
    elif yaw_degrees < 0:
        controller.step(action="RotateLeft", degrees=abs(yaw_degrees))


    # Compute movement distance
    distance = np.sqrt((x-xi)**2 + (y-yi)**2)

    # Move forward by the computed distance
    controller.step(action="MoveAhead", moveMagnitude=distance)

    print(f"Updated Agent Position: x={x}, y={y}, yaw={yaw}, v={v}, delta_f={delta_f}")


def navigate_to_target(controller, target_position):
    event = controller.step(action="Pass")  # Initialize the first frame
    prev_frame = event.frame  # Store the initial frame
    
    max_attempts = 10
    attempt = 0

    controller.step(action="RotateLeft")  # Example AI2-THOR action
    i=0
    while i<max_iterations:
        event = controller.step(action="Pass")

        curr_frame = event.frame
        diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
        print(f"Frame Difference: {diff}")
        if diff == 0.0:  # If frames are too similar, wait for a better frame
            print("Frames are too similar, skipping optical flow computation")
            controller.step(action="RotateLeft")  # Example AI2-THOR action
            continue

        agent_pos = event.metadata["agent"]["position"]
        print(f"Agent Position: x={agent_pos['x']}, y={agent_pos['y']}, z={agent_pos['z']}")


        # Convert target dictionary to NumPy array
        target_vector = np.array([target_position['x'], target_position['z']])
        agent_position = np.array([event.metadata['agent']['position'][key] for key in ['x', 'z']])
        
        # Compute optical flow

        prev_points, curr_points = compute_optical_flow(prev_frame, curr_frame)
        flow_vectors = curr_points - prev_points if len(prev_points) > 0 else np.array([])

        # Detect Focus of Expansion (FOE)
        while attempt < max_attempts:
            foe = detect_foe(flow_vectors, curr_points)
    
            if np.array_equal(foe, np.array([0, 0])):
                print(f"ERROR: FOE failed {attempt + 1}/{max_attempts} times. Retrying...")
                attempt += 1
                continue
    
            break  # Exit loop if we get valid flow

        if attempt == max_attempts:
            print("CRITICAL ERROR: FOE computation failed too many times. Exiting.")
            exit(1)

        # Compute force vector using visual potential field
        force_vector = compute_potential_field(flow_vectors, foe, target_vector, agent_position)

        # Update previous frame
        prev_frame = curr_frame.copy()

        # Convert to AI2-THOR movement actions
        initial_x = event.metadata["agent"]["position"]["x"]
        initial_y = event.metadata["agent"]["position"]["z"]  # AI2-THOR uses Z for forward movement
        initial_yaw = np.deg2rad(event.metadata["agent"]["rotation"]["y"])  # Convert to radians
        initial_v = 0.0  # Start from rest

        agent_state = dict()
        agent_state = {
            "x": initial_x,
            "y": initial_y,
            "yaw": initial_yaw,
            "v": initial_v,
            "delta_f": 0.0
        }
        move_ego(controller, agent_state, force_vector)
        
        # Stop condition: If the agent reaches close to the target
        if np.linalg.norm(target_vector - agent_position) < 0.2:
            break 
        i=i+1
    if i==100 :
        print("Stopped after 100 iterations")

#main
controller = ai2thor.controller.Controller(scene="FloorPlan1")
target_position = {'x': 2.0, 'z': 3.0}  # Example target
max_iterations = 100

navigate_to_target(controller, target_position)
controller.stop()





