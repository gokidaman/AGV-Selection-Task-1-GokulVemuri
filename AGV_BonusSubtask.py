import cv2 as cv
import numpy as np

def calculate_dense_optical_flow(prev_gray, gray):
    # Compute dense optical flow manually using frame differencing and gradients
    dx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=5)
    dy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=5)
    dt = gray.astype(np.float32) - prev_gray.astype(np.float32)
    
    flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
    
    mag = np.sqrt(dx**2 + dy**2)
    mag[mag == 0] = 1  # Avoid division by zero
    flow[..., 0] = -dx * dt / mag  # Optical flow in x-direction
    flow[..., 1] = -dy * dt / mag  # Optical flow in y-direction
    
    return flow

def draw_optical_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    mask = np.zeros_like(frame)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude for visualization
    norm_mag = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue based on direction
    hsv[..., 2] = norm_mag.astype(np.uint8)  # Value based on magnitude
    
    output = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return output

cap = cv.VideoCapture('OPTICAL_FLOW - Clip1.mp4')
ret, prev_frame = cap.read()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

# Get video properties
frame_width = int(cap.get(3))  # Width of the frame
frame_height = int(cap.get(4))  # Height of the frame
fps = int(cap.get(cv.CAP_PROP_FPS)) or 30  # Frames per second (default to 30 if unknown)

# Define VideoWriter object (output filename, codec, FPS, frame size)
output_filename = 'output_DenseOpticalFlow.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for AVI (use 'mp4v' for MP4)
out = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = calculate_dense_optical_flow(prev_gray, gray)
    output = draw_optical_flow(frame, flow)
    
    cv.imshow("Dense Optical Flow", output)
    out.write(output)
    prev_gray = gray
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
