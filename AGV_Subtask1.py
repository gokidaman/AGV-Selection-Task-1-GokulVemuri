import cv2 as cv
import numpy as np

def lucas_kanade_optical_flow_pyramidal(prev_img, next_img, features, window_size=5, levels=4, scale=0.5):
    pyr_prev = [prev_img]
    pyr_next = [next_img]
    
    # generate image pyramids
    for m in range(1, levels):
        pyr_prev.append(cv.resize(pyr_prev[-1], None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR))
        pyr_next.append(cv.resize(pyr_next[-1], None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR))
    
    # scale features for the coarsest level
    features = features * (scale ** (levels-1))
    
    for level in range(levels - 1, -1, -1):
        img1, img2 = pyr_prev[level], pyr_next[level]
        flow_vectors = lucas_kanade_optical_flow(img1, img2, features, window_size)
        
        if level > 0:
            features = (features + flow_vectors) / scale  # upscale feature points
    
    return flow_vectors


def lucas_kanade_optical_flow(prev_img, next_img, features, window_size=5):
    Ix = cv.Sobel(prev_img, cv.CV_64F, 1, 0, ksize=3)  # X gradient
    Iy = cv.Sobel(prev_img, cv.CV_64F, 0, 1, ksize=3)  # Y gradient
    It = next_img.astype(np.float32) - prev_img.astype(np.float32)  # temporal gradient

    half_win = window_size // 2
    flow_vectors = []

    for i in features: 
        x = int(i[0])
        y = int(i[1])

        if x - half_win < 0 or x + half_win >= prev_img.shape[1] or y - half_win < 0 or y + half_win >= prev_img.shape[0]:
            flow_vectors.append((0, 0))
            continue

        # extract local window
        Ix_window = Ix[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()
        Iy_window = Iy[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()
        It_window = -It[y-half_win:y+half_win+1, x-half_win:x+half_win+1].flatten()

        A = np.vstack((Ix_window, Iy_window)).T  # matrix A
        b = It_window  # matrix b

        # solve for (u, v) using least squares
        if A.shape[0] >= 2 and np.linalg.matrix_rank(A) == 2:
            flow = np.linalg.pinv(A) @ b  # pseudo-inverse
            flow_vectors.append((flow[0], flow[1]))
        else:
            flow_vectors.append((0, 0))

    return np.array(flow_vectors)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("OPTICAL_FLOW - Clip1.mp4")
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
features = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
features = np.float32(features).reshape(-1, 2)
mask = np.zeros_like(first_frame)  # Create mask for optical flow visualization

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, next_frame = cap.read()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
    features = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    features = np.float32(features).reshape(-1, 2)
    # Calculates sparse optical flow by Lucas-Kanade method
    flow_vectors = lucas_kanade_optical_flow_pyramidal(prev_gray, next_gray, features)

    for i, (x, y) in enumerate(features):
        u, v = flow_vectors[i]
        x, y, u, v = map(int, [x, y, u, v])
        # Draw optical flow tracks
        mask = cv.line(mask, (x, y), (x + u, y + v), color, 2)
        next_frame = cv.circle(next_frame, (x+u, y+v), 3, color, -1)
    
    output = cv.add(next_frame, mask)
    prev_gray = next_gray.copy()
    features = features.reshape(-1, 1, 2)

    cv.imshow("Pyramidal Lucas-Kanade Optical Flow", output)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()