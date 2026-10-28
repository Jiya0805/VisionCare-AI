import numpy as np

# MediaPipe FaceMesh indices for left and right eyes
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def calculate_single_ear(landmarks, eye_indices, width, height):
    """Calculates EAR for a single eye given 6 contour landmarks."""
    pts = [np.array([landmarks[idx].x * width, landmarks[idx].y * height]) for idx in eye_indices]
    
    # Vertical distances
    d_v1 = np.linalg.norm(pts[1] - pts[5])
    d_v2 = np.linalg.norm(pts[2] - pts[4])
    
    # Horizontal distance
    d_h = np.linalg.norm(pts[0] - pts[3])
    
    if d_h == 0:
        return 0.30
    return (d_v1 + d_v2) / (2.0 * d_h)

def calculate_avg_ear(landmarks, width, height):
    """Returns the mean EAR across both eyes."""
    l_ear = calculate_single_ear(landmarks, LEFT_EYE_INDICES, width, height)
    r_ear = calculate_single_ear(landmarks, RIGHT_EYE_INDICES, width, height)
    return (l_ear + r_ear) / 2.0