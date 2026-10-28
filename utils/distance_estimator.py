import numpy as np

# Outer canthus landmark indices
LEFT_OUTER_CANTHUS = 263
RIGHT_OUTER_CANTHUS = 33

# Anatomical and focal constants
REAL_EYE_WIDTH_CM = 9.5   # Mean adult outer-canthi distance
CALIBRATED_FOCAL = 650     # Standard calibrated focal length for standard webcams

def estimate_distance_cm(landmarks, width, height):
    """Calculates metric distance in cm based on outer eye pixel width."""
    p1 = np.array([landmarks[LEFT_OUTER_CANTHUS].x * width, landmarks[LEFT_OUTER_CANTHUS].y * height])
    p2 = np.array([landmarks[RIGHT_OUTER_CANTHUS].x * width, landmarks[RIGHT_OUTER_CANTHUS].y * height])
    
    pixel_width = np.linalg.norm(p1 - p2)
    if pixel_width == 0:
        return 60.0
    
    distance_cm = (REAL_EYE_WIDTH_CM * CALIBRATED_FOCAL) / pixel_width
    return distance_cm