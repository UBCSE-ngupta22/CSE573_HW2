import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Rotation matrix around z-axis for alpha
    Rz_alpha = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad), 0],
        [np.sin(alpha_rad), np.cos(alpha_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix around x'-axis for beta
    Rx_beta = np.array([
        [1, 0, 0],
        [0, np.cos(beta_rad), -np.sin(beta_rad)],
        [0, np.sin(beta_rad), np.cos(beta_rad)]
    ])

    # Rotation matrix around z''-axis for gamma
    Rz_gamma = np.array([
        [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
        [np.sin(gamma_rad), np.cos(gamma_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix is Rz_gamma * Rx_beta * Rz_alpha
    rot_xyz2XYZ = Rz_gamma @ Rx_beta @ Rz_alpha

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation

    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Rotation matrix around z''-axis for gamma
    Rz_gamma = np.array([
        [np.cos(gamma_rad), np.sin(gamma_rad), 0],
        [-np.sin(gamma_rad), np.cos(gamma_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix around x'-axis for beta
    Rx_beta = np.array([
        [1, 0, 0],
        [0, np.cos(beta_rad), np.sin(beta_rad)],
        [0, -np.sin(beta_rad), np.cos(beta_rad)]
    ])

    # Rotation matrix around z-axis for alpha
    Rz_alpha = np.array([
        [np.cos(alpha_rad), np.sin(alpha_rad), 0],
        [-np.sin(alpha_rad), np.cos(alpha_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix is Rz_alpha * Rx_beta * Rz_gamma
    rot_XYZ2xyz = Rz_alpha @ Rx_beta @ Rz_gamma
    
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1



#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation

    global clicked_points
    clicked_points = []  # Reset the list for a new image

    # Display the image and set the mouse callback to record clicks
    print("Please click on exactly 32 corner points on the image.")
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', mouse_callback)

    # Loop until exactly 32 points have been selected
    while len(clicked_points) < 32:
        cv2.imshow('Select Points', image)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Process aborted.")
            sys.exit(1)

    cv2.destroyAllWindows()

    # Convert to numpy array for further calculations
    img_coord = np.array(clicked_points, dtype=np.float32)

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation

    # Left Side (first 16 points)
    world_coord[0] = [0, 40, 40]
    world_coord[1] = [0, 30, 40]
    world_coord[2] = [0, 20, 40]
    world_coord[3] = [0, 10, 40]
    world_coord[4] = [0, 40, 30]
    world_coord[5] = [0, 30, 30]
    world_coord[6] = [0, 20, 30]
    world_coord[7] = [0, 10, 30]
    world_coord[8] = [0, 40, 20]
    world_coord[9] = [0, 30, 20]
    world_coord[10] = [0, 20, 20]
    world_coord[11] = [0, 10, 20]
    world_coord[12] = [0, 40, 10]
    world_coord[13] = [0, 30, 10]
    world_coord[14] = [0, 20, 10]
    world_coord[15] = [0, 10, 10]

    # Right Side (next 16 points)
    world_coord[16] = [10, 0, 40]
    world_coord[17] = [20, 0, 40]
    world_coord[18] = [30, 0, 40]
    world_coord[19] = [40, 0, 40]
    world_coord[20] = [10, 0, 30]
    world_coord[21] = [20, 0, 30]
    world_coord[22] = [30, 0, 30]
    world_coord[23] = [40, 0, 30]
    world_coord[24] = [10, 0, 20]
    world_coord[25] = [20, 0, 20]
    world_coord[26] = [30, 0, 20]
    world_coord[27] = [40, 0, 20]
    world_coord[28] = [10, 0, 10]
    world_coord[29] = [20, 0, 10]
    world_coord[30] = [30, 0, 10]
    world_coord[31] = [40, 0, 10]

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation

    mat_img_world = []
    for i in range(len(img_coord)):
        X, Y, Z = world_coord[i]
        u, v = img_coord[i]
        
        mat_img_world.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        mat_img_world.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])

    _, _, vh = np.linalg.svd(np.array(mat_img_world))
    proj_matrix = vh[-1].reshape(3, 4)

    M = proj_matrix[:, :3]
    
    _, intrinsic_matrix = np.linalg.qr(np.linalg.inv(M))
    intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

    intrinsic_matrix_diag = np.diag(intrinsic_matrix)

    intrinsic_matrix = np.dot(intrinsic_matrix, np.diag(np.sign(intrinsic_matrix_diag)))

    intrinsic_matrix /= intrinsic_matrix[-1, -1]
    
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation

    mat_img_world = []
    for i in range(len(img_coord)):
        X, Y, Z = world_coord[i]
        u, v = img_coord[i]
        
        mat_img_world.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        mat_img_world.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])

    _, _, vh = np.linalg.svd(np.array(mat_img_world))
    proj_matrix = vh[-1].reshape(3, 4)
    
    M = proj_matrix[:, :3]
    rot_matrix, intrinsic_matrix = np.linalg.qr(np.linalg.inv(M))
    intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

    intrinsic_matrix_diag = np.diag(intrinsic_matrix)
    intrinsic_matrix = np.diag(np.sign(intrinsic_matrix_diag))

    intrinsic_matrix /= intrinsic_matrix[-1, -1]
    
    R = np.linalg.inv(rot_matrix)
    T = np.dot(np.linalg.inv(intrinsic_matrix), proj_matrix[:, 3])

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2

import sys
# Global list to store clicked points
clicked_points = []

# Mouse callback function to store the coordinates of clicked points
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 32:
            # Add the clicked point to the list
            clicked_points.append((x, y))
            print(f"Point recorded: ({x}, {y})")
        if len(clicked_points) == 32:
            print("32 points have been recorded.")

#---------------------------------------------------------------------------------------------------------------------