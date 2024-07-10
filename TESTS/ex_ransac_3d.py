import numpy as np
from scipy.spatial import distance
import random
def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points = points - centroid
	furthest_distance = np.max(np.sqrt(np.sum(np.square(np.abs(points)),axis=-1)))
	points = points/furthest_distance

	return points
def euclidean_transform_3D(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)

    # number of points
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # covariance of datasets
    H = np.transpose(AA) * BB

    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    R = Vt.T * U.T
    print('R', R)
    # prinyt(Vt)
    print(Vt)
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        print("sign")
        # thanks to @valeriy.krygin to pointing me on a bug here
        Vt[2, :] *= -1
        R = Vt.T * U.T
        print('new R', R)

    t = -R * centroid_A.T + centroid_B.T

    return R, t


def test():
    R = np.mat(np.random.rand(3, 3))
    t = np.mat(np.random.rand(3, 1))

    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

    if np.linalg.det(R) < 0:
        print('sign')
        Vt[2, :] *= -1
        R = U * Vt
    n = 10

    A = np.mat(np.random.rand(n, 3));
    B = R * A.T + np.tile(t, (1, n))
    B = B.T;

    # recover the transformation
    Rc, tc = euclidean_transform_3D(A, B)

    A_transformed = (Rc * A.T) + np.tile(tc, (1, n))
    A_transformed = A_transformed.T

    # Find the error
    rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
    print("RMSE:", rmse)


#test()

R = np.mat(np.random.rand(3, 3))
t = np.mat(np.random.rand(3, 1))

U, S, Vt = np.linalg.svd(R)
R = U * Vt

if np.linalg.det(R) < 0:
    print('sign')
    Vt[2, :] *= -1
    R = U * Vt
n = 10

A = np.mat(np.random.rand(n, 3));
B = R * A.T + np.tile(t, (1, n))
B = B.T;

# recover the transformation
Rc, tc = euclidean_transform_3D(A, B)

A_transformed = (Rc * A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T

# Find the error
rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
print("RMSE:", rmse)

# Sign case
#
# A = np.mat([[0,0,1],
#               [0,0,-1]])
# B = np.mat([[0,0,-1],
#               [0,0, 1]])
# r,tk = euclidean_transform_3D(A,B)
# r[0,0] = 1
# A_transformed = (r.dot(A.T)) + np.tile(tk, (1, 2))
# A_transformed = A_transformed.T
#
# # Find the error
# rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
# print ("RMSE:", rmse)