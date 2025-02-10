import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def load_npy(filename):
    with open(filename, 'rb') as f:
        a = np.load(f)

        b = np.load(f)

    return a, b

def get_distance_3D(p1,p2):
    squared_dist = np.sum(np.square(p1 - p2))
    distance = np.sqrt(squared_dist)
    return distance
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
def euclidean_transform_3D_2(A, B):
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
    H = np.transpose(AA) @ BB

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
da = distance.cdist(A,A)
db = distance.cdist(B,B)
Ap = np.asarray(A)
Bp = np.asarray(B)
A1p, B1p = load_npy('test0.npy')
A1 = np.mat(A1p)
B1 = np.mat(B1p)
np.zeros_like(A)

da1 = distance.cdist(A1,A1)
db1 = distance.cdist(B1,B1)
print('g',A.shape,B.shape)
print('r', A1.shape, B1.shape)
for i in range(10):
    print('da', np.max(da[i]))
    print('db', np.max(db[i]))
    print('da1', np.max(da1[i]))
    print('db1', np.max(db1[i]))

# recover the transformation
#Rc, tc = euclidean_transform_3D(A, B)

#A_transformed = (Rc * A.T) + np.tile(tc, (1, n))
#A_transformed = A_transformed.T

# Find the error
#rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
#print("RMSE:", rmse)

# Split the data into x, y, and z arrays
xa = Ap[:, 0]
ya = Ap[:, 1]
za = Ap[:, 2]

xb = Bp[:, 0]
yb = Bp[:, 1]
zb = Bp[:, 2]

xa1 = A1p[:, 0]
ya1 = A1p[:, 1]
za1 = A1p[:, 2]

xb1 = B1p[:, 0]
yb1 = B1p[:, 1]
zb1 = B1p[:, 2]


# Create a 3D figure
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xa,ya,za,color='blue')
ax.scatter(xb,yb,zb,color='red')
for i in range(A.shape[0]):
    x = np.asarray([xa[i], xb[i]])
    y = np.asarray([ya[i], yb[i]])
    z = np.asarray([za[i], zb[i]])
    print('x',x)
    ax.plot(x,y,z, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')


# Create a 3D figure
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xa1,ya1,za1,color='blue')
ax.scatter(xb1,yb1,zb1,color='red')
for i in range(A.shape[0]):
    x = np.asarray([xa1[i], xb1[i]])
    y = np.asarray([ya1[i], yb1[i]])
    z = np.asarray([za1[i], zb1[i]])
    print('x',x)
    ax.plot(x,y,z, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
plt.show()

# recover the transformation
Rc, tc = euclidean_transform_3D(A, B)

A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T

Rc1, tc1 = euclidean_transform_3D(A1, B1)

A1_transformed = (Rc1*A1.T) + np.tile(tc1, (1, n))
A1_transformed = A1_transformed.T

A_transformed_p = np.asarray(A_transformed)

A1_transformed_p = np.asarray(A1_transformed)


# Split the data into x, y, and z arrays
xat = A_transformed_p[:, 0]
yat = A_transformed_p[:, 1]
zat = A_transformed_p[:, 2]

xbt = Bp[:, 0]
ybt = Bp[:, 1]
zbt = Bp[:, 2]

xa1t = A1_transformed_p[:, 0]
ya1t = A1_transformed_p[:, 1]
za1t = A1_transformed_p[:, 2]

xb1t = B1p[:, 0]
yb1t = B1p[:, 1]
zb1t = B1p[:, 2]

# Create a 3D figure
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xat,yat,zat,color='blue')
ax.scatter(xbt,ybt,zbt,color='red')
for i in range(A.shape[0]):
    xt = np.asarray((xat[0], xbt[0]))
    yt = np.asarray((yat[0], ybt[0]))
    zt = np.asarray((zat[0], zbt[0]))
    print('xt', xt)
    print('yt', yt)
    print('zt', zt)

    ax.plot(xt,yt,zt, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')


# Create a 3D figure
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xa1t,ya1t,za1t,color='blue')

ax.scatter(xb1t,yb1t,zb1t,color='red')

# xa1t = xa1t.T
# ya1t = ya1t.T
# za1t = za1t.T
#
# xb1t = xb1t.T
# yb1t = yb1t.T
# zb1t = zb1t.T

for i in range(A1p.shape[0]):
    x1t = np.asarray((xa1t[0], xb1t[0]))
    y1t = np.asarray((ya1t[0], yb1t[0]))
    z1t = np.asarray((za1t[0], zb1t[0]))
    print('xt', xt)
    print('yt', yt)
    print('zt', zt)
    print('i',i)
    ax.plot(x1t,y1t,z1t, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
plt.show()
