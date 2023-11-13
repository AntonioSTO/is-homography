import numpy as np

def my_DLT(pts1,pts2):

  pts1 = pts1.T
  pts1 = np.vstack((pts1, np.ones(pts1.shape[1])))

  pts2 = pts2.T
  pts2 = np.vstack((pts2, np.ones(pts2.shape[1])))

  for i in range(pts1.shape[1]):
    Ai = np.array([[0,0,0, *-pts2[2,i]*pts1[:,i], *pts2[1,i]*pts1[:,i]],
                   [*pts2[2,i]*pts1[:,i], 0,0,0, *-pts2[0,i]*pts1[:,i]]])
    if i == 0:
      A = Ai
    else:
      A = np.vstack((A,Ai))

  print(A)

  U,S,Vt = np.linalg.svd(A)

  h = Vt[-1]
  H_matrix = h.reshape((3,3))

  return H_matrix

def normalize_points(points):

  centroid = np.mean(points, axis=0)

  avg_distance = np.mean(np.linalg.norm(points - centroid, axis=1))

  scale = np.sqrt(2) / avg_distance

  T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

  homogen_pts = np.column_stack((points, np.ones(len(points))))
  norm_pts = np.dot(T, homogen_pts.T).T[:, :2]

  return T, norm_pts

def my_homography(pts1,pts2):

  T1, norm_pts1 = normalize_points(pts1)
  T2, norm_pts2 = normalize_points(pts2)

  H_normalized = my_DLT(norm_pts1, norm_pts2)

  H = np.dot(np.linalg.inv(T2), np.dot(H_normalized, T1))


  return H