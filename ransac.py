#RANSAC Algorithm
import numpy as np
import normDLT

class Ransac():
    
    def __init__(self,NinL,N,dis_threshold):
        self.NinL = NinL  # Number of data points to fit the model
        self.N = N  # Maximum number of iterations
        self.dis_threshold = dis_threshold  # Threshold to determine inliers
        
    def ransac(self, pts1, pts2):
        num_points = len(pts1)
    
        max_inliers = 0
        best_pts1_in = None
        best_pts2_in = None
        
        for _ in range(self.N):
            indices = np.random.choice(num_points, self.NinL, replace=False)
            sample_pts1 = pts1[indices]
            sample_pts2 = pts2[indices]

            H = normDLT.my_homography(sample_pts1, sample_pts2)

            inliers = self.find_inliers(pts1, pts2, H)

            if np.sum(inliers) > max_inliers:
                max_inliers = np.sum(inliers)
                best_pts1_in = pts1[inliers]
                best_pts2_in = pts2[inliers]

        H_final = normDLT.my_homography(best_pts1_in, best_pts2_in)

        return H_final, best_pts1_in, best_pts2_in

    def find_inliers(self,pts1, pts2, H):

        pts1_homogeneous = np.column_stack((pts1, np.ones(len(pts1))))
        pts2_transformed = np.dot(H, pts1_homogeneous.T).T
        pts2_transformed /= pts2_transformed[:, 2][:, np.newaxis]
        pts2_transformed = pts2_transformed[:, :2]

        distances = np.linalg.norm(pts2 - pts2_transformed, axis=1)

        inliers = distances < self.dis_threshold
        return inliers