#RANSAC Algorithm
import numpy as np
import normDLT

class Ransac():
    
    def __init__(self,NinL,N,dis_threshold):
        self.NinL = NinL  # Number of data points to fit the model
        self.N = N  # Maximum number of iterations
        self.dis_threshold = dis_threshold  # Threshold to determine inliers
        self.inliers = None  #Inliers

        
    def ransac(self, pts1, pts2):
        i = 0
        num_points = len(pts1)
    
        max_inliers = 0
        best_pts1_in = None
        best_pts2_in = None
        
        while i < self.N:
            indices = np.random.choice(num_points, self.NinL, replace=False)
            sample_pts1 = pts1[indices]
            sample_pts2 = pts2[indices]

            H = normDLT.my_homography(sample_pts1, sample_pts2)

            self.inliers = self.find_inliers(pts1, pts2, H)

            if np.sum(self.inliers) > max_inliers:
                max_inliers = np.sum(self.inliers)
                best_pts1_in = pts1[self.inliers]
                best_pts2_in = pts2[self.inliers]

                self.N = self.update_N(pts1,pts2)

            i += 1

        H_final = normDLT.my_homography(best_pts1_in, best_pts2_in)
        print(max_inliers)
        print(i)

        return H_final, best_pts1_in, best_pts2_in

    def find_inliers(self,pts1, pts2, H):

        pts1_homogeneous = np.column_stack((pts1, np.ones(len(pts1))))
        pts2_transformed = np.dot(H, pts1_homogeneous.T).T
        pts2_transformed /= pts2_transformed[:, 2][:, np.newaxis]
        pts2_transformed = pts2_transformed[:, :2]

        distances = np.linalg.norm(pts2 - pts2_transformed, axis=1)

        inliers = distances < self.dis_threshold
        return inliers
    

    def update_N(self,pts1,pts2):
        e = 1 - (np.sum(self.inliers)/len(pts1))
        N = (np.log(1-0.99)//np.log(1-((1-e)**4))) + 1

        #print(N)

        return N
    
    