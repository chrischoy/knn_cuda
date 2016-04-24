import numpy as np
import knn
import cv2

bf = cv2.BFMatcher()

n = 3
c = 2
height = 4
width = 2

n_corr = 4
k = 2

activation1 = np.random.rand(n, c, height, width)
activation1 = activation1.astype(np.float32)

activation2 = np.random.rand(n, c, height, width)
activation2 = activation2.astype(np.float32)

correspondences1 = np.random.rand(n, n_corr, 3)
correspondences1 = correspondences1.astype(np.float32)
correspondences1[:, :, 0] = np.tile(np.arange(n).reshape(n, 1), [1, n_corr])
correspondences1[:, :, 1] *= width
correspondences1[:, :, 2] *= height

correspondences2 = np.random.rand(n, n_corr, 3)
correspondences2 = correspondences2.astype(np.float32)
correspondences2[:, :, 0] = np.tile(np.arange(n).reshape(n, 1), [1, n_corr])
correspondences2[:, :, 1] *= width
correspondences2[:, :, 2] *= height

queries = knn.extract(activation1, correspondences1)
refs = knn.extract(activation2, correspondences2)

print queries


# for query, ref in zip(queries, refs):
#     dist, ind = knn.knn(query, ref, k)
# 
#     print query.T
#     print ref.T

    # matches = bf.knnMatch(query.T, ref.T, k)
    # for ms, ds, inds in zip(matches, dist.T, ind.T):
    #     m, n = ms
    #     print m.distance, n.distance, ds
    #     print m.trainIdx, n.trainIdx, inds - 1
