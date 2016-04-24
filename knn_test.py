import numpy as np
import knn
import cv2

bf = cv2.BFMatcher()

n = 4
c = 2
height = 5
width = 3

n_corr = 6
k = 2

activation1 = np.random.rand(n, c, height, width)
activation1 = activation1.astype(np.float32)
activation1[:, 0, :, :] *= -1

activation2 = np.random.rand(n, c, height, width)
activation2 = activation2.astype(np.float32)

correspondences1 = np.random.rand(n, n_corr, 2)
correspondences1 = correspondences1.astype(np.float32)
# correspondences1[:, :, 0] = np.tile(np.arange(n).reshape(n, 1), [1, n_corr])
correspondences1[:, :, 0] *= (width - 1)
correspondences1[:, :, 1] *= (height - 1)

correspondences2 = np.random.rand(n, n_corr, 2)
correspondences2 = correspondences2.astype(np.float32)
# correspondences2[:, :, 0] = np.tile(np.arange(n).reshape(n, 1), [1, n_corr])
correspondences2[:, :, 0] *= (width - 1)
correspondences2[:, :, 1] *= (height - 1)

queries = knn.extract(activation1, correspondences1)
refs = knn.extract(activation2, correspondences2)

for i in range(n):
    print i
    print activation1[i]
    print correspondences1[i]
    print queries[i]

for query, ref in zip(queries, refs):
    dist, ind = knn.knn(query, ref, k)

    print 'Query'
    print query.T
    print 'Ref'
    print ref.T

    matches = bf.knnMatch(query.T, ref.T, k)
    for ms, ds, inds in zip(matches, dist.T, ind.T):
        m, n = ms
        print m.distance, n.distance, ds
        print m.trainIdx, n.trainIdx, inds - 1
