import numpy as np
import knn
import cv2

bf = cv2.BFMatcher()

n = 48
m = 8192
dim = 128
k = 2

query = np.random.rand(dim, n)
# query = np.arange(n)
# query = np.array([query, query]).T
query = query.astype(np.float32)

ref = np.random.rand(dim, n)
# ref = 3 * np.arange(m)
# ref = np.array([ref, ref]).T
ref = ref.astype(np.float32)

dist, ind = knn.knn(query, ref, k)

# print query.T
# print ref.T
print dist.T
print ind.T - 1

matches = bf.knnMatch(query.T, ref.T, k)
for ms, ds, inds in zip(matches, dist.T, ind.T):
    m, n = ms
    print m.distance, n.distance, ds
    print m.trainIdx, n.trainIdx, inds - 1
