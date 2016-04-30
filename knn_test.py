import numpy as np
import knn
import cv2

bf = cv2.BFMatcher()

c = 128

for n in range(100):
    activation1 = np.random.rand(c, 3000)
    activation1 = activation1.astype(np.float32)

    activation2 = np.random.rand(c, 3000)
    activation2 = activation2.astype(np.float32)

    # Index is 1-based
    dist, ind = knn.knn(activation1.reshape(c, -1), activation2.reshape(c, -1), 1)

    print ind
