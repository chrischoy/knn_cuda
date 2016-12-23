import numpy as np
import knn

c = 128

for n in range(4):
    query = np.random.rand(c, 1000).astype(np.float32)

    reference = np.random.rand(c, 4000).astype(np.float32)

    # Index is 1-based
    dist, ind = knn.knn(query.reshape(c, -1),
                        reference.reshape(c, -1), 2)

    print ind
