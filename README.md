# KNN CUDA Wrapper

This repository provides a python wrapper for the [kNN-CUDA library](https://github.com/vincentfpgarcia/kNN-CUDA).

# Installation

Please modify the `Makefile.config` to make sure all the dependencies are set correctly.

```
git clone https://github.com/chrischoy/knn_cuda.git
cd knn_cuda
make
```

# Example

Once you build the wrapper, run

```
python example.py
[[3367 2785 1523 ..., 1526  569 3616]
 [1929 3353  339 ...,  690  463 2972]]
[[3413 3085 1528 ...,  608 2258  733]
 [1493 3849 1616 ...,  743 2012 1786]]
[[2446 3320 2379 ..., 2718  598 1854]
 [1348 3857 1393 ..., 3258 1642 3436]]
[[3044 2604 3972 ..., 3968 1710 2916]
 [ 812 1090  355 ...,  699 3231 2302]]
```

# Usage

In python, after you `import knn`, you can access the knn function.

## knn.knn(query_points, reference_points, K)

Both query_points and reference_points must be numpy arrays with float32 format.
For both query and reference, the first dimension is the dimension of the vector and the second dimension is the number of vectors.

K is the number of nearest neighbors.

For each vector in the query_points, the function returns the 1-based indices of the K nearest neighbors.

# Warning

The returned index is 1-base.
