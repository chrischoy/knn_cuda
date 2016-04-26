/**
  *
  * Date         03/07/2009
  * ====
  *
  * Authors      Vincent Garcia
  * =======      Eric    Debreuve
  *              Michel  Barlaud
  *
  * Description  Given a reference point set and a query point set, the program returns
  * ===========  firts the distance between each query point and its k nearest neighbors in
  *              the reference point set, and second the indexes of these k nearest neighbors.
  *              The computation is performed using the API NVIDIA CUDA.
  *
  * Paper        Fast k nearest neighbor search using GPU
  * =====
  *
  * BibTeX       @INPROCEEDINGS{2008_garcia_cvgpu,
  * ======         author = {V. Garcia and E. Debreuve and M. Barlaud},
  *                title = {Fast k nearest neighbor search using GPU},
  *                booktitle = {CVPR Workshop on Computer Vision on GPU},
  *                year = {2008},
  *                address = {Anchorage, Alaska, USA},
  *                month = {June}
  *              }
  *
  */

// Includes
#include <cstdio>
#include "cuda.h"

// Constants used by the program
#define BLOCK_DIM                      16


//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//
__global__ void extract_with_interpolation(
    int nthreads,
    float *data, float *n_xy_coords, float *extracted_data,
    int n_max_coord, int channels, int height, int width) {

  int x0, x1, y0, y1, nc;
  float wx0, wx1, wy0, wy1;
  int n, nd;
  float x, y;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < (nthreads);
       index += blockDim.x * gridDim.x) {
    n = (index / n_max_coord);
    nd = n * n_max_coord * channels;
    x = n_xy_coords[index * 2];
    y = n_xy_coords[index * 2 + 1];

    x0 = static_cast<int>(floor(x));
    x1 = x0 + 1;
    y0 = static_cast<int>(floor(y));
    y1 = y0 + 1;

    x0 = x0 <= 0 ? 0 : (x0 >= (width - 1)  ? (width - 1) : x0);
    y0 = y0 <= 0 ? 0 : (y0 >= (height - 1) ? (height - 1) : y0);
    x1 = x1 <= 0 ? 0 : (x1 >= (width - 1)  ? (width - 1) : x1);
    y1 = y1 <= 0 ? 0 : (y1 >= (height - 1) ? (height - 1) : y1);

    wx0 = static_cast<float>(x1) - x;
    wx1 = x - x0;
    wy0 = static_cast<float>(y1) - y;
    wy1 = y - y0;

    if(x0 == x1){ wx0 = 1; wx1 = 0; }
    if(y0 == y1){ wy0 = 1; wy1 = 0; }
    for(int c=0; c < channels; c++) {
      nc = (n * channels + c) * height;
      // extracted_data[index * channels + c] = wy0 * wx0 * data[(nc + y0) * width + x0]
      // extracted_data[nd + index % n_max_coord + n_max_coord * c] = index;
      extracted_data[nd + index % n_max_coord + n_max_coord * c] = wy0 * wx0 * data[(nc + y0) * width + x0]
       + wy1 * wx0 * data[(nc + y1) * width + x0]
       + wy0 * wx1 * data[(nc + y0) * width + x1]
       + wy1 * wx1 * data[(nc + y1) * width + x1];
    }
  }
}

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param pA    pitch of matrix A given in number of columns
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param pB    pitch of matrix B given in number of columns
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal( float* A, int wA, int pA,
    float* B, int wB, int pB, int dim,  float* AB){

  // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  float tmp;
  float ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * pA;
  step_B  = BLOCK_DIM * pB;
  end_A   = begin_A + (dim-1) * pA;

    // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
  int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix

  // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
    if (a/pA + ty < dim){
      shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0;
    }
    else{
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
    if (cond2 && cond1){
      for (int k = 0; k < BLOCK_DIM; ++k){
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp*tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1)
    AB[(begin_A + ty) * pB + begin_B + tx] = ssd;
}


/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param dist_pitch  pitch of the distance matrix given in number of columns
  * @param ind         index matrix
  * @param ind_pitch   pitch of the index matrix given in number of columns
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
__global__ void cuInsertionSort(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k){

  // Variables
  int l, i, j;
  float *p_dist;
  int   *p_ind;
  float curr_dist, max_dist;
  int   curr_row,  max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (xIndex<width){
    // Pointer shift, initialization, and max value
    p_dist   = dist + xIndex;
    p_ind    = ind  + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 1;

    // Part 1 : sort kth firt elementZ
    for (l=1; l<k; l++){
      curr_row  = l * dist_pitch;
      curr_dist = p_dist[curr_row];
      if (curr_dist<max_dist){
        i=l-1;
        for (int a=0; a<l-1; a++){
          if (p_dist[a*dist_pitch]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=l; j>i; j--){
          p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
          p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
        }
        p_dist[i*dist_pitch] = curr_dist;
        p_ind[i*ind_pitch]   = l+1;
      } else {
        p_ind[l*ind_pitch] = l+1;
      }
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k-1)*dist_pitch;
    for (l=k; l<height; l++){
      curr_dist = p_dist[l*dist_pitch];
      if (curr_dist<max_dist){
        i=k-1;
        for (int a=0; a<k-1; a++){
          if (p_dist[a*dist_pitch]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=k-1; j>i; j--){
          p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
          p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
        }
        p_dist[i*dist_pitch] = curr_dist;
        p_ind[i*ind_pitch]   = l+1;
        max_dist             = p_dist[max_row];
      }
    }
  }
}


/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param pitch   pitch of the distance matrix given in number of columns
  * @param k       number of neighbors to consider
  */
__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (xIndex<width && yIndex<k)
    dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}


//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//


/**
  * Prints the error message return during the memory allocation.
  *
  * @param error        error value return by the memory allocation function
  * @param memorySize   size of memory tried to be allocated
  */
void printErrorMessage(cudaError_t error, int memorySize){
  printf("==================================================\n");
  printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
  printf("Whished allocated memory : %d\n", memorySize);
  printf("==================================================\n");
}


/**
  * Feature extraction algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
  *
  */
void extract_cuda(float* activation, int n_batch, int n_channel, int height,
    int width, float* coords, int n_max_coord, int dim_coord,
    float *extracted_activation){
  // activation n_batch x n_channel x height x width
  // coords n_batch x n_max_coord x dim_coord
  // uninitialized empty pointer which will be filled with extracted_activation
  // n_batch x n_channel x n_max_coord. KNN requires dim x n_feature format
  unsigned int size_of_float = sizeof(float);

  // Variables
  float *activation_device;
  float *coord_device;
  float *extracted_activation_device;

  // CUDA Initialisation
  cuInit(0);

  // Allocation of global memory for query points and for distances, CUDA_CHECK
  cudaMalloc((void **) &activation_device,
      n_batch * n_channel * height * width * size_of_float);
  cudaMalloc((void **) &extracted_activation_device,
      n_batch * n_channel * n_max_coord * size_of_float);
  cudaMalloc((void **) &coord_device,
      n_batch * n_max_coord * dim_coord * size_of_float);

  // Grids ans threads
  dim3 g_size_r((n_batch * n_max_coord * dim_coord) / 256, 1, 1);
  dim3 t_size_r(256, 1, 1);
  if ((n_batch * n_max_coord * dim_coord) % 256 != 0) g_size_r.x += 1;

  cudaMemset(extracted_activation_device, 0, n_batch * n_channel * n_max_coord * size_of_float);

  // Copy coordinates to the device
  cudaMemcpy(coord_device, &coords[0],
     n_batch * n_max_coord * dim_coord * size_of_float,
     cudaMemcpyHostToDevice);

  // Copy of part of query actually being treated
  cudaMemcpy(activation_device, &activation[0],
      n_batch * n_channel * height * width * size_of_float,
      cudaMemcpyHostToDevice);

  // Grids ans threads
  dim3 g_size((n_batch * n_max_coord) / 256, 1, 1);
  dim3 t_size(256, 1, 1);
  if ((n_batch * n_max_coord) % 256 != 0) g_size.x += 1;

  extract_with_interpolation<<<g_size, t_size>>>(n_batch * n_max_coord,
    activation_device, coord_device, extracted_activation_device,
    n_max_coord, n_channel, height, width);

  // Memory copy of output from device to host
  cudaMemcpy(extracted_activation, &extracted_activation_device[0],
      n_batch * n_channel * n_max_coord * size_of_float,
      cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(coord_device);
  cudaFree(activation_device);
  cudaFree(extracted_activation_device);
}

/**
  * K nearest neighbor algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
  *
  */
void knn_cuda(float* ref_host, int ref_width, float* query_host,
    int query_width, int height, int k, float* dist_host, int* ind_host){

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int   = sizeof(int);

  // Variables
  float  *query_dev;
  float  *ref_dev;
  float  *dist_dev;
  int    *ind_dev;
  size_t query_pitch;
  size_t query_pitch_in_bytes;
  size_t ref_pitch;
  size_t ref_pitch_in_bytes;
  size_t ind_pitch;
  size_t ind_pitch_in_bytes;

  // CUDA Initialisation
  cuInit(0);

  // Allocation of global memory for query points and for distances, CUDA_CHECK
  cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes,
      query_width * size_of_float, height + ref_width);

  query_pitch = query_pitch_in_bytes/size_of_float;
  dist_dev    = query_dev + height * query_pitch;

  // Allocation of global memory for indexes CUDA_CHECK
  cudaMallocPitch((void **) &ind_dev, &ind_pitch_in_bytes,
      query_width * size_of_int, k);

  ind_pitch = ind_pitch_in_bytes / size_of_int;

  // Allocation of global memory CUDA_CHECK
  cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes,
      ref_width * size_of_float, height);

  ref_pitch = ref_pitch_in_bytes/size_of_float;
  cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host,
      ref_width*size_of_float, ref_width*size_of_float, height,
      cudaMemcpyHostToDevice);

  // Copy of part of query actually being treated
  cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[0],
      query_width*size_of_float, query_width*size_of_float, height,
      cudaMemcpyHostToDevice);

  // Grids ans threads
  dim3 g_16x16(query_width/16, ref_width/16, 1);
  dim3 t_16x16(16, 16, 1);
  if (query_width%16 != 0) g_16x16.x += 1;
  if (ref_width  %16 != 0) g_16x16.y += 1;
  //
  dim3 g_256x1(query_width/256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_width%256 != 0) g_256x1.x += 1;

  dim3 g_k_16x16(query_width/16, k/16, 1);
  dim3 t_k_16x16(16, 16, 1);
  if (query_width%16 != 0) g_k_16x16.x += 1;
  if (k  %16 != 0) g_k_16x16.y += 1;

  // Kernel 1: Compute all the distances
  cuComputeDistanceGlobal<<<g_16x16,t_16x16>>>(ref_dev, ref_width,
      ref_pitch, query_dev, query_width, query_pitch, height,
      dist_dev);

  // Kernel 2: Sort each column
  cuInsertionSort<<<g_256x1,t_256x1>>>(dist_dev, query_pitch, ind_dev,
      ind_pitch, query_width, ref_width, k);

  // Kernel 3: Compute square root of k first elements
  cuParallelSqrt<<<g_k_16x16,t_k_16x16>>>(dist_dev, query_width, query_pitch, k);

  // Memory copy of output from device to host
  cudaMemcpy2D(&dist_host[0], query_width*size_of_float, dist_dev,
      query_pitch_in_bytes, query_width*size_of_float, k,
      cudaMemcpyDeviceToHost);

  cudaMemcpy2D(&ind_host[0],  query_width*size_of_int,   ind_dev,
      ind_pitch_in_bytes,   query_width*size_of_int,   k,
      cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(ref_dev);
  cudaFree(ind_dev);
  cudaFree(query_dev);
}


/**
  * Example of use of kNN search CUDA.
  */
int main(void){
  // Variables and parameters
  float* ref;                 // Pointer to reference point array
  float* query;               // Pointer to query point array
  float* dist;                // Pointer to distance array
  int*   ind;                 // Pointer to index array
  int    ref_nb     = 4096;   // Reference point number, max=65535
  int    query_nb   = 4096;   // Query point number,     max=65535
  int    dim        = 32;     // Dimension of points
  int    k          = 20;     // Nearest neighbors to consider
  int    iterations = 100;
  int    i;

  // Memory allocation
  ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
  query  = (float *) malloc(query_nb * dim * sizeof(float));
  dist   = (float *) malloc(query_nb * k * sizeof(float));
  ind    = (int *)   malloc(query_nb * k * sizeof(float));

  // Init 
  srand(time(NULL));
  for (i=0 ; i<ref_nb   * dim ; i++) ref[i]    = (float)rand() / (float)RAND_MAX;
  for (i=0 ; i<query_nb * dim ; i++) query[i]  = (float)rand() / (float)RAND_MAX;

  // Variables for duration evaluation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time;

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb  );
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim     );
  printf("Number of neighbors to consider : %4d\n", k       );
  printf("Processing kNN search           :"                );

  // Call kNN search CUDA
  cudaEventRecord(start, 0);
  for (i=0; i<iterations; i++) {
    knn_cuda(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf(" done in %f s for %d iterations (%f s by iteration)\n",
      elapsed_time/1000, iterations, elapsed_time/(iterations*1000));

  // Destroy cuda event object and free memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(ind);
  free(dist);
  free(query);
  free(ref);
}
