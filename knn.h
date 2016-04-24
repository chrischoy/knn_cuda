void knn_cuda(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);
void extract_cuda(float* activation, int n_batch, int n_channel, int height,
    int width, float* coords, int n_max_coord, int dim_coord, float* extracted_activation);
