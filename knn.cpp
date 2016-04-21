// Python
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "knn.h"

using namespace boost::python;

// Takes
object knn(PyObject* query_points_, PyObject* ref_points_, int k)
{
  PyArrayObject* query_points = (PyArrayObject*) query_points_;
  PyArrayObject* ref_points   = (PyArrayObject*) ref_points_;
  int n_query = query_points->dimensions[1];
  int n_ref   = ref_points->dimensions[1];
  int dim     = query_points->dimensions[0];
  // float* query_points_c = (float*)((PyArrayObject_fields *)(query_points_))->data;
  // float* ref_points_c   = (float*)((PyArrayObject_fields *)(ref_points_))->data;
  float* query_points_c = new float[n_query * dim];
  float* ref_points_c   = new float[n_ref * dim];
  float* dist = new float[n_query * k];
  int* ind    = new int[n_query * k];

  // Copy python objects
  for(int i = 0; i < n_query; i++) {
    for(int j = 0; j < dim; j++) {
      query_points_c[n_query * j + i] = *(float*)PyArray_GETPTR2(query_points, j, i);
    }
  }

  for(int i = 0; i < n_ref; i++) {
    for(int j = 0; j < dim; j++) {
      ref_points_c[n_ref * j + i] = *(float*)PyArray_GETPTR2(ref_points, j, i);
    }
  }

  knn_cuda(ref_points_c, n_ref, query_points_c, n_query, dim, k, dist, ind);

  npy_intp dims[2] = {k, n_query};
  PyObject* py_obj_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dist);
  PyObject* py_obj_ind  = PyArray_SimpleNewFromData(2, dims, NPY_INT, ind);
  handle<> handle_dist(py_obj_dist);
  handle<> handle_ind(py_obj_ind);

  numeric::array arr_dist(handle_dist);
  numeric::array arr_ind(handle_ind);

  free(query_points_c);
  free(ref_points_c);
  free(dist);
  free(ind);

  return make_tuple(arr_dist.copy(), arr_ind.copy());
}

BOOST_PYTHON_MODULE(knn)
{
  import_array();
  numeric::array::set_module_and_type("numpy", "ndarray");
  def("knn", knn);
}
