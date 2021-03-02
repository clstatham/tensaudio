#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
//#include "debug.h"
//#include "entities.h"
//#include "nnls.h"
//#include "nmf.h"
#include <iostream>
extern "C" {
#include "nnls_py.h"
}

#define NNLS_MEM_SCALE 1
#define A_ROWS NNLS_MEM_SCALE * 128
#define A_COLS NNLS_MEM_SCALE * 32769
#define A_SIZE NNLS_MEM_SCALE * 32769 * 128
#define B_ROWS NNLS_MEM_SCALE * 31
#define B_COLS NNLS_MEM_SCALE * 32769
#define B_SIZE NNLS_MEM_SCALE * 32769 * 31

/*
typedef std::vector<dtype> _1d_vector;
typedef std::vector<_1d_vector> _2d_vector;
PYBIND11_MAKE_OPAQUE(std::vector<dtype>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<dtype>>);
*/
namespace py = pybind11;

dtype* _2d_ndarray_to_pointer(py::array_t<dtype> pyarr, size_t rows, size_t cols, bool rotate=true)
{
    py::buffer_info b = pyarr.request();
    dtype* p = (dtype*)b.ptr;
    //std::cout << "_2d_ndarray_to_pointer:: Rows: " << rows << " Cols: " << cols << " | Total size: " << rows * cols * sizeof(dtype) << " bytes." << std::endl;
    dtype* out = new dtype[rows*cols];

    //out = colmaj_to_rowmaj_and_vice_versa_1d(p, rows, cols);
    
    for (int col = 0; col < cols; col++) {
        dtype* colp = new dtype[rows];
        for (int row = 0; row < rows; row++) {
            if (rotate) {
                _2d_assign_1d(out, row, col, _2d_access_1d(p, row, col));
            }
            else {
                _2d_assign_1d(out, col, row, _2d_access_1d(p, row, col));
            }
            
        }
    }
    
    return out;
}

dtype_matrix _2d_ndarray_to_matrix(py::array_t<dtype> pyarr, size_t rows, size_t cols, bool rotate = true)
{
    //py::buffer_info b = pyarr.request();
    //dtype* p = (dtype*)b.ptr;
    auto r = pyarr.unchecked<2>();
    //std::cout << "_2d_ndarray_to_pointer:: Rows: " << rows << " Cols: " << cols << " | Total size: " << rows * cols * sizeof(dtype) << " bytes." << std::endl;
    dtype_matrix out;
    if (rotate) {
        dtm_init(&out, cols, rows);
    }
    else {
        dtm_init(&out, rows, cols);
    }
    

    //out = colmaj_to_rowmaj_and_vice_versa_1d(p, rows, cols);

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            if (rotate) {
                //_2d_assign_access_1d_conversion(out, row, col, p);
                dtm_to(&out, col, row, r(row, col));
            }
            else {
                //_2d_assign_access_1d_noconversion(out, row, col, p);
                dtm_to(&out, row, col, r(row, col));
            }
        }
    }

    return out;
}

py::array_t<dtype*> _2d_vector_to_ndarray(std::vector<std::vector<dtype>> A)
{
    size_t rows = A.size();
    size_t cols = A[0].size();
    static dtype** ptr = new dtype * [rows];
    py::capsule free_when_done(ptr, [](void* f) {
        dtype** foo = reinterpret_cast<dtype**>(f);
        std::cerr << "Element [0] = " << foo[0] << "\n";
        std::cerr << "freeing memory @ " << f << "\n";
        delete[] foo;
    });
    
    //static auto out_r = out.mutable_unchecked<2>();
    //std::cout << "Creating Array. rows=" << rows << " cols=" << cols << std::endl;
    int i = 0;
    for (std::vector<dtype> row : A) {
        int j = 0;
        ptr[i] = new dtype[cols];
        for (dtype col : row) {
            //std::cout << "i=" << i << " j=" << j << std::endl;
            ptr[i][j] = col;
            j++;
        }
        i++;
    }
    //std::cout << "Returning Array." << std::endl;
    py::array_t<dtype*, py::array::c_style | py::array::forcecast> result(
        { rows, cols },
        { sizeof(dtype) * cols, sizeof(dtype) },
        ptr,
        free_when_done
        );
    return result;
}

py::array_t<dtype> _pointer_to_2d_ndarray(dtype* A, size_t rows, size_t cols)
{
    static py::array_t<dtype, py::array::c_style | py::array::forcecast> out({ rows, cols });
    //auto r = out.mutable_unchecked();
    //std::cout << "_pointer_to_2d_ndarray:: Rows: " << rows << " Cols: " << cols << " | Total size: " << rows * cols * sizeof(dtype) << std::endl;

    py::buffer_info b = out.request();
    static dtype* p = (dtype*)b.ptr;
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            //std::cout << col << ", " << row << ": " << (dtype)(A[col][row]) << std::endl;
            _2d_assign_1d(p, row, col, _2d_access_1d(A, row, col));
        }
    }
    return out;
}

double* ta_nnls(dtype_matrix a, double* b)
{
    int m = a.rows; // A will be rotated from the input
    int n = a.cols;
    static double* A = new double[m * n];
    static double* B = new double[m];
    static double* X = new double[n];

    //std::cout << "Memory allocated, copying." << std::endl;

    for (int col = 0; col < a.cols; col++) {
        for (int row = 0; row < a.rows; row++) {
            _2d_assign_1d(A, col, row, dtm_at(a,row,col));
            //A[col * arows + col] = (double)(a[col][row]);
        }
    }
    //std::cout << "A copied." << std::endl;

    memcpy(B, b, m);
    
    //B = b.data;
    //std::cout << "B Copied. Going to the subroutine." << std::endl;
    

    static double Rnorm = 0;
    static double* W = new double[n];
    static double* ZZ = new double[m];
    static int* index = new int[n];
    static int mode = 0;
    static int mda = m;

    

    int retval = nnls_c(A, &mda, &m, &n, B, X, &Rnorm, W, ZZ, index, &mode);
    //std::cout << "Subroutine done. Freeing A and B copies." << std::endl;
    //free(A);
    //free(B);
    //std::cout << "A & B freed. Returning." << std::endl;
    return X;
}

class NNLSSolver {
public:
    NNLSSolver() {
    }
    ~NNLSSolver() {}
    py::array_t<dtype> get_result() {
        return py::cast<py::array_t<dtype>>(_pointer_to_2d_ndarray(this->result.data, this->result.rows, this->result.cols));
    }
    py::ssize_t result_rows() {
        return this->result.rows;
    }
    py::ssize_t result_cols() {
        return this->result.cols;
    }
    py::array_t<dtype> solve(py::buffer a, py::buffer b, size_t desired_iters) {
        //std::cout << "================================================================" << std::endl;
        //static py::array_t<dtype> out;
        py::buffer_info ab = a.request();
        if (ab.ndim != 2) {
            std::cerr << "Error, wrong input format (a.ndim=" << ab.ndim << ")" << std::endl;
        }
        py::buffer_info bb = b.request();
        if (bb.ndim > 2) {
            std::cerr << "Error, wrong input format (b.ndim=" << bb.ndim << ")" << std::endl;
        }

        
        
        
        //std::cout << "Got inputs. Converting." << std::endl;
        static dtype_matrix A = _2d_ndarray_to_matrix(a, ab.shape[0], ab.shape[1], false);
        //std::cout << "A converted." << std::endl;
        static dtype_matrix B = _2d_ndarray_to_matrix(b, bb.shape[0], bb.shape[1], true);
        //std::cout << "B converted." << std::endl;

        

        //std::cout << "A.rows, cols: " << A.rows << " | " << A.cols << std::endl; // 16385 | 128
        //std::cout << "B.rows, cols: " << B.rows << " | " << B.cols << std::endl; // 63    | 128
        //std::cout << "X will have " << A.cols << " elements." << std::endl;
        static dtype** res = new dtype * [A.cols * desired_iters];
        //dtm_init(res, desired_iters, A.cols); // 63, 16385
        static double* input_b;
        //std::cout << "marco" << std::endl;
        //dtm_init(&input_b, A.cols, 1); // 16385, 1
        for (int i = 0; i < desired_iters; i++) { // 63
            
            //dtm_row_to(&input_b, 0, dtm_row_at(B, i)); // input_b[:][0] = B[:][i]
            input_b = dtm_row_at(B, i);
            
            static double* X = ta_nnls(A, input_b);
            res[i] = X;
            //std::cout << std::endl;
            //std::cout << "bar" << std::endl;
        }
        //std::cout << "polo" << std::endl;

        py::array_t<dtype> out({ (py::ssize_t)A.cols, (py::ssize_t)desired_iters });
        for (int row = 0; row < desired_iters; row++) {
            for (int col = 0; col < A.cols; col++) {
                if (res[row][col] != 0) {
                    //std::cout << "res[" << row << "][" << col << "]: " << res[row][col] << std::endl;
                }
                
                out.mutable_at(col, row) = res[row][col];
            }
        }

        /*
        py::capsule free_when_done(res, [](void* f) {
            dtype** foo = reinterpret_cast<dtype**>(f);
            std::cerr << "Element [0] = " << foo[0] << "\n";
            std::cerr << "freeing memory @ " << f << "\n";
            delete[] foo;
        });
        */

        //std::cout << "Done! out.rows=" << this->result.rows << "\tout.cols=" << this->result.cols << std::endl;
        //dtm_free(&A);
        //dtm_free(&B);
        //free(input_b);
        //dtm_free(&result);
        //std::cout << "Memory freed." << std::endl;
        return out;
    }
public:
    dtype_matrix result;
    static NNLSSolver& instance()
    {
        static NNLSSolver singleton;
        return singleton;
    }
};

PYBIND11_MODULE(nnls, m) {
    m.doc() = "nnls python binding";
    //m.def("nnls", &nnls, py::keep_alive<0, 2>(), py::keep_alive<0, 1>(), py::return_value_policy::copy);
    pybind11::class_<dtype_matrix>(m, "DTypeMatrix")
        .def(py::init<>())
        .def("create", &dtm_init)
        .def("rows", &dtm_rows)
        .def("cols", &dtm_cols)
        .def("row_at", &dtm_row_at)
        .def("col_at", &dtm_col_at)
        .def("at", &dtm_at);
    pybind11::class_<NNLSSolver>(m, "NNLSSolver", pybind11::buffer_protocol())
        //.def(py::init<>())
        .def("result_rows", &NNLSSolver::result_rows)
        .def("result_cols", &NNLSSolver::result_cols)
        .def("instance", &NNLSSolver::instance, py::return_value_policy::reference)
        .def("solve", &NNLSSolver::solve, py::keep_alive<0, 2>(), py::keep_alive<1, 0>(), py::return_value_policy::move)
        .def_buffer([](NNLSSolver& s) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            // Pointer to buffer
            s.result.data,
            // Size of one scalar
            sizeof(dtype),
            // Python struct-style format descriptor
            pybind11::format_descriptor<dtype>::format(),
            // Number of dimensions
            2,
            // Buffer dimensions
            { s.result_rows(), s.result_cols() },
            // Strides (in bytes) for each index
                {
                    sizeof(dtype) * s.result_cols(),
                    sizeof(dtype)
                }
                );
    });
}