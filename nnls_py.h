#ifndef NNLS_PY_H
#define NNLS_PY_H

typedef double dtype;

dtype _2d_access_1d(dtype* a, size_t row, size_t col) {
    return a[col + sizeof(dtype) * row];
}
dtype* _2d_assign_1d(dtype* a, size_t row, size_t col, dtype v) {
    a[col + sizeof(dtype) * row] = (dtype)v;
    return a;
}

typedef struct {
    dtype* data;
    size_t rows, cols;
} dtype_matrix;
dtype_matrix dtm_init(dtype_matrix* m, size_t rows, size_t cols) {
    m->rows = rows;
    m->cols = cols;
    m->data = (dtype*)malloc(rows * cols * sizeof(dtype));
    return *m;
}
void dtm_free(dtype_matrix* m) {
    free(m->data);
}

size_t dtm_rows(dtype_matrix m) {
    return m.rows;
}
size_t dtm_cols(dtype_matrix m) {
    return m.cols;
}

dtype dtm_at(dtype_matrix m, size_t row, size_t col) {
    return _2d_access_1d(m.data, row, col);
}
dtype* dtm_col_at(dtype_matrix m, size_t col) {
    static dtype* col_val = new dtype[m.rows];
    for (int i = 0; i < m.rows; i++) {
        col_val[i] = dtm_at(m, i, col);
    }
    return col_val;
}
dtype* dtm_row_at(dtype_matrix m, size_t row) {
    static dtype* row_val = new dtype[m.rows];
    for (int i = 0; i < m.rows; i++) {
        row_val[i] = dtm_at(m, row, i);
    }
    return row_val;
}
dtype* dtm_to(dtype_matrix* m, size_t row, size_t col, dtype val) {
    return _2d_assign_1d(m->data, row, col, val);
}
dtype* dtm_row_to(dtype_matrix *m, size_t row, dtype* val) {
    for (int j = 0; j < m->cols; j++) {
        dtm_to(m, row, j, val[j]);
    }
    return m->data;
}
dtype* dtm_col_to(dtype_matrix *m, size_t col, dtype* val) {
    for (int i = 0; i < m->rows; i++) {
        dtm_to(m, i, col, val[i]);
    }
    return m->data;
}



int nnls_c(double* a, const int* mda, const int* m, const int* n, double* b,
    double* x, double* rnorm, double* w, double* zz, int* index,
    int* mode);


#endif