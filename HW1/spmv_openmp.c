/* Student stub: spmv_openmp.c
 * Keeps IO, timing, verification. Students implement CSR conversion and OpenMP SpMV.
 */
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double time_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* Minimal MTX reader (coordinate) */
int read_mtx(const char *path, int *M, int *N, int *nnz,
             int **rows, int **cols, double **vals) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[1024]; do { if (!fgets(line,sizeof(line),f)){fclose(f);return-1;} } while(line[0]=='%');
    int m,n,k; if (sscanf(line, "%d %d %d", &m,&n,&k)!=3) { fclose(f); return -1; }
    *M=m; *N=n; *nnz=k;
    *rows=malloc(sizeof(int)*k); *cols=malloc(sizeof(int)*k); *vals=malloc(sizeof(double)*k);
    for (int i=0;i<k;i++){int r,c; double v; if (fscanf(f, "%d %d %lf", &r,&c,&v)!=3){fclose(f);return-1;} (*rows)[i]=r-1; (*cols)[i]=c-1; (*vals)[i]=v; }
    fclose(f); return 0;
}

int read_vec(const char *path, int N, double **x) {
    /* Align vector x to cache line */
    if (posix_memalign((void**)x, 64, N * sizeof(double)) != 0) return -1;
    FILE *f = fopen(path, "r");
    if (!f) { for(int i=0;i<N;i++) (*x)[i]=1.0; return 0; }
    for (int i=0;i<N;i++) { if (fscanf(f, "%lf", &(*x)[i])!=1) (*x)[i]=1.0; }
    fclose(f); return 0;
}

int read_gold(const char *path, int M, double **ygold) { FILE *f=fopen(path,"r"); if(!f) return -1; *ygold=malloc(sizeof(double)*M); for(int i=0;i<M;i++) if(fscanf(f,"%lf",&(*ygold)[i])!=1) (*ygold)[i]=0.0; fclose(f); return 0; }

int verify(int M, double *y, double *ygold) { double tol=0.02; for(int i=0;i<M;i++) if (fabs(y[i]-ygold[i])>tol) return 0; return 1; }

void build_csr(int M, int N, int nnz, int *rows, int *cols, double *vals,
               int **row_ptr, int **col_idx, double **vals_csr) {
    (void)N;

    /* Align row_ptr to cache line (64 bytes) */
    if (posix_memalign((void**)row_ptr, 64, (M + 1) * sizeof(int)) != 0) { exit(1); }
    memset(*row_ptr, 0, (M + 1) * sizeof(int));
    
    /* Align column indices and values to cache line */
    if (posix_memalign((void**)col_idx, 64, nnz * sizeof(int)) != 0) { exit(1); }
    if (posix_memalign((void**)vals_csr, 64, nnz * sizeof(double)) != 0) { exit(1); }

    for (int k = 0; k < nnz; k++) {
        int r = rows[k];
        (*row_ptr)[r + 1]++;
    }

    for (int i = 0; i < M; i++) {
        (*row_ptr)[i + 1] += (*row_ptr)[i];
    }

    /* Working cursor per row for stable COO->CSR placement. */
    int *offset = (int*)calloc((size_t)M, sizeof(int));
    if (!offset) { exit(1); }

    for (int k = 0; k < nnz; k++) {
        int r = rows[k];
        int dest = (*row_ptr)[r] + offset[r]++;
        (*col_idx)[dest] = cols[k];
        (*vals_csr)[dest] = vals[k];
    }

    free(offset);
}

/* Ultimate Optimized SpMV: Quad Accumulators + SIMD Hints + Prefetch Tuning */
void spmv_csr_openmp(int M, int * __restrict__ row_ptr, int * __restrict__ col_idx, 
                     double * __restrict__ vals_csr, double * __restrict__ x, double * __restrict__ y) {
    const int pf_dist = 32;

    #pragma omp parallel proc_bind(spread)
    {
        #pragma omp for schedule(static, 64)
        for (int i = 0; i < M; i++) {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];
            const int len = row_end - row_start;

            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            int j = row_start;

        

            #pragma GCC ivdep
            for (; j <= row_end - 8; j += 8) {
                if (j + pf_dist < row_end) {
                    __builtin_prefetch(&col_idx[j + pf_dist], 0, 2);
                    __builtin_prefetch(&vals_csr[j + pf_dist], 0, 2);
                }

                sum0 += vals_csr[j]     * x[col_idx[j]];
                sum1 += vals_csr[j + 1] * x[col_idx[j + 1]];
                sum2 += vals_csr[j + 2] * x[col_idx[j + 2]];
                sum3 += vals_csr[j + 3] * x[col_idx[j + 3]];
                sum0 += vals_csr[j + 4] * x[col_idx[j + 4]];
                sum1 += vals_csr[j + 5] * x[col_idx[j + 5]];
                sum2 += vals_csr[j + 6] * x[col_idx[j + 6]];
                sum3 += vals_csr[j + 7] * x[col_idx[j + 7]];
            }

            double final_sum = (sum0 + sum1) + (sum2 + sum3);
            for (; j < row_end; j++) {
                final_sum += vals_csr[j] * x[col_idx[j]];
            }
            y[i] = final_sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,"Usage: %s matrix.mtx [vector.txt]\n", argv[0]); return 1; }
    const char *mtx = argv[1]; const char *vec = (argc>2?argv[2]:NULL);
    int M,N,nnz; int *rows=NULL,*cols=NULL; double *vals=NULL;
    if (read_mtx(mtx,&M,&N,&nnz,&rows,&cols,&vals)!=0) { fprintf(stderr,"Failed to read mtx\n"); return 1; }
    
    double *x = NULL;
    if (read_vec(vec, N, &x) != 0) { fprintf(stderr,"Failed to read vector\n"); return 1; }

    /* Declare CSR pointers and call build_csr */
    int *row_ptr = NULL, *col_idx = NULL;
    double *vals_csr = NULL;
    build_csr(M, N, nnz, rows, cols, vals, &row_ptr, &col_idx, &vals_csr);

    double *y = NULL;
    if (posix_memalign((void**)&y, 64, M * sizeof(double)) != 0) { exit(1); }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) y[i] = 0.0;

    double t0 = time_ms();
    spmv_csr_openmp(M, row_ptr, col_idx, vals_csr, x, y);
    double t1 = time_ms();
    fprintf(stderr,"spmv_openmp_time_ms=%.3f\n", t1-t0);

    char goldpath[1024]; snprintf(goldpath,sizeof(goldpath),"%s.gold", mtx);
    double *ygold=NULL; if (read_gold(goldpath,M,&ygold)==0) {
        if (verify(M,y,ygold)) fprintf(stderr,"OK\n"); else fprintf(stderr,"WRONG\n"); free(ygold);
    } else fprintf(stderr,"No gold found (%s) — skipping verify\n", goldpath);

    free(rows); free(cols); free(vals); free(row_ptr); free(col_idx); free(vals_csr); free(x); free(y);
    return 0;
}