#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Per-record: float32 x, y, radius (12 bytes) + uint8 r, g, b (3 bytes) = 15 bytes
#define RECSZ 15

/*
 * Render all circles into a local row strip [row_start, row_end).
 * Circles are processed back-to-front (file order), so later ones overwrite earlier ones.
 * Strip buffer is row-major, rows indexed as (y - row_start).
 */
static void render_strip(unsigned char *strip, int W, int row_start, int row_end,
                         const unsigned char *records, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        const unsigned char *ptr = records + i * RECSZ;
        float cx, cy, radius;
        memcpy(&cx,     ptr,     4);
        memcpy(&cy,     ptr + 4, 4);
        memcpy(&radius, ptr + 8, 4);
        unsigned char cr = ptr[12], cg = ptr[13], cb = ptr[14];

        // Quick reject: circle bounding box doesn't touch this row band
        int ymin = (int)floorf(cy - radius);
        int ymax = (int)floorf(cy + radius);
        if (ymax < row_start || ymin >= row_end) continue;

        // Clamp to this rank's row band
        if (ymin < row_start) ymin = row_start;
        if (ymax >= row_end)  ymax = row_end - 1;

        int xmin = (int)floorf(cx - radius);
        int xmax = (int)floorf(cx + radius);
        if (xmin < 0) xmin = 0;
        if (xmax >= W) xmax = W - 1;

        float r2 = radius * radius;

        for (int y = ymin; y <= ymax; y++) {
            float dy  = (y + 0.5f) - cy;
            float dy2 = dy * dy;
            if (dy2 > r2) continue;

            size_t row_off = (size_t)(y - row_start) * W;
            for (int x = xmin; x <= xmax; x++) {
                float dx = (x + 0.5f) - cx;
                if (dx * dx + dy2 <= r2) {
                    size_t idx = (row_off + x) * 3;
                    strip[idx + 0] = cr;
                    strip[idx + 1] = cg;
                    strip[idx + 2] = cb;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input.bin> <output.png>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    uint64_t count = 0;
    int W = 0, H = 0;
    unsigned char *all_records = NULL;
    double t0 = 0.0;

    // Rank 0 reads the file
    if (rank == 0) {
        t0 = MPI_Wtime();
        FILE *f = fopen(argv[1], "rb");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }

        char magic[4];
        uint32_t version;
        float bbox[6];
        if (fread(magic, 1, 4, f) != 4)     { MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&version, 4, 1, f) != 1)  { MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&count, 8, 1, f) != 1)    { MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(bbox, 4, 6, f) != 6)      { MPI_Abort(MPI_COMM_WORLD, 1); }

        W = (int)roundf(bbox[3] - bbox[0]);
        H = (int)roundf(bbox[4] - bbox[1]);
        if (W <= 0) W = 640;
        if (H <= 0) H = 480;

        size_t totsz = (size_t)count * RECSZ;
        all_records = malloc(totsz);
        if (!all_records) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(all_records, 1, totsz, f) != totsz) { MPI_Abort(MPI_COMM_WORLD, 1); }
        fclose(f);
    }

    // Broadcast image dimensions and circle count
    MPI_Bcast(&W,     1, MPI_INT,      0, MPI_COMM_WORLD);
    MPI_Bcast(&H,     1, MPI_INT,      0, MPI_COMM_WORLD);
    MPI_Bcast(&count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Broadcast all circle data to every rank
    if (rank != 0) {
        all_records = malloc((size_t)count * RECSZ);
        if (!all_records) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    // Bcast in 1 GB chunks to avoid int overflow for very large datasets
    size_t total_bytes = (size_t)count * RECSZ;
    for (size_t off = 0; off < total_bytes; ) {
        size_t remain = total_bytes - off;
        int chunk = (remain > (size_t)1073741824) ? 1073741824 : (int)remain;
        MPI_Bcast(all_records + off, chunk, MPI_BYTE, 0, MPI_COMM_WORLD);
        off += chunk;
    }

    // Assign a contiguous row band to each rank
    int base_rows = H / nprocs;
    int remainder = H % nprocs;
    int my_start  = rank * base_rows + (rank < remainder ? rank : remainder);
    int my_end    = my_start + base_rows + (rank < remainder ? 1 : 0);
    int my_rows   = my_end - my_start;

    // Render all circles into local strip (back-to-front order preserved)
    unsigned char *strip = calloc((size_t)my_rows * W * 3, 1);
    if (!strip) { perror("calloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    render_strip(strip, W, my_start, my_end, all_records, count);
    free(all_records);

    // Gather strips at rank 0
    int *recvcounts = NULL, *displs = NULL;
    unsigned char *full_img = NULL;
    if (rank == 0) {
        recvcounts = malloc(nprocs * sizeof(int));
        displs     = malloc(nprocs * sizeof(int));
        int off = 0;
        for (int i = 0; i < nprocs; i++) {
            int r_rows    = base_rows + (i < remainder ? 1 : 0);
            recvcounts[i] = r_rows * W * 3;
            displs[i]     = off;
            off          += recvcounts[i];
        }
        full_img = malloc((size_t)H * W * 3);
        if (!full_img) { perror("malloc full_img"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Gatherv(strip,    my_rows * W * 3, MPI_BYTE,
                full_img, recvcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
    free(strip);

    // Rank 0 writes PNG and reports time
    if (rank == 0) {
        stbi_write_png(argv[2], W, H, 3, full_img, W * 3);
        fprintf(stderr, "rank0: Total time: %.6f s\n", MPI_Wtime() - t0);
        free(full_img);
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
