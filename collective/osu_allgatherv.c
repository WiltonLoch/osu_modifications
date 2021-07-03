#define BENCHMARK "OSU MPI%s Allgatherv Latency Test"
/*
 * Copyright (C) 2002-2021 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_mpi.h>
#include <size_distributions.h>

int main(int argc, char *argv[])
{
    int i, numprocs, rank, size, disp, rank_size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0, *iteration_means = NULL;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, std_dev = 0.0;
    char *sendbuf, *recvbuf;
    int *rdispls=NULL, *recvcounts=NULL;
    int po_ret;
    size_t bufsize;
    options.bench = COLLECTIVE;
    options.subtype = LAT;

    extern int (*distribution_functions[])(int, int, int);
    extern int (*distribution_total_blocks[])(int, int);

    set_header(HEADER);
    set_benchmark_name("osu_allgather");
    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));

    switch (po_ret) {
        case PO_BAD_USAGE:
            print_bad_usage_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    iteration_means = (double *) malloc(options.iterations * sizeof(double));
    if(iteration_means == NULL){
        fprintf(stderr, "Could Not Allocate Memory For Iteration Meands [randk %d]\n", rank);
        exit(EXIT_FAILURE);
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }


    if ((options.max_message_size * numprocs) > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit / numprocs;
    }

    if (allocate_memory_coll((void**)&recvcounts, numprocs*sizeof(int), NONE)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    if (allocate_memory_coll((void**)&rdispls, numprocs*sizeof(int), NONE)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }

    rank_size = distribution_functions[options.size_distribution](numprocs, options.max_message_size, rank);

    if (allocate_memory_coll((void**)&sendbuf, rank_size, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    /* set_buffer(sendbuf, options.accel, 1, options.max_message_size); */
    set_buffer(sendbuf, options.accel, 1, rank_size);

    rank_size = distribution_total_blocks[options.size_distribution](numprocs, options.max_message_size);
    /* if(rank == 0) fprintf(stdout, "rank_size 1: %d\n", rank_size); */

    /* bufsize = options.max_message_size * numprocs; */
    if (allocate_memory_coll((void**)&recvbuf, rank_size,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }
    /* set_buffer(recvbuf, options.accel, 0, bufsize); */
    /* if(rank == 1) fprintf(stdout, "rank_size 2: %d\n", rank_size); */
    set_buffer(recvbuf, options.accel, 0, rank_size);


    print_preamble(rank);

    for(size=options.min_message_size; size <= options.max_message_size; size *= 2) {
        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        disp = 0;
        for ( i = 0; i < numprocs; i++) {
            rank_size = distribution_functions[options.size_distribution](numprocs, size, i);
            recvcounts[i] = rank_size;
            rdispls[i] = disp;
            /* if(rank == 0) printf("rank %d: %d = %d\n", i, disp, rank_size); */
            disp += rank_size;
        }
        rank_size = distribution_functions[options.size_distribution](numprocs, size, rank);

        /* if(rank == 1) fprintf(stdout, "rank_size 3: %d\n", rank_size); */


        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        timer=0.0;
        for(i=0; i < options.iterations + options.skip ; i++) {

            t_start = MPI_Wtime();

            MPI_CHECK(MPI_Allgatherv(sendbuf, rank_size, MPI_CHAR, recvbuf, recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD));

            t_stop = MPI_Wtime();

            if(i >= options.skip) {

                /* block for printing each iterations mean execution time over all processes */
                latency = t_stop - t_start;
                MPI_CHECK(MPI_Reduce(&latency, &iteration_means[i - options.skip], 1, MPI_DOUBLE, MPI_SUM, 0,
                    MPI_COMM_WORLD));
                if(rank == 0)
                     iteration_means[i - options.skip] = (double)(iteration_means[i - options.skip] * 1e6) / numprocs;

                timer+= t_stop-t_start;
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        latency = (double)(timer * 1e6) / options.iterations;

        MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_time = avg_time/numprocs;

        print_stats(rank, size, avg_time, min_time, max_time);

        if(rank == 0){
            for (int i = 0; i < options.iterations; ++i) {
                std_dev += (iteration_means[i] - avg_time)*(iteration_means[i] - avg_time);
                std_dev /= (double) options.iterations;
                std_dev = sqrt(std_dev);
            }
            fprintf(stdout, "%-*s", 10, "Std dev:");
            fprintf(stdout, "%*.*lf\n", FIELD_WIDTH, FLOAT_PRECISION, std_dev);
        }


        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    free_buffer(rdispls, NONE);
    free_buffer(recvcounts, NONE);
    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}
/* vi: set sw=4 sts=4 tw=80: */
