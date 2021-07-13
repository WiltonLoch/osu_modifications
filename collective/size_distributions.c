#include "size_distributions.h"
#include <math.h>
#include <stdio.h>

int regular(int p, int default_size, int rank){
    return default_size;
}
int regular_total_blocks(int p, int default_size){
    return p * default_size;
}

int broadcast(int p, int default_size, int rank){
    return rank == 0 ? default_size : 0;
}
int broadcast_total_blocks(int p, int default_size){
    return default_size;
}

int spike(int p, int default_size, int rank){
    return rank == 0 ? (p * default_size) : (p * default_size) / (p - 1);
}
int spike_total_blocks(int p, int default_size){
    return 2 * p * default_size;
}

int half_full(int p, int default_size, int rank){
    return ((rank % 2) == 0) ? 2 * default_size : 0;
}
int half_full_total_blocks(int p, int default_size){
    return (p + (p % 2)) * default_size;
}

int linearly_decreasing(int p, int default_size, int rank){
    return 2.0f * (float) default_size * ((float) (p - 1 - rank) / (p - 1));
}
int linearly_decreasing_total_blocks(int p, int default_size){
    return p * default_size;
}

int geometric_curve(int p, int default_size, int rank){
    return (p * default_size) / ((rank + 1.5) * log(p + 1));
}
int geometric_curve_total_blocks(int p, int default_size){
    return p * default_size;
}

int (*distribution_functions[])(int, int, int) = {
    regular,
    broadcast,
    spike,
    half_full,
    linearly_decreasing,
    geometric_curve
};

int (*distribution_total_blocks[])(int, int) = {
    regular_total_blocks,
    broadcast_total_blocks,
    spike_total_blocks,
    half_full_total_blocks,
    linearly_decreasing_total_blocks,
    geometric_curve_total_blocks
};
