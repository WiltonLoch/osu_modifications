#pragma once

enum size_distributions{
    REGULAR,
    BROADCAST,
    SPIKE,
    HALF_FULL,
    LINEARLY_DECREASING,
    GEOMETRIC_CURVE,
    TOTAL_FUNCTIONS // maintains the count of functions, always keep last
};

int regular(int p, int default_size, int rank);
int broadcast(int p, int default_size, int rank);
int spike(int p, int default_size, int rank);
int half_full(int p, int default_size, int rank);
int linearly_decreasing(int p, int default_size, int rank);
int geometric_curve(int p, int default_size, int rank);

int regular_total_blocks(int p, int default_size);
int broadcast_total_blocks(int p, int default_size);
int spike_total_blocks(int p, int default_size);
int half_full_total_blocks(int p, int default_size);
int linearly_decreasing_total_blocks(int p, int default_size);
int geometric_curve_total_blocks(int p, int default_size);
