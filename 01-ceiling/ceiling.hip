#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.hpp"
#include "timing.hpp"
#include "device.hpp"

int main(int argc, char** argv) {
    print_device();
    return 0;
}