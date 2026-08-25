#include <cstring>
#include <iostream>
#include "tensorcore/tensorcore.h"

#define TC_STR2(x) #x
#define TC_STR(x) TC_STR2(x)

int main() {
    const char* expected =
        "tensorcore " TC_STR(TENSORCORE_VERSION_MAJOR)
        "." TC_STR(TENSORCORE_VERSION_MINOR)
        "." TC_STR(TENSORCORE_VERSION_PATCH);
    if (std::strcmp(tc_version(), expected) != 0) {
        std::cerr << "unexpected version: " << tc_version() << "\n";
        return 1;
    }
    if (tc_dtype_size(TC_DTYPE_F16) != 2 ||
        std::strcmp(tc_backend_name(TC_BACKEND_NONE), "none") != 0) {
        std::cerr << "public C++ helper check failed\n";
        return 1;
    }
    std::cout << tc_version() << "\n";
    return 0;
}
