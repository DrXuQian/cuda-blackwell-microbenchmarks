#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void test_cluster_support() {
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    
    if (threadIdx.x == 0) {
        printf("Block %d: Thread block size = %d\n", blockIdx.x, block.size());
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Cooperative Launch Support: " << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
    std::cout << "Cooperative Multi-Device Launch: " << (prop.cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl;
    
    // Try basic kernel launch
    test_cluster_support<<<2, 32>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
