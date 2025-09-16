// Test cluster with larger shared memory like deformable attention
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Test with parameters similar to deformable attention
__global__ void test_large_cluster(__half* output) {
    extern __shared__ __half smem[];
    auto cluster = cg::this_cluster();
    
    int tid = threadIdx.x;
    int block_rank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    
    // Simulate deformable attention memory layout
    // Each block stores 2 channels * 20522 spatial = 41044 halfs = 82KB
    const int spatial_size = 20522;
    const int channels_per_block = 2;
    const int elements_per_block = spatial_size * channels_per_block;
    
    // Initialize shared memory with pattern
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        smem[i] = __float2half(block_rank + (float)i / elements_per_block);
    }
    __syncthreads();
    
    // Debug print before sync
    if (tid == 0) {
        printf("Block %d: Initialized %d elements, first value = %f\n", 
               block_rank, elements_per_block, __half2float(smem[0]));
    }
    
    // Try cluster sync
    cluster.sync();
    
    if (tid == 0) {
        printf("Block %d: After cluster.sync()\n", block_rank);
    }
    
    // Try to read from other blocks - similar to deformable attention pattern
    if (tid < 16) {  // Only first 16 threads, like processing 16 channels
        __half sum = __float2half(0.0f);
        
        // Read from all blocks in cluster
        for (int b = 0; b < cluster_size; b++) {
            __half* remote_smem = (__half*)cluster.map_shared_rank(smem, b);
            
            // Read a few values (simulating bilinear interpolation access)
            int idx = tid * 100;  // Spread out accesses
            if (idx < elements_per_block) {
                __half val = remote_smem[idx];
                sum = __hadd(sum, val);
                
                if (tid == 0) {
                    printf("Block %d: Read %f from block %d at idx %d\n", 
                           block_rank, __half2float(val), b, idx);
                }
            }
        }
        
        // Write result
        output[block_rank * 16 + tid] = sum;
    }
}

// Test with exact deformable attention launch config
__global__ void test_deform_config(__half* data_value, __half* output) {
    extern __shared__ __half smem[];
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    const int tid = threadIdx.x;
    const int cluster_size = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int channels_per_block = 2;
    
    // Calculate batch and output channel like deformable attention
    const int bi = blockIdx.x / (2 * cluster_size);
    const int oc = (blockIdx.x % (2 * cluster_size)) / cluster_size;
    
    if (tid == 0 && blockIdx.x < 16) {
        printf("Block %d: bi=%d, oc=%d, block_rank=%d\n", blockIdx.x, bi, oc, block_rank);
    }
    
    // Load data cooperatively
    const int spatial_size = 20522;
    const int channels = 32;
    const int my_channel_offset = (block_rank * channels_per_block) + (oc * 16);
    
    const __half* src = data_value + 
        (bi * spatial_size * channels) + my_channel_offset;
    
    // Cooperative copy
    int total_elements = spatial_size * channels_per_block;
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int s = idx / channels_per_block;
        int c = idx % channels_per_block;
        if (s < spatial_size && c < channels_per_block) {
            smem[s * channels_per_block + c] = src[s * channels + c];
        }
    }
    __syncthreads();
    
    // Try cluster sync
    if (tid == 0 && blockIdx.x == 0) {
        printf("Before cluster.sync()\n");
    }
    cluster.sync();
    if (tid == 0 && blockIdx.x == 0) {
        printf("After cluster.sync() - SUCCESS!\n");
    }
    
    // Simple test: read from other blocks
    if (tid == 0) {
        __half sum = __float2half(0.0f);
        for (int b = 0; b < cluster_size; b++) {
            __half* remote_smem = (__half*)cluster.map_shared_rank(smem, b);
            sum = __hadd(sum, remote_smem[0]);
        }
        output[blockIdx.x] = sum;
        printf("Block %d: Sum = %f\n", blockIdx.x, __half2float(sum));
    }
}

int main() {
    printf("=== Testing Large Cluster Configuration (like deformable attention) ===\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // Test 1: Large shared memory with cluster size 8
    printf("\n--- Test 1: Cluster size 8 with 82KB shared memory ---\n");
    {
        const int cluster_size = 8;
        const int threads_per_block = 256;
        const int spatial_size = 20522;
        const int channels_per_block = 2;
        const size_t shared_mem_size = spatial_size * channels_per_block * sizeof(__half);
        
        printf("Shared memory per block: %zu KB\n", shared_mem_size / 1024);
        
        __half* d_output;
        cudaMalloc(&d_output, cluster_size * 16 * sizeof(__half));
        
        cudaLaunchConfig_t config = {0};
        config.gridDim = cluster_size;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = shared_mem_size;
        
        // Set max dynamic shared memory
        cudaFuncSetAttribute(test_large_cluster,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            shared_mem_size);
        
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaLaunchKernelEx(&config, test_large_cluster, d_output);
        cudaError_t err = cudaDeviceSynchronize();
        
        if (err != cudaSuccess) {
            printf("ERROR: %s\n", cudaGetErrorString(err));
        } else {
            printf("SUCCESS: Large cluster test completed!\n");
        }
        
        cudaFree(d_output);
    }
    
    // Test 2: Exact deformable attention configuration
    printf("\n--- Test 2: Exact deformable attention config (48 batches, 768 blocks) ---\n");
    {
        const int batch_size = 48;
        const int cluster_size = 8;
        const int grid_size = batch_size * cluster_size * 2;  // 768 blocks
        const int threads_per_block = 512;
        const int spatial_size = 20522;
        const int channels = 32;
        const size_t shared_mem_size = spatial_size * 2 * sizeof(__half);
        
        printf("Grid: %d blocks, Cluster size: %d, Shared mem: %zu KB\n", 
               grid_size, cluster_size, shared_mem_size / 1024);
        
        // Allocate data
        size_t value_size = batch_size * spatial_size * channels * sizeof(__half);
        __half *d_value, *d_output;
        cudaMalloc(&d_value, value_size);
        cudaMalloc(&d_output, grid_size * sizeof(__half));
        
        // Initialize with pattern
        __half* h_value = new __half[batch_size * spatial_size * channels];
        for (int i = 0; i < batch_size * spatial_size * channels; i++) {
            h_value[i] = __float2half((i % 100) * 0.01f);
        }
        cudaMemcpy(d_value, h_value, value_size, cudaMemcpyHostToDevice);
        
        cudaLaunchConfig_t config = {0};
        config.gridDim = grid_size;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = shared_mem_size;
        
        cudaFuncSetAttribute(test_deform_config,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            shared_mem_size);
        
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.numAttrs = 1;
        config.attrs = attribute;
        
        printf("Launching kernel...\n");
        cudaLaunchKernelEx(&config, test_deform_config, d_value, d_output);
        
        // Check with timeout
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event);
        
        bool completed = false;
        for (int i = 0; i < 50; i++) {  // 5 second timeout
            cudaError_t query = cudaEventQuery(event);
            if (query == cudaSuccess) {
                printf("SUCCESS: Kernel completed!\n");
                completed = true;
                break;
            }
            usleep(100000);  // 100ms
        }
        
        if (!completed) {
            printf("TIMEOUT: Kernel appears to be hanging\n");
        }
        
        cudaEventDestroy(event);
        cudaFree(d_value);
        cudaFree(d_output);
        delete[] h_value;
    }
    
    return 0;
}