// Test cluster distributed shared memory on RTX 5070 (Blackwell)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Simple kernel to test cluster.map_shared_rank
__global__ void test_cluster_basic(int* output) {
    extern __shared__ int smem[];
    auto cluster = cg::this_cluster();
    
    int tid = threadIdx.x;
    int block_rank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    
    // Each block writes its rank to shared memory
    if (tid == 0) {
        smem[0] = block_rank;
        printf("Block %d: Writing %d to smem[0]\n", block_rank, block_rank);
    }
    __syncthreads();
    
    // Try cluster sync
    printf("Block %d, Thread %d: Before cluster.sync()\n", block_rank, tid);
    cluster.sync();
    printf("Block %d, Thread %d: After cluster.sync()\n", block_rank, tid);
    
    // Try to read from other blocks' shared memory
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < cluster_size; i++) {
            int* remote_smem = (int*)cluster.map_shared_rank(smem, i);
            int val = remote_smem[0];
            sum += val;
            printf("Block %d: Read %d from block %d\n", block_rank, val, i);
        }
        output[block_rank] = sum;
    }
}

// Test without cluster features (baseline)
__global__ void test_no_cluster(int* output) {
    extern __shared__ int smem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Each block writes its ID to shared memory
    if (tid == 0) {
        smem[0] = bid;
        printf("Block %d: Writing %d to smem[0] (no cluster)\n", bid, bid);
    }
    __syncthreads();
    
    // Write to output
    if (tid == 0) {
        output[bid] = smem[0];
    }
}

// Test with manual barrier instead of cluster.sync
__global__ void test_cluster_barrier(int* output) {
    extern __shared__ int smem[];
    auto cluster = cg::this_cluster();
    
    int tid = threadIdx.x;
    int block_rank = cluster.block_rank();
    
    // Each block writes its rank to shared memory
    if (tid == 0) {
        smem[0] = block_rank * 100;  // Make values distinct
    }
    __syncthreads();
    
    // Use barrier instead of cluster.sync
    auto block = cg::this_thread_block();
    block.sync();
    
    // Try to read from own shared memory first
    if (tid == 0) {
        int local_val = smem[0];
        printf("Block %d: Local smem[0] = %d\n", block_rank, local_val);
        output[block_rank] = local_val;
    }
}

int main() {
    printf("=== Testing Cluster Distributed Shared Memory on RTX 5070 ===\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    
    // Test parameters
    const int cluster_size = 4;  // Start with small cluster
    const int threads_per_block = 32;
    const int shared_mem_size = sizeof(int) * 10;  // Small shared memory
    
    int* d_output;
    cudaMalloc(&d_output, cluster_size * sizeof(int));
    
    printf("\n--- Test 1: Baseline (no cluster) ---\n");
    test_no_cluster<<<cluster_size, threads_per_block, shared_mem_size>>>(d_output);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error in baseline test: %s\n", cudaGetErrorString(err));
    } else {
        printf("Baseline test completed successfully\n");
    }
    
    printf("\n--- Test 2: Cluster with barrier (no cluster.sync) ---\n");
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = cluster_size;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = shared_mem_size;
        
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaLaunchKernelEx(&config, test_cluster_barrier, d_output);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error in cluster barrier test: %s\n", cudaGetErrorString(err));
        } else {
            printf("Cluster barrier test completed successfully\n");
        }
    }
    
    printf("\n--- Test 3: Full cluster test (with cluster.sync and map_shared_rank) ---\n");
    printf("WARNING: This may hang based on previous tests\n");
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = cluster_size;
        config.blockDim = threads_per_block;
        config.dynamicSmemBytes = shared_mem_size;
        
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.numAttrs = 1;
        config.attrs = attribute;
        
        // Set timeout for safety
        printf("Launching kernel (5 second timeout)...\n");
        cudaLaunchKernelEx(&config, test_cluster_basic, d_output);
        
        // Use cudaEventQuery to check if kernel is still running
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event);
        
        // Wait up to 5 seconds
        for (int i = 0; i < 50; i++) {
            cudaError_t query_result = cudaEventQuery(event);
            if (query_result == cudaSuccess) {
                printf("Kernel completed successfully!\n");
                break;
            } else if (i == 49) {
                printf("Kernel appears to be hanging (5 seconds elapsed)\n");
                printf("This confirms the cluster.sync/map_shared_rank issue\n");
                // Note: We can't actually kill the kernel, it will keep running
            }
            usleep(100000);  // Sleep 100ms
        }
        
        cudaEventDestroy(event);
    }
    
    // Copy results back (if kernel completed)
    int h_output[cluster_size];
    cudaMemcpy(h_output, d_output, cluster_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nOutput values:\n");
    for (int i = 0; i < cluster_size; i++) {
        printf("Block %d: %d\n", i, h_output[i]);
    }
    
    cudaFree(d_output);
    
    printf("\n=== Alternative Solution: Use L2 Cache or Global Memory ===\n");
    printf("Since distributed shared memory appears problematic on this GPU,\n");
    printf("consider these alternatives:\n");
    printf("1. Use L2 cache resident data (cudaAccessPolicyWindow)\n");
    printf("2. Use global memory with coalesced access patterns\n");
    printf("3. Use texture memory for read-only data\n");
    printf("4. Split work differently to fit in single block's shared memory\n");
    
    return 0;
}