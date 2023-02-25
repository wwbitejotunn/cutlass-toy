#include "cutlass_fpAintB/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "stdlib.h"
#include <chrono>
#include "iostream"
#include <string>
namespace fastertransformer{
template class CutlassFpAIntBGemmRunner<half, uint8_t>;
}  // namespace fastertransformer

int main(int argc, char *argv[]){
    // m n k
    // argv[1], argv[2], argv[3]
    int m = strtol(argv[1], nullptr, 0);
    int n = strtol(argv[2], nullptr, 0);
    int k = strtol(argv[3], nullptr, 0);

    const auto kWarmTime=3;
    const auto kTestTime=10;

    auto mixed_gemm_runner = fastertransformer::CutlassFpAIntBGemmRunner<half, uint8_t>();

    int mixgemm_max_size=std::max(m,k);
    int mixgemm_workspace_size_bytes=mixed_gemm_runner.getWorkspaceSize(m, mixgemm_max_size, mixgemm_max_size);
    char *mixgemm_workspace_data;
    cudaMalloc(&mixgemm_workspace_data, mixgemm_workspace_size_bytes);
    std::vector<half> a_half(m*k);
    for(auto & i:a_half){
        i=(float)rand()/RAND_MAX*20.0;
    }
    std::vector<int8_t> b_int(k*n);
    for(auto & i:b_int){
        i=rand()%256-127;
    }
    std::vector<half> b_scale_half(n);
    for(auto & i:b_scale_half){
        i=(float)rand()/RAND_MAX*0.05;
    }
    std::vector<half> c_half(m*n);
    void* d_a_half;
    void* d_b_int;
    void* d_b_scale;
    void* d_c_half;
    cudaMalloc(&d_a_half, m*k*sizeof(half));
    cudaMalloc(&d_b_int, k*n*sizeof(int8_t));
    cudaMalloc(&d_b_scale, n*sizeof(half));
    cudaMalloc(&d_c_half, m*n*sizeof(half));
    cudaMemcpy(d_a_half,a_half.data(),m*k*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_int,b_int.data(),k*n*sizeof(int8_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale,b_scale_half.data(),n*sizeof(half),cudaMemcpyHostToDevice);
    std::cout<<"=== do warm up for "<<kWarmTime<<" times"<<std::endl;
    for(int i=0;i<kWarmTime;i++){
        mixed_gemm_runner.gemm(
            reinterpret_cast<const half*>(d_a_half),
            reinterpret_cast<const uint8_t*>(d_b_int),
            reinterpret_cast<const half*>(d_b_scale),
            reinterpret_cast<half*>(d_c_half),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            0
        );
    }
    cudaDeviceSynchronize();
    auto start = std::chrono::system_clock::now();    
    for(int i=0;i<kTestTime;i++){
        mixed_gemm_runner.gemm(
            reinterpret_cast<const half*>(d_a_half),
            reinterpret_cast<const uint8_t*>(d_b_int),
            reinterpret_cast<const half*>(d_b_scale),
            reinterpret_cast<half*>(d_c_half),
            m,
            n,
            k,
            mixgemm_workspace_data,
            mixgemm_workspace_size_bytes,
            0
        );
    }
    cudaDeviceSynchronize();
    auto stop = std::chrono::system_clock::now();    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>((stop - start));
    std::cout<<"avg time for "<<kTestTime<<" run:"<<duration.count()/kTestTime<<std::endl;
    return 0;
}