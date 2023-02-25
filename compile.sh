cudaroot=/home/workplace/cuda-11.6/
${cudaroot}/bin/nvcc -gencode arch=compute_86,code=sm_86 -I${cudaroot}/include -L${cudaroot}/lib64 \
                 -I`pwd`/cutlass/include -I`pwd` -I${cudaroot}/ \
fpAintB_test.cu -o fpAintB_test