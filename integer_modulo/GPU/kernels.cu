// Addition
__global__ void add(int n, int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Classic modulo
__global__ void modulo(int n, int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] % b[i];
}