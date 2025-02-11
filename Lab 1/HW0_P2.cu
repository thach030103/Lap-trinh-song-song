#include<iostream>
using namespace std;
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};
void addVecOnHost(float* in1, float* in2, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = in1[i] + in2[i];    
}

__global__ void addVecOnDevice(float* in1, float* in2, float* out, int n, int version)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (version == 1) {
        // Version 1:
        int j = i + blockIdx.x * blockDim.x; // Xử lý các phần tử khi N lớn 

        // Mỗi thread xử lý các phần tử cách nhau blockDim.x
        while (i < n) {
            if (i < n) {
                out[i] = in1[i] + in2[i]; // Phần tử đầu tiên
            }
            if (i + blockDim.x < n) {
                out[i + blockDim.x] = in1[i + blockDim.x] + in2[i + blockDim.x]; // Phần tử tiếp theo
            }
            i += j; // Di chuyển thread đến phần tử tiếp theo cách j vị trí
        }
    } else if (version == 2) {
        // Version 2: 
        int k = i * 2; 
        if (k < n) {
            out[k] = in1[k] + in2[k]; 
        }
        if (k + 1 < n) {
            out[k + 1] = in1[k + 1] + in2[k + 1]; 
        }
    }
}
void addVec(float* in1, float* in2, float* out, int n, int version, bool useDevice = false)
{
	GpuTimer timer;

	if (!useDevice)
    {
        timer.Start();
        addVecOnHost(in1, in2, out, n);
        timer.Stop();
    }
	else{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		cout<<"GPU name: "<<devProp.name<<endl;
		cout<<"GPU compute capability: "<<devProp.major<<"."<< devProp.minor<<endl;

		// Host allocates memories on device
		float *d_in1, *d_in2, *d_out;
		size_t nBytes = n * sizeof(float);
		CHECK(cudaMalloc(&d_in1, nBytes));
		CHECK(cudaMalloc(&d_in2, nBytes));
		CHECK(cudaMalloc(&d_out, nBytes));

		// Host copies data to device memories
		CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

		// Host invokes kernel function to add vectors on device
        dim3 blockSize(256);
        dim3 gridSize((n + (2 * blockSize.x) - 1) / (2 * blockSize.x)); 
        timer.Start();
        addVecOnDevice<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n, version); 
				
		cudaDeviceSynchronize(); 
		timer.Stop();
		// Host copies result from device memory
		CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_in1));
		CHECK(cudaFree(d_in2));
		CHECK(cudaFree(d_out));
	}
	
	float time = timer.Elapsed();
	cout << "Processing time (" << (useDevice ? "use version " + to_string(version) : "use host") << "): " << time << " ms" << endl;
}
int main(int argc, char ** argv)
{
    int vectorSizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    int numSizes = sizeof(vectorSizes)/sizeof(vectorSizes[0]);

    for(int i = 0; i < numSizes ; i++)
    {
        int N = vectorSizes[i];
        cout<<"Vector Size = "<< N <<endl;
        float *in1, *in2; // Input vectors
        float *out, *correctOut;  // Output vector

        // Allocate memories for in1, in2, out
        size_t nBytes = N * sizeof(float);
        in1 = (float *)malloc(nBytes);
        in2 = (float *)malloc(nBytes);
        out = (float *)malloc(nBytes);
        correctOut = (float *)malloc(nBytes);
        
        // Input data into in1, in2
        for (int i = 0; i < N; i++)
        {
            in1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            in2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
        
        // Add vectors (on host)
        addVec(in1, in2, correctOut, N, 0);

        // Add in1 & in2 on device using version 1
        addVec(in1, in2, out, N, 1, true);

        // Check correctness
        for (int i = 0; i < N; i++)
        {
            if (out[i] != correctOut[i])
            {
                cout<<"INCORRECT for version 1"<<endl;
                return 1;
            }
        }
        cout<<"CORRECT for version 1"<<endl;

        // Add in1 & in2 on device using version 2
        addVec(in1, in2, out, N, 2, true);
        
        // Check correctness for version 2
        for (int i = 0; i < N; i++)
        {
            if (out[i] != correctOut[i])
            {
                cout<<"INCORRECT for version 2"<<endl;
                return 1;
            }
        }
        cout<<"CORRECT for version 2"<<endl;
    }
}