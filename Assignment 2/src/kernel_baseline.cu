#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "defs.h"

using namespace std;

#define BLOCK_SIZE 1024

__device__ bool byteCharsMatch(uint8_t byte, char c1, char c2);

__global__ void matchFile(const uint8_t* file_data, size_t file_len, const char* signature, size_t len, int sigIdx, bool* matchArr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t sig_len = len / 2;
	
	if (idx + sig_len > file_len) {
		return;
	}
	bool match = true;
	for (size_t i = 0; i < sig_len; i++) {
		if (!byteCharsMatch(file_data[idx + i], signature[2*i], signature[2*i+1])) {
			match = false;
			break;
		}
	}
	if (match) {
		matchArr[sigIdx] = 1;
	}
}

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& inputs)
{
	{
		cudaDeviceProp prop;
		check_cuda_error(cudaGetDeviceProperties(&prop, 0));

		fprintf(stderr, "cuda stats:\n");
		fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
		fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
		fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
		fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
	}

	/*
		Here, we are creating one stream per file just for demonstration purposes;
		you should change this to fit your own algorithm and/or implementation.
	*/
	std::vector<cudaStream_t> streams {};
	streams.resize(inputs.size());

	std::vector<uint8_t*> file_bufs {};
	for(size_t i = 0; i < inputs.size(); i++)
	{
		cudaStreamCreate(&streams[i]);

		// allocate memory on the device for the file
		uint8_t* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
		file_bufs.push_back(ptr);
	}

	// allocate memory for the signatures
	std::vector<char*> sig_bufs {};
	for(size_t i = 0; i < signatures.size(); i++)
	{
		char* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, signatures[i].size));
		cudaMemcpy(ptr, signatures[i].data, signatures[i].size, cudaMemcpyHostToDevice);
		sig_bufs.push_back(ptr);
	}

	vector<bool*> results;
	results.resize(inputs.size());

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		// asynchronously copy the file contents from host memory
		// (the `inputs`) to device memory (file_bufs, which we allocated above)
		cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data, inputs[file_idx].size,
			cudaMemcpyHostToDevice, streams[file_idx]);    // pass in the stream here to do this async

		bool* matchArr;
		check_cuda_error(cudaMallocManaged(&matchArr, sizeof(bool)*signatures.size()));

		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			size_t n_tasks = (inputs[file_idx].size - (signatures[sig_idx].size)/2 + 1);
			size_t grid_size = (n_tasks + BLOCK_SIZE - 1) / BLOCK_SIZE;

			matchFile<<<grid_size, BLOCK_SIZE, /* shared memory per block: */ 0, streams[file_idx]>>>(
				file_bufs[file_idx], inputs[file_idx].size,
				sig_bufs[sig_idx], signatures[sig_idx].size, sig_idx, matchArr);
		}

		// wait for the stream to finish
		cudaStreamSynchronize(streams[file_idx]);

		results[file_idx] = matchArr;
	}

	cudaDeviceSynchronize();

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			if (results[file_idx][sig_idx]) {
				printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
			}
		}
	}

	// free the device memory, though this is not strictly necessary
	// (the CUDA driver will clean up when your program exits)
	for(auto buf : file_bufs)
		cudaFree(buf);

	for(auto buf : sig_bufs)
		cudaFree(buf);

	// clean up streams (again, not strictly necessary)
	for(auto& s : streams)
		cudaStreamDestroy(s);
}

__device__ bool byteCharsMatch(uint8_t byte, char c1, char c2)
{
	const uint8_t c1_byte = c1 >= '0' && c1 <= '9' ? c1 - '0' : c1 - 'a' + 10;
	const uint8_t c2_byte = c2 >= '0' && c2 <= '9' ? c2 - '0' : c2 - 'a' + 10;
	if (c1 == '?' && c2 == '?') {
		return true;
	} else if (c1 == '?') {
		return (byte & 0x0f) == c2_byte;
	} else if (c2 == '?') {
		return (byte >> 4) == c1_byte;
	} else {
		return byte == ((c1_byte << 4) | c2_byte);
	}
}
