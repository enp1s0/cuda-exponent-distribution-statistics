#include <iostream>
#include <cu_exp_statistics.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/memory.hpp>

namespace {
using count_t = unsigned long long int;

template <class T>
__global__ void statistics_kernel(
		count_t* const result_ptr,
		const T* const ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t ld
		) {
	const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= m * n) {
		return;
	}

	const auto im = tid % m;
	const auto in = tid / m;

	const auto mem_index = im + in * ld;
	const auto value = ptr[mem_index];

	if ((cutf::experimental::fp::reinterpret_as_uint(value) << 1) == 0) {
		atomicAdd(result_ptr, 1lu);
		return;
	}

	const auto exp = cutf::experimental::fp::mask_exponent(value) >> cutf::experimental::fp::get_mantissa_size<T>();

	atomicAdd(result_ptr + exp + 1, 1lu);
}

__global__ void init_array_kernel(
		count_t* const ptr,
		const std::size_t size
		) {
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= size) {
		return;
	}

	ptr[tid] = 0;
}
} // unnamed namespace

template <class T>
mtk::cu_exp_statistics::result_t mtk::cu_exp_statistics::take_vector_statistics(
		const T* const ptr,
		const std::size_t size,
		cudaStream_t cuda_stream
		) {
	return mtk::cu_exp_statistics::take_matrix_statistics(
			ptr,
			size,
			1,
			1
			);
}

template <class T>
mtk::cu_exp_statistics::result_t mtk::cu_exp_statistics::take_matrix_statistics(
		const T* const ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t ld_,
		cudaStream_t cuda_stream
		) {
	const std::size_t ld = ld_ == 0 ? m : ld_;

	const std::size_t statistics_array_size = (1lu << cutf::experimental::fp::get_exponent_size<T>()) + 1;
	count_t *dev_count;
	count_t *hos_count;
	CUTF_CHECK_ERROR(cudaMalloc(&dev_count, sizeof(count_t) * statistics_array_size));
	CUTF_CHECK_ERROR(cudaMallocHost(&hos_count, sizeof(count_t) * statistics_array_size));

	const std::size_t size = m * n;
	const auto block_size = 256;
	const auto grid_size = (size + block_size - 1) / block_size;

	init_array_kernel<<<grid_size, block_size, 0, cuda_stream>>>(dev_count, statistics_array_size);
	statistics_kernel<<<grid_size, block_size, 0, cuda_stream>>>(dev_count, ptr, m, n, ld);

	CUTF_CHECK_ERROR(cudaMemcpyAsync(hos_count, dev_count, sizeof(count_t) * statistics_array_size, cudaMemcpyDefault, cuda_stream));
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

	mtk::cu_exp_statistics::result_t result;
	result.num_zero = hos_count[0];

	for (std::uint32_t i = 0; i < (1u << cutf::experimental::fp::get_exponent_size<T>()); i++) {
		if (hos_count[i + 1] != 0) {
			result.distribution.insert(std::make_pair(
						static_cast<int>(i) - cutf::experimental::fp::get_bias<T>(),
						hos_count[i + 1]
						));
		}
	}

	CUTF_CHECK_ERROR(cudaFreeHost(hos_count));
	CUTF_CHECK_ERROR(cudaFree(dev_count));

	return result;
}

std::string mtk::cu_exp_statistics::to_json(
		const mtk::cu_exp_statistics::result_t& result
		) {
	std::string str = "{";
	str += "num_zero:" + std::to_string(result.num_zero);

	for (int exp = -10000; exp <= 10000; exp++) {
		if (result.distribution.count(exp) != 0) {
			str += ",\"" + std::to_string(exp) + "\":" + std::to_string(result.distribution.at(exp));
		}
	}
	str += "}";

	return str;
}

#define TAKE_MATRIX_STATISTICS_INSTANCE(type)\
	template mtk::cu_exp_statistics::result_t mtk::cu_exp_statistics::take_matrix_statistics<type>( \
		const type* const ptr, \
		const std::size_t m, \
		const std::size_t n, \
		const std::size_t ld, \
		cudaStream_t cuda_stream \
		)
#define TAKE_VECTOR_STATISTICS_INSTANCE(type)\
	template mtk::cu_exp_statistics::result_t mtk::cu_exp_statistics::take_vector_statistics<type>( \
		const type* const ptr, \
		const std::size_t size, \
		cudaStream_t cuda_stream \
		)

TAKE_MATRIX_STATISTICS_INSTANCE(half  );
TAKE_MATRIX_STATISTICS_INSTANCE(float );
TAKE_MATRIX_STATISTICS_INSTANCE(double);
TAKE_VECTOR_STATISTICS_INSTANCE(half  );
TAKE_VECTOR_STATISTICS_INSTANCE(float );
TAKE_VECTOR_STATISTICS_INSTANCE(double);
