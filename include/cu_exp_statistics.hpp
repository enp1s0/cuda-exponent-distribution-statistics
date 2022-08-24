#pragma once
#include <unordered_map>

namespace mtk {
namespace cu_exp_statistics {

struct result_t {
	std::size_t num_zero = 0;
	std::unordered_map<int, std::size_t> distribution;
};

template <class T>
result_t take_vector_statistics(
		const T* const ptr,
		const std::size_t size,
		cudaStream_t cuda_stream = 0
		);

template <class T>
result_t take_matrix_statistics(
		const T* const ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t ld = 0,
		cudaStream_t cuda_stream = 0
		);

std::string to_json(
		const result_t& result
		);

} // namespace cu_exp_statistics
} // namespace mtk
