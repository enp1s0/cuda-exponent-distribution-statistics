#pragma once
#include <string>
#include <unordered_map>

namespace mtk {
namespace cu_exp_statistics {

struct result_t {
	std::size_t num_zero = 0;
	std::size_t num_inf = 0;
	std::size_t num_nan = 0;
	std::unordered_map<int, std::size_t> distribution;

	inline result_t operator+=(const result_t& res) {
		result_t& self = *this;
		self.num_inf += res.num_inf;
		self.num_nan += res.num_nan;
		self.num_zero += res.num_zero;

		for (int exp = -2048; exp <= 2048; exp++) {
			if (self.distribution.count(exp) && res.distribution.count(exp)) {
				self.distribution[exp] += res.distribution.at(exp);
			} else if (res.distribution.count(exp)) {
				self.distribution.insert(std::make_pair(exp, res.distribution.at(exp)));
			}
		}

		return self;
	}

	inline result_t operator+(const result_t& res) const {
		result_t self = *this;
		self += res;

		return self;
	}
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
