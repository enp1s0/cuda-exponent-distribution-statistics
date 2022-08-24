#include <iostream>
#include <random>
#include <cu_exp_statistics.hpp>

template <class T>
void test(const std::size_t m, const std::size_t n) {
	const auto size = m * n;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<T> dist(-10, 10);

	T* test_matrix_ptr;
	cudaMallocManaged(&test_matrix_ptr, sizeof(T) * size);
	for (std::size_t i = 0; i < size; i++) {
		test_matrix_ptr[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	const auto result = mtk::cu_exp_statistics::take_matrix_statistics(
			test_matrix_ptr,
			m, n
			);
	std::printf("# zero = %lu\n", result.num_zero);
	for (int exp = -2048; exp <= 2048; exp++) {
		if (result.distribution.count(exp) != 0) {
			std::size_t count = result.distribution.at(exp);
			std::printf("[%+4d]: %lu\n", exp, count);
		}
	}

	cudaFree(test_matrix_ptr);
}

int main() {
	test<float>(10000, 10000);
}
