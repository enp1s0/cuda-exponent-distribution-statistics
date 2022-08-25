#include <iostream>
#include <random>
#include <cu_exp_statistics.hpp>

template <class T>
struct base_t {
	using type = T;
	static const unsigned num_elements = 1;
};
template <> struct base_t<double2> {using type = double;static const unsigned num_elements = 2;};
template <> struct base_t<float2 > {using type = float; static const unsigned num_elements = 2;};

template <class T>
void test(const std::size_t m, const std::size_t n) {
	const auto size = m * n * base_t<T>::num_elements;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<typename base_t<T>::type> dist(-10, 10);

	T* test_matrix_ptr;
	cudaMallocManaged(&test_matrix_ptr, sizeof(T) * size);
	for (std::size_t i = 0; i < size; i++) {
		reinterpret_cast<typename base_t<T>::type*>(test_matrix_ptr)[i] = dist(mt);
	}
	cudaDeviceSynchronize();

	const auto result = mtk::cu_exp_statistics::take_matrix_statistics(
			test_matrix_ptr,
			m, n
			);
	std::printf("# zero = %lu\n", result.num_zero);
	std::printf("# inf = %lu\n", result.num_inf);
	std::printf("# nan = %lu\n", result.num_nan);
	for (int exp = -2048; exp <= 2048; exp++) {
		if (result.distribution.count(exp) != 0) {
			std::size_t count = result.distribution.at(exp);
			std::printf("[%+4d]: %lu\n", exp, count);
		}
	}

	std::printf("JSON: %s\n", mtk::cu_exp_statistics::to_json(result).c_str());

	cudaFree(test_matrix_ptr);
}

int main() {
	test<float  >(20000, 20000);
	test<double >(20000, 20000);
	test<float2 >(20000, 20000);
	test<double2>(20000, 20000);
}
