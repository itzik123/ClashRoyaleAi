// Verifies UPSTREAM item 36's proposed fix: random primitives built ONLY on raw
// std::mt19937 output give the same draws on every standard library.
//
// Self-contained on purpose: it carries the proposed header's body inline, so
// it can be run on MSVC before anyone edits include/. Diff the output of two
// builds; it must be identical, where seed_portability_probe.cpp's is not.
//
//   bash tools/audit/build.sh portable_random_check                 # libstdc++
//   clang++ -std=c++17 -O2 -stdlib=libc++ tools/audit/portable_random_check.cpp \
//       -o tools/audit/bin/portable_random_check_libcxx            # libc++
//   powershell -File tools/audit/build.ps1 portable_random_check     # MSVC
//
// Measured 2026-11-10: identical under libstdc++ (g++ 13) and libc++ (clang 18).
// MSVC not yet run.
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace portable {

// Uniform in [0, n), n >= 1. Rejection sampling: draws below 2^32 mod n are
// discarded, so the accepted range is an exact multiple of n and x % n is
// unbiased.
inline std::uint32_t uniformIndex(std::mt19937& rng, std::uint32_t n) {
    const std::uint32_t reject_below = (0u - n) % n;   // == 2^32 mod n
    std::uint32_t x;
    do {
        x = static_cast<std::uint32_t>(rng());
    } while (x < reject_below);
    return x % n;
}

// Uniform in [0, 1): the top 24 bits, exactly representable in a float.
inline float uniform01(std::mt19937& rng) {
    return static_cast<float>(static_cast<std::uint32_t>(rng()) >> 8)
           * (1.0f / 16777216.0f);
}

// Fisher-Yates over uniformIndex.
template <class T>
void shuffle(std::vector<T>& v, std::mt19937& rng) {
    for (std::size_t i = v.size(); i > 1; --i) {
        const std::size_t j = uniformIndex(rng, static_cast<std::uint32_t>(i));
        std::swap(v[i - 1], v[j]);
    }
}

}  // namespace portable
#include <cstdio>
#include <numeric>
int main() {
    for (unsigned s : {0u, 7u, 4242u}) {
        std::mt19937 g(s);
        std::vector<int> v(8);
        std::iota(v.begin(), v.end(), 0);
        portable::shuffle(v, g);
        std::printf("shuffle seed %u:", s);
        for (int x : v) std::printf(" %d", x);
        std::printf("\n");
    }
    std::mt19937 g(7);
    std::printf("uniformIndex(100):");
    for (int i = 0; i < 6; ++i) { const unsigned v = portable::uniformIndex(g, 100); std::printf(" %u", v); }
    std::printf("\nuniform01:");
    for (int i = 0; i < 3; ++i) { const float f = portable::uniform01(g); std::printf(" %.7f", f); }
    std::printf("\n");
    // Uniformity sanity: 600k draws over 6 bins.
    std::mt19937 h(1);
    long bins[6] = {0};
    for (int i = 0; i < 600000; ++i) ++bins[portable::uniformIndex(h, 6)];
    std::printf("bins:");
    for (long b : bins) std::printf(" %ld", b);
    std::printf("\n");
}
