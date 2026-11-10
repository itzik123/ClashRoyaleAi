// Does a seed name the same deck and the same opening hand on every build?
// UPSTREAM_REQUESTS.md item 36.
//
// std::mt19937 is pinned by the standard; the distributions and std::shuffle
// built on it are not, and unordered_map iteration order is not either. This
// prints each layer separately so two builds can be diffed line by line:
//
//   bash tools/audit/build.sh seed_portability_probe     # libstdc++
//   clang++ -std=c++20 -O1 -stdlib=libc++ -Iinclude/core -Iinclude/entities \
//       -Iinclude/rendering tools/audit/seed_portability_probe.cpp \
//       -o tools/audit/bin/seed_portability_probe_libcxx
//   powershell -File tools/audit/build.ps1 seed_portability_probe   # MSVC
//
// A line that differs between two builds is a layer that is not portable.
// Every value is printed in its own statement: function-argument evaluation
// order is itself unspecified, and printing two draws as two arguments of one
// call reorders them between compilers.
#include "ClashEnv.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

int main() {
    const unsigned seeds[] = {0u, 7u, 4242u};

    std::printf("raw mt19937(7):");
    {
        std::mt19937 g(7);
        for (int i = 0; i < 3; ++i) {
            const unsigned long v = g();
            std::printf(" %lu", v);
        }
    }
    std::printf("   <- pinned by the standard; must match everywhere\n");

    std::printf("uniform_int(0,99) on mt19937(7):");
    {
        std::mt19937 g(7);
        std::uniform_int_distribution<size_t> u(0, 99);
        for (int i = 0; i < 6; ++i) {
            const size_t v = u(g);
            std::printf(" %zu", v);
        }
    }
    std::printf("\n");

    for (unsigned s : seeds) {
        std::mt19937 g(s);
        std::vector<size_t> order(8);
        std::iota(order.begin(), order.end(), size_t{0});
        std::shuffle(order.begin(), order.end(), g);
        std::printf("std::shuffle of 0..7, seed %u:", s);
        for (size_t v : order) std::printf(" %zu", v);
        std::printf("   <- PlayerState::initializeDeck, i.e. env.seed()\n");
    }

    const std::vector<int> ids = getAllCardIds();
    std::printf("getAllCardIds()[:10]:");
    for (size_t i = 0; i < 10 && i < ids.size(); ++i) std::printf(" %d", ids[i]);
    std::printf("\n");

    for (unsigned s : seeds) {
        std::mt19937 g(s);
        const std::vector<int> deck = sampleRandomDeck(g);
        std::printf("sampleRandomDeck, seed %u:", s);
        for (int c : deck) std::printf(" %d", c);
        std::printf("   <- sample_random_deck(seed=%u)\n", s);
    }
    return 0;
}
