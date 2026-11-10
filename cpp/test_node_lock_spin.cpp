/**
 *  @file       test_node_lock_spin.cpp
 *  @brief      Regression tests for the bitset_gt node-lock spin, which starved its own release.
 *
 *  Root cause: the spin probed with atomic_set(), an acquire RMW. A sanitizer runtime maps the
 *  acquiring and the releasing RMW of one word onto a single internal reader/writer lock, taken
 *  shared for the acquire and exclusive for the release, and prefers its readers. The holder's
 *  atomic_reset() therefore has to wait for a moment when no prober holds that lock, and the wait
 *  for such a moment is starved for an unbounded time that grows with the number of concurrent
 *  probers. atomic_spin_set() probes with a relaxed load, which performs no synchronization at
 *  all, and only then attempts the acquire.
 *
 *  Every test here bounds itself and reports a failure rather than hanging, since the defect's
 *  natural symptom is a hang.
 *
 *  Compile (with TSAN, which is where the defect is observable):
 *    g++ -std=c++17 -fsanitize=thread -fno-omit-frame-pointer -O2 -g \
 *        -DUSEARCH_USE_SIMSIMD=0 -DUSEARCH_USE_FP16LIB=0 \
 *        -I../include test_node_lock_spin.cpp \
 *        -o test_node_lock_spin -lpthread
 *
 *  Measured release latency of a contended bit under TSAN, x86-64, clang 22, -O2:
 *    32 probers on 4 cores     before: over 60 s in 7 of 8 runs, 25 s in the eighth
 *                              after:  0.001 ms to 0.002 ms
 *    64 probers on 384 cores   before: over 60 s in 3 of 4 runs, over 5 s in the fourth
 *                              after:  0.009 ms to 0.024 ms
 *  Without a sanitizer both arms release in under 0.02 ms at both counts. With the defect present
 *  the 60 s deadline is the oracle that usually fires and the 5000 ms release threshold catches the
 *  shorter tail; at 16 probers neither fired in 4 of 8 runs, hence the floor of 32.
 */
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#define USEARCH_USE_SIMSIMD 0
#define USEARCH_USE_FP16LIB 0
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace unum::usearch;

static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT(cond)                                                                     \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__);   \
            tests_failed++;                                                              \
            return;                                                                      \
        }                                                                                \
    } while (0)

/// A hang is the defect's symptom, so an over-running test must kill the process instead of
/// waiting for a harness timeout that would report nothing.
class deadline_t {
    std::atomic<bool> finished_{false};
    std::thread thread_;

  public:
    deadline_t(char const* name, unsigned seconds) {
        thread_ = std::thread([this, name, seconds] {
            auto const until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
            while (std::chrono::steady_clock::now() < until)
                if (finished_.load(std::memory_order_relaxed))
                    return;
                else
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::fprintf(stderr, "\n  FAIL: %s did not finish within %u s\n", name, seconds);
            std::fflush(nullptr);
            std::_Exit(1);
        });
    }
    ~deadline_t() {
        finished_.store(true, std::memory_order_relaxed);
        thread_.join();
    }
};

static std::size_t prober_count() {
    std::size_t const hardware = std::thread::hardware_concurrency();
    std::size_t const wanted = hardware > 1 ? hardware * 2 : 16;
    return wanted < 32 ? 32 : (wanted > 64 ? 64 : wanted);
}

/// The regression test: a bit held while others probe it must still be releasable.
void test_release_is_not_starved_by_probes() {
    std::size_t const probers = prober_count();
    std::printf("  test_release_is_not_starved_by_probes (%zu probers) ... ", probers);
    std::fflush(stdout);
    deadline_t deadline("test_release_is_not_starved_by_probes", 60);

    bitset_t bits(64);
    std::atomic<bool> start{false};
    std::atomic<std::size_t> spinning{0};

    bits.atomic_spin_set(0);

    std::vector<std::thread> threads;
    threads.reserve(probers);
    for (std::size_t i = 0; i != probers; ++i)
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_relaxed))
                ;
            spinning.fetch_add(1, std::memory_order_relaxed);
            bits.atomic_spin_set(0);
            bits.atomic_reset(0);
        });

    start.store(true, std::memory_order_relaxed);
    while (spinning.load(std::memory_order_relaxed) != probers)
        ;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto const before = std::chrono::steady_clock::now();
    bits.atomic_reset(0);
    auto const elapsed = std::chrono::steady_clock::now() - before;

    for (auto& thread : threads)
        thread.join();

    auto const milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
    EXPECT(milliseconds < 5000.0);
    std::printf("PASS (release took %.3f ms)\n", milliseconds);
    tests_passed++;
}

/// The relaxed probe must not weaken the lock: exactly one holder at a time, no lost updates.
void test_spin_set_is_mutually_exclusive() {
    std::size_t const threads_count = 8;
    std::size_t const rounds = 4096;
    std::printf("  test_spin_set_is_mutually_exclusive ... ");
    std::fflush(stdout);
    deadline_t deadline("test_spin_set_is_mutually_exclusive", 60);

    bitset_t bits(64);
    std::size_t guarded_counter = 0;
    std::atomic<bool> inside{false};
    std::atomic<std::size_t> overlaps{0};

    std::vector<std::thread> threads;
    threads.reserve(threads_count);
    for (std::size_t i = 0; i != threads_count; ++i)
        threads.emplace_back([&] {
            for (std::size_t round = 0; round != rounds; ++round) {
                bits.atomic_spin_set(7);
                if (inside.exchange(true, std::memory_order_relaxed))
                    overlaps.fetch_add(1, std::memory_order_relaxed);
                ++guarded_counter;
                inside.store(false, std::memory_order_relaxed);
                bits.atomic_reset(7);
            }
        });
    for (auto& thread : threads)
        thread.join();

    EXPECT(overlaps.load() == 0);
    EXPECT(guarded_counter == threads_count * rounds);
    EXPECT(!bits.test(7));
    std::printf("PASS (%zu increments, no overlap)\n", guarded_counter);
    tests_passed++;
}

/// The HNSW node locks are the caller that stalled in production: build one index from many
/// threads and require every key to be findable afterwards.
void test_concurrent_index_add() {
    std::printf("  test_concurrent_index_add ... ");
    std::fflush(stdout);
    deadline_t deadline("test_concurrent_index_add", 300);

    using index_t = index_dense_gt<std::int64_t, std::uint32_t>;
    std::size_t const dimensions = 8;
    std::size_t const vectors = 4096;
    std::size_t const threads_count = 16;

    index_config_t config;
    config.connectivity = 8; // low connectivity concentrates contention near the entry point
    metric_punned_t metric(dimensions, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    index_t index = index_t::make(metric, config);
    index.reserve({vectors, threads_count});

    std::vector<float> data(vectors * dimensions);
    std::default_random_engine engine(7);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (auto& value : data)
        value = distribution(engine);

    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(threads_count);
    for (std::size_t t = 0; t != threads_count; ++t)
        threads.emplace_back([&, t] {
            for (;;) {
                std::size_t const i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= vectors)
                    return;
                if (!index.add(static_cast<std::int64_t>(i), data.data() + i * dimensions, t))
                    failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (auto& thread : threads)
        thread.join();

    EXPECT(failures.load() == 0);
    EXPECT(index.size() == vectors);
    for (std::size_t i = 0; i < vectors; i += 512) {
        auto found = index.search(data.data() + i * dimensions, 1);
        EXPECT(found.size() == 1);
    }
    std::printf("PASS (%zu vectors, %zu threads)\n", vectors, threads_count);
    tests_passed++;
}

int main() {
    std::printf("=== bitset_gt node-lock spin regression tests ===\n\n");
    test_release_is_not_starved_by_probes();
    test_spin_set_is_mutually_exclusive();
    test_concurrent_index_add();
    std::printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
