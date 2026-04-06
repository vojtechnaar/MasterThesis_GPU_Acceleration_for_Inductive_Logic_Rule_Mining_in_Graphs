#include "RdfIndexes.hpp"
#include "FinalRule.hpp"
#include "SupportCounting.hpp"
#include "RuleParser.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

static std::string termToString(const Term& t, const RdfIndexes& indexes) {
    if (t.isVariable()) {
        return "?" + std::to_string(t.value);
    }
    return indexes.mapper.getValue(t.value);
}

static void printRule(const FinalRule& rule, const RdfIndexes& indexes) {
    for (std::size_t i = 0; i < rule.body.size(); ++i) {
        const Atom& a = rule.body[i];
        std::cout << "( "
                  << termToString(a.subject, indexes) << " "
                  << indexes.mapper.getValue(a.predicate) << " "
                  << termToString(a.object, indexes) << " )";

        if (i + 1 < rule.body.size()) {
            std::cout << " ^ ";
        }
    }

    std::cout << " => ";

    const Atom& h = rule.head;
    std::cout << "( "
              << termToString(h.subject, indexes) << " "
              << indexes.mapper.getValue(h.predicate) << " "
              << termToString(h.object, indexes) << " )";
}

int main(int argc, char* argv[]) {
    std::string ttlFile = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";

    if (argc > 1) {
        ttlFile = argv[1];
    }
    if (argc > 2) {
        rulesFile = argv[2];
    }

    try {
        auto totalStart = std::chrono::high_resolution_clock::now();

        auto indexStart = std::chrono::high_resolution_clock::now();

        RdfIndexes indexes;
        if (!indexes.parseTurtleFile(ttlFile)) {
            std::cerr << "Failed to parse TTL file: " << ttlFile << "\n";
            return 1;
        }

        auto indexEnd = std::chrono::high_resolution_clock::now();

        std::cout << "=== Index built ===\n";
        indexes.printStats();

        auto rulesStart = std::chrono::high_resolution_clock::now();

        RuleParser parser(indexes, ttlFile);
        std::vector<FinalRule> rules = parser.parseRuleFile(rulesFile);

        auto rulesEnd = std::chrono::high_resolution_clock::now();

        std::cout << "\n=== Rules loaded ===\n";
        std::cout << "Rule count: " << rules.size() << "\n";
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

        auto supportStart = std::chrono::high_resolution_clock::now();

        SupportCounting counter(indexes);

        int nRules = static_cast<int>(rules.size());
        std::vector<long long> ruleTimes(nRules, 0);

        // Rule-level parallelism: each thread processes different rules simultaneously
        // Index is read-only, each rule writes only to its own FinalRule — no shared mutable state
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nRules; ++i) {
            auto ruleStart = std::chrono::high_resolution_clock::now();
            counter.countSupport(rules[i]);
            auto ruleEnd = std::chrono::high_resolution_clock::now();
            ruleTimes[i] = std::chrono::duration_cast<std::chrono::milliseconds>(ruleEnd - ruleStart).count();
        }

        auto supportEnd = std::chrono::high_resolution_clock::now();
        auto totalEnd = std::chrono::high_resolution_clock::now();

        // Print results sequentially (after all rules are done)
        long long sumRuleMs = 0;
        std::map<int, std::vector<long long>> timesByBodySize;

        for (int i = 0; i < nRules; ++i) {
            sumRuleMs += ruleTimes[i];
            timesByBodySize[static_cast<int>(rules[i].body.size())].push_back(ruleTimes[i]);

            std::cout << "\nRule " << (i + 1) << ": ";
            printRule(rules[i], indexes);
            std::cout << " | Support: " << rules[i].measures.support
                      << ", HeadCoverage: " << rules[i].measures.headCoverage
                      << ", HeadSupport: " << rules[i].measures.headSupport
                      << ", HeadSize: " << rules[i].measures.headSize
                      << ", Time: " << ruleTimes[i] << " ms\n";
        }

        auto indexMs = std::chrono::duration_cast<std::chrono::milliseconds>(indexEnd - indexStart).count();
        auto rulesMs = std::chrono::duration_cast<std::chrono::milliseconds>(rulesEnd - rulesStart).count();
        auto supportMs = std::chrono::duration_cast<std::chrono::milliseconds>(supportEnd - supportStart).count();
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

        double avgRuleMs = rules.empty() ? 0.0 : static_cast<double>(sumRuleMs) / static_cast<double>(nRules);

        // Compute detailed statistics
        std::vector<long long> sortedTimes(ruleTimes);
        std::sort(sortedTimes.begin(), sortedTimes.end());
        long long minMs = sortedTimes.empty() ? 0 : sortedTimes.front();
        long long maxMs = sortedTimes.empty() ? 0 : sortedTimes.back();
        long long medianMs = 0;
        long long p95Ms = 0;
        double stdDev = 0.0;

        if (!sortedTimes.empty()) {
            std::size_t n = sortedTimes.size();
            medianMs = (n % 2 == 1) ? sortedTimes[n / 2]
                                     : (sortedTimes[n / 2 - 1] + sortedTimes[n / 2]) / 2;
            std::size_t p95Idx = static_cast<std::size_t>(std::ceil(0.95 * n)) - 1;
            p95Ms = sortedTimes[std::min(p95Idx, n - 1)];

            double sumSqDiff = 0.0;
            for (auto t : sortedTimes) {
                double diff = static_cast<double>(t) - avgRuleMs;
                sumSqDiff += diff * diff;
            }
            stdDev = std::sqrt(sumSqDiff / static_cast<double>(n));
        }

        std::cout << "\n=== Timing summary ===\n";
        std::cout << "Indexing time:          " << indexMs << " ms\n";
        std::cout << "Rule parsing time:      " << rulesMs << " ms\n";
        std::cout << "Support counting time:  " << supportMs << " ms\n";
        std::cout << "Total time:             " << totalMs << " ms\n";
        std::cout << "\n=== Per-rule statistics (" << nRules << " rules) ===\n";
        std::cout << "Average: " << avgRuleMs << " ms\n";
        std::cout << "Median:  " << medianMs << " ms\n";
        std::cout << "Std Dev: " << stdDev << " ms\n";
        std::cout << "Min:     " << minMs << " ms\n";
        std::cout << "Max:     " << maxMs << " ms\n";
        std::cout << "P95:     " << p95Ms << " ms\n";

        // Per-body-size statistics
        std::cout << "\n=== Statistics by body size ===\n";
        for (auto& [bodySize, times] : timesByBodySize) {
            std::sort(times.begin(), times.end());
            std::size_t n = times.size();
            long long sum = 0;
            for (auto t : times) sum += t;
            double avg = static_cast<double>(sum) / static_cast<double>(n);
            long long med = (n % 2 == 1) ? times[n / 2]
                                          : (times[n / 2 - 1] + times[n / 2]) / 2;
            std::size_t p95i = static_cast<std::size_t>(std::ceil(0.95 * n)) - 1;
            long long p95 = times[std::min(p95i, n - 1)];

            std::cout << "Body size " << bodySize
                      << " (" << n << " rules): "
                      << "avg=" << avg << " ms, "
                      << "median=" << med << " ms, "
                      << "min=" << times.front() << " ms, "
                      << "max=" << times.back() << " ms, "
                      << "P95=" << p95 << " ms\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}