#include "RdfIndexes.hpp"
#include "FinalRule.hpp"
#include "SupportCounting.hpp"
#include "RuleParser.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

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

        RuleParser parser(indexes);
        std::vector<FinalRule> rules = parser.parseRuleFile(rulesFile);

        auto rulesEnd = std::chrono::high_resolution_clock::now();

        std::cout << "\n=== Rules loaded ===\n";
        std::cout << "Rule count: " << rules.size() << "\n";

        auto supportStart = std::chrono::high_resolution_clock::now();

        SupportCounting counter(indexes);

        long long sumRuleMs = 0;

        for (std::size_t i = 0; i < rules.size(); ++i) {
            auto ruleStart = std::chrono::high_resolution_clock::now();

            counter.countSupport(rules[i]);

            auto ruleEnd = std::chrono::high_resolution_clock::now();
            auto ruleMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(ruleEnd - ruleStart).count();

            sumRuleMs += ruleMs;

            std::cout << "\nRule " << (i + 1) << ": ";
            printRule(rules[i], indexes);
            std::cout << " | Support: " << rules[i].measures.support
                      << ", HeadCoverage: " << rules[i].measures.headCoverage
                      << ", HeadSupport: " << rules[i].measures.headSupport
                      << ", HeadSize: " << rules[i].measures.headSize
                      << ", Time: " << ruleMs << " ms\n";
        }

        auto supportEnd = std::chrono::high_resolution_clock::now();
        auto totalEnd = std::chrono::high_resolution_clock::now();

        auto indexMs = std::chrono::duration_cast<std::chrono::milliseconds>(indexEnd - indexStart).count();
        auto rulesMs = std::chrono::duration_cast<std::chrono::milliseconds>(rulesEnd - rulesStart).count();
        auto supportMs = std::chrono::duration_cast<std::chrono::milliseconds>(supportEnd - supportStart).count();
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count();

        double avgRuleMs = rules.empty() ? 0.0 : static_cast<double>(sumRuleMs) / static_cast<double>(rules.size());

        std::cout << "\n=== Timing summary ===\n";
        std::cout << "Indexing time: " << indexMs << " ms\n";
        std::cout << "Rule parsing time: " << rulesMs << " ms\n";
        std::cout << "Support counting time: " << supportMs << " ms\n";
        std::cout << "Total time: " << totalMs << " ms\n";
        std::cout << "Average time per rule: " << avgRuleMs << " ms\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}