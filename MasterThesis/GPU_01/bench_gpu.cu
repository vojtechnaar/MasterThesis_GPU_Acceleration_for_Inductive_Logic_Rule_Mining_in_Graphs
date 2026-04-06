/*  bench_gpu.cu — GPU rule-support counting (binary search + warp-coop)
 *
 *  Uses CPU_01's data structures (FinalRule, RdfIndexes, RuleParser)
 *  for rule parsing and index loading, then converts to GPU format.
 *
 *  Compile:
 *    nvcc -O3 -std=c++17 bench_gpu.cu FinalRule.cpp RdfIndexes.cpp \
 *         RuleParser.cpp $(pkg-config --cflags --libs raptor2) -o bench_gpu
 *
 *  Run:
 *    ./bench_gpu <data.ttl> <rules.txt>
 *
 *  File layout:
 *    FinalRule.hpp/cpp   — Term, Atom, Measures, FinalRule (shared with CPU_01)
 *    RdfIndexes.hpp/cpp  — IdMapper, PredIndex, RdfIndexes (shared with CPU_01)
 *    RuleParser.hpp/cpp  — RuleParser with prefix expansion (shared with CPU_01)
 *    gpu_common.cuh      — CUDA_OK macro, Clock, algorithm constants
 *    gpu_kernels.cuh     — AtomGPU, binary search, DFS, kernels, launch helpers
 *    gpu_index.cuh       — PredHost/PredDev, build from RdfIndexes, upload
 *    bench_gpu.cu        — main() entry point (this file)
 */

#include "gpu_kernels.cuh"
#include "gpu_index.cuh"
#include "FinalRule.hpp"
#include "RuleParser.hpp"

#include <cmath>
#include <climits>
#include <map>
#include <iostream>

/* ═══════════════════════════════════════════════
 *  Convert FinalRule to GPU-friendly format
 * ═══════════════════════════════════════════════ */

/* GPU-side body atom (variable slots only) */
struct GpuBodyAtom { int pred, sVar, oVar; };

/* GPU-side rule representation */
struct GpuRule {
    int headPred, hS, hO;
    std::vector<GpuBodyAtom> body;
    int numVars;
};

/* Convert a FinalRule (CPU_01 format) to a GpuRule.
 * Variables keep their original slot IDs.
 * Constants get "virtual" variable slots (needed for GPU kernel's
 * flat v[] array), but this only matters if rules have constant
 * subject/object terms — our dataset uses variables only. */
static GpuRule convert_rule(const FinalRule& rule) {
    GpuRule gr;
    gr.headPred = rule.head.predicate;

    /* find max variable ID in the rule */
    int maxId = -1;
    auto track = [&](const Term& t) {
        if (t.isVariable() && t.value > maxId) maxId = t.value;
    };
    track(rule.head.subject);
    track(rule.head.object);
    for (auto& b : rule.body) {
        track(b.subject);
        track(b.object);
    }
    int nextSlot = maxId + 1;

    /* head variable slots */
    gr.hS = rule.head.subject.isVariable() ? rule.head.subject.value : nextSlot++;
    gr.hO = rule.head.object.isVariable()  ? rule.head.object.value  : nextSlot++;

    /* body atoms */
    for (auto& atom : rule.body) {
        GpuBodyAtom ba;
        ba.pred = atom.predicate;
        ba.sVar = atom.subject.isVariable() ? atom.subject.value : nextSlot++;
        ba.oVar = atom.object.isVariable()  ? atom.object.value  : nextSlot++;
        gr.body.push_back(ba);
    }

    gr.numVars = nextSlot;
    return gr;
}

/* ═══════════════════════════════════════════════
 *  Body-atom ordering (maxfan-aware greedy)
 * ═══════════════════════════════════════════════ */

struct Ordered { int pred; int keySlot, valSlot, chkSlot; bool useSPO; };

static std::vector<Ordered> order_body(
        const GpuRule& r,
        const std::unordered_map<int, PredHost>& idx)
{
    int nb = (int)r.body.size();
    std::vector<bool> used(nb, false);
    std::vector<bool> bound(MAXVAR, false);
    bound[r.hS] = true;
    bound[r.hO] = true;

    std::vector<Ordered> out;

    for (int step = 0; step < nb; ++step) {
        int best = -1, bSh = -1, bCnt = INT_MAX;
        for (int i = 0; i < nb; ++i) {
            if (used[i]) continue;
            auto& b = r.body[i];
            int sh = (bound[b.sVar] ? 1 : 0) + (bound[b.oVar] ? 1 : 0);
            if (sh == 0) continue;
            int c = 0;
            auto it = idx.find(b.pred);
            if (it != idx.end()) {
                if (sh == 2) {
                    c = (int)it->second.spo.size();
                } else {
                    bool sB = bound[b.sVar];
                    c = sB ? it->second.spo_maxfan : it->second.pos_maxfan;
                }
            }
            if (sh > bSh || (sh == bSh && c < bCnt)) {
                best = i; bSh = sh; bCnt = c;
            }
        }
        if (best < 0) {
            for (int i = 0; i < nb; ++i)
                if (!used[i]) { best = i; break; }
        }
        if (best < 0) break;

        used[best] = true;
        auto& b = r.body[best];
        Ordered o{};
        o.pred = b.pred;
        bool sB = bound[b.sVar], oB = bound[b.oVar];

        if (sB && oB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = -1;
            o.chkSlot = b.oVar;
        } else if (sB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = b.oVar;
            o.chkSlot = -1;
            bound[b.oVar] = true;
        } else {
            o.useSPO  = false;
            o.keySlot = b.oVar;
            o.valSlot = b.sVar;
            o.chkSlot = -1;
            bound[b.sVar] = true;
        }
        out.push_back(o);
    }
    return out;
}

/* ═══════════════════════════════════════════════
 *  Printing helpers (same format as CPU_01)
 * ═══════════════════════════════════════════════ */

static std::string termToString(const Term& t, const RdfIndexes& indexes) {
    if (t.isVariable())
        return "?" + std::to_string(t.value);
    return indexes.mapper.getValue(t.value);
}

static void printRule(const FinalRule& rule, const RdfIndexes& indexes) {
    for (std::size_t i = 0; i < rule.body.size(); ++i) {
        const Atom& a = rule.body[i];
        std::cout << "( "
                  << termToString(a.subject, indexes) << " "
                  << indexes.mapper.getValue(a.predicate) << " "
                  << termToString(a.object, indexes) << " )";
        if (i + 1 < rule.body.size())
            std::cout << " ^ ";
    }
    std::cout << " => ";
    const Atom& h = rule.head;
    std::cout << "( "
              << termToString(h.subject, indexes) << " "
              << indexes.mapper.getValue(h.predicate) << " "
              << termToString(h.object, indexes) << " )";
}

/* ══════════════════════  main  ══════════════════════ */
int main(int argc, char** argv) {
    std::string ttlFile = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";
    if (argc > 1) ttlFile = argv[1];
    if (argc > 2) rulesFile = argv[2];

    try {
        auto T0 = Clock::now();

        /* ── Load RDF via RdfIndexes (same as CPU_01) ── */
        RdfIndexes indexes;
        if (!indexes.parseTurtleFile(ttlFile)) {
            fprintf(stderr, "Failed to parse TTL file: %s\n", ttlFile.c_str());
            return 1;
        }
        auto T1 = Clock::now();

        std::cout << "=== Index built ===\n";
        indexes.printStats();

        /* ── Build GPU index from RdfIndexes ── */
        auto hIdx = build_gpu_index(indexes);
        auto T2 = Clock::now();
        printf("GPU index built: %zu predicates  (%.0f ms)\n",
               hIdx.size(), ms_between(T1, T2));

        /* ── Upload to GPU ── */
        auto dIdx = upload(hIdx);
        auto T3 = Clock::now();
        printf("GPU upload done  (%.0f ms)\n", ms_between(T2, T3));

        /* ── Parse rules (same as CPU_01) ── */
        RuleParser parser(indexes, ttlFile);
        std::vector<FinalRule> rules = parser.parseRuleFile(rulesFile);
        auto T4 = Clock::now();

        std::cout << "\n=== Rules loaded ===\n";
        std::cout << "Rule count: " << rules.size() << "\n\n";

        /* ── Convert rules to GPU format ── */
        int nRules = (int)rules.size();
        std::vector<GpuRule> gpuRules(nRules);
        for (int i = 0; i < nRules; ++i) {
            gpuRules[i] = convert_rule(rules[i]);
            if (gpuRules[i].numVars > MAXVAR) {
                fprintf(stderr, "Warning: rule %d has %d vars (max %d), will skip\n",
                        i + 1, gpuRules[i].numVars, MAXVAR);
            }
        }

        /* ── Allocate device result + CUDA events ── */
        unsigned long long* d_res;
        CUDA_OK(cudaMalloc(&d_res, sizeof(unsigned long long)));
        cudaEvent_t evStart, evStop;
        CUDA_OK(cudaEventCreate(&evStart));
        CUDA_OK(cudaEventCreate(&evStop));

        /* ── Process rules sequentially ── */
        auto supportStart = Clock::now();

        std::vector<double> ruleTimesMs(nRules, 0.0);
        int nCoop = 0;

        for (int ri = 0; ri < nRules; ++ri) {
            FinalRule& rule = rules[ri];
            GpuRule& gr = gpuRules[ri];
            int nb = (int)gr.body.size();

            /* head predicate info */
            const PredIndex* pi = indexes.getPred(rule.head.predicate);
            int headSize = pi ? pi->totalPairs() : 0;
            rule.measures.headSize = headSize;

            /* headSupport (same logic as CPU_01) */
            int headSupport = 0;
            if (pi) {
                bool sBound = rule.head.subject.isConstant();
                bool oBound = rule.head.object.isConstant();
                if (!sBound && !oBound) {
                    headSupport = pi->totalPairs();
                } else if (sBound && !oBound) {
                    int cnt = 0;
                    pi->spoRange(rule.head.subject.value, cnt);
                    headSupport = cnt;
                } else if (!sBound && oBound) {
                    int cnt = 0;
                    pi->posRange(rule.head.object.value, cnt);
                    headSupport = cnt;
                } else {
                    headSupport = pi->hasTriple(rule.head.subject.value,
                                                rule.head.object.value) ? 1 : 0;
                }
            }
            rule.measures.headSupport = headSupport;

            /* skip if no head triples or too many body atoms */
            auto headIt = hIdx.find(rule.head.predicate);
            int headN = headIt != hIdx.end() ? (int)headIt->second.spo.size() : 0;

            if (nb < 1 || nb > MAX_BODY || headN == 0 || gr.numVars > MAXVAR) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            auto devIt = dIdx.find(rule.head.predicate);
            if (devIt == dIdx.end() || devIt->second.spo_n == 0) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* order body atoms */
            auto ord = order_body(gr, hIdx);
            if ((int)ord.size() != nb) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* check all body predicates exist on device */
            bool skip = false;
            for (auto& o : ord) {
                auto di = dIdx.find(o.pred);
                if (di == dIdx.end()) { skip = true; break; }
                int cnt = o.useSPO ? di->second.spo_n : di->second.pos_n;
                if (cnt == 0) { skip = true; break; }
            }
            if (skip) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* decide: standard vs warp-cooperative */
            bool useCoop = false;
            if (ord[0].valSlot >= 0) {
                auto pit = hIdx.find(ord[0].pred);
                if (pit != hIdx.end()) {
                    int mf = ord[0].useSPO ? pit->second.spo_maxfan
                                           : pit->second.pos_maxfan;
                    useCoop = (mf > COOP_FAN);
                }
            }

            /* upload atom descriptors to __constant__ */
            AtomGPU atoms[MAX_BODY];
            for (int i = 0; i < nb; ++i) {
                auto& o  = ord[i];
                auto& dp = dIdx[o.pred];
                atoms[i].arr     = o.useSPO ? dp.spo : dp.pos;
                atoms[i].n       = o.useSPO ? dp.spo_n : dp.pos_n;
                atoms[i].keySlot = o.keySlot;
                atoms[i].valSlot = o.valSlot;
                atoms[i].chkSlot = o.chkSlot;
            }
            CUDA_OK(cudaMemcpyToSymbol(c_atoms, atoms, nb * sizeof(AtomGPU)));
            CUDA_OK(cudaMemset(d_res, 0, sizeof(unsigned long long)));

            int blk = 256;

            /* injective mapping: skip head triples where two distinct
               variables would be mapped to the same entity */
            int ssl = (rule.head.subject.isVariable() &&
                       rule.head.object.isVariable() &&
                       rule.head.subject.value != rule.head.object.value) ? 1 : 0;

            CUDA_OK(cudaEventRecord(evStart));
            if (useCoop) {
                int wpb = blk >> 5;
                int grd = std::min((headN + wpb - 1) / wpb, 2048);
                launch_coop(nb, grd, blk, devIt->second.spo, headN,
                            gr.hS, gr.hO, ssl, d_res);
                ++nCoop;
            } else {
                int grd = std::min((headN + blk - 1) / blk, 2048);
                launch_std(nb, grd, blk, devIt->second.spo, headN,
                           gr.hS, gr.hO, ssl, d_res);
            }
            CUDA_OK(cudaEventRecord(evStop));
            CUDA_OK(cudaEventSynchronize(evStop));

            unsigned long long sup;
            CUDA_OK(cudaMemcpy(&sup, d_res, sizeof(sup), cudaMemcpyDeviceToHost));
            float gMs;
            CUDA_OK(cudaEventElapsedTime(&gMs, evStart, evStop));

            rule.setMeasures((int)sup, headSize, headSupport);
            ruleTimesMs[ri] = (double)gMs;
        }

        auto supportEnd = Clock::now();
        auto totalEnd   = Clock::now();

        /* ── Print per-rule results (same format as CPU_01) ── */
        double sumRuleMs = 0.0;
        std::map<int, std::vector<double>> timesByBodySize;

        for (int i = 0; i < nRules; ++i) {
            sumRuleMs += ruleTimesMs[i];
            timesByBodySize[(int)rules[i].body.size()].push_back(ruleTimesMs[i]);

            std::cout << "\nRule " << (i + 1) << ": ";
            printRule(rules[i], indexes);
            std::cout << " | Support: " << rules[i].measures.support
                      << ", HeadCoverage: " << rules[i].measures.headCoverage
                      << ", HeadSupport: " << rules[i].measures.headSupport
                      << ", HeadSize: " << rules[i].measures.headSize
                      << ", Time: " << ruleTimesMs[i] << " ms\n";
        }

        /* ── Timing summary ── */
        auto indexMs   = ms_between(T0, T1);
        auto gpuIdxMs  = ms_between(T1, T2);
        auto uploadMs  = ms_between(T2, T3);
        auto rulesMs   = ms_between(T3, T4);
        auto supMs     = ms_between(supportStart, supportEnd);
        auto totalMs   = ms_between(T0, totalEnd);
        double avgMs   = nRules > 0 ? sumRuleMs / nRules : 0.0;

        std::vector<double> sortedTimes(ruleTimesMs);
        std::sort(sortedTimes.begin(), sortedTimes.end());
        double minMs = sortedTimes.empty() ? 0.0 : sortedTimes.front();
        double maxMs = sortedTimes.empty() ? 0.0 : sortedTimes.back();
        double medianMs = 0.0, p95Ms = 0.0, stdDev = 0.0;

        if (!sortedTimes.empty()) {
            size_t n = sortedTimes.size();
            medianMs = (n % 2 == 1) ? sortedTimes[n / 2]
                                     : (sortedTimes[n / 2 - 1] + sortedTimes[n / 2]) / 2.0;
            size_t p95Idx = (size_t)(std::ceil(0.95 * n)) - 1;
            p95Ms = sortedTimes[std::min(p95Idx, n - 1)];
            double sumSqDiff = 0.0;
            for (auto t : sortedTimes) {
                double diff = t - avgMs;
                sumSqDiff += diff * diff;
            }
            stdDev = std::sqrt(sumSqDiff / (double)n);
        }

        std::cout << "\n=== Timing summary ===\n";
        std::cout << "Indexing time:          " << indexMs << " ms\n";
        std::cout << "GPU index build time:   " << gpuIdxMs << " ms\n";
        std::cout << "GPU upload time:        " << uploadMs << " ms\n";
        std::cout << "Rule parsing time:      " << rulesMs << " ms\n";
        std::cout << "Support counting time:  " << supMs << " ms\n";
        std::cout << "Total time:             " << totalMs << " ms\n";
        std::cout << "Warp-coop rules:        " << nCoop
                  << "  (threshold maxfan > " << COOP_FAN << ")\n";

        std::cout << "\n=== Per-rule statistics (" << nRules << " rules) ===\n";
        std::cout << "Average: " << avgMs << " ms\n";
        std::cout << "Median:  " << medianMs << " ms\n";
        std::cout << "Std Dev: " << stdDev << " ms\n";
        std::cout << "Min:     " << minMs << " ms\n";
        std::cout << "Max:     " << maxMs << " ms\n";
        std::cout << "P95:     " << p95Ms << " ms\n";

        /* per-body-size statistics */
        std::cout << "\n=== Statistics by body size ===\n";
        for (auto& [bodySize, times] : timesByBodySize) {
            std::sort(times.begin(), times.end());
            size_t n = times.size();
            double sum = 0.0;
            for (auto t : times) sum += t;
            double avg = sum / (double)n;
            double med = (n % 2 == 1) ? times[n / 2]
                                       : (times[n / 2 - 1] + times[n / 2]) / 2.0;
            size_t p95i = (size_t)(std::ceil(0.95 * n)) - 1;
            double p95 = times[std::min(p95i, n - 1)];

            std::cout << "Body size " << bodySize
                      << " (" << n << " rules): "
                      << "avg=" << avg << " ms, "
                      << "median=" << med << " ms, "
                      << "min=" << times.front() << " ms, "
                      << "max=" << times.back() << " ms, "
                      << "P95=" << p95 << " ms\n";
        }

        /* cleanup */
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        cudaFree(d_res);
        free_dev(dIdx);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
