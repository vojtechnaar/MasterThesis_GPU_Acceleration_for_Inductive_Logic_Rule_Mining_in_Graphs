/*  bench_gpu.cu — GPU_03: CSR-indexed DFS with __constant__ memory
 *
 *  Combines:
 *    - CSR row pointers for O(1) entity→neighbors lookup (from GPU_02)
 *    - __constant__ memory atom descriptors (from GPU_01)
 *    - Template-specialized DFS with compile-time unrolling (from GPU_01)
 *    - Warp-cooperative kernel with __any_sync early-exit (from GPU_01)
 *    - __ldg texture cache for all global memory reads (from GPU_01)
 *
 *  Per-atom lookup: O(1) via rowPtr vs O(log |P|) binary search in GPU_01.
 *  Existence checks: O(log d) where d = entity degree, vs O(log |P|) in GPU_01.
 *
 *  Compile:
 *    nvcc -O3 -std=c++17 bench_gpu.cu FinalRule.cpp RdfIndexes.cpp \
 *         RuleParser.cpp $(pkg-config --cflags --libs raptor2) -o bench_gpu
 *
 *  Run:
 *    ./bench_gpu <data.ttl> <rules.txt>
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

struct GpuBodyAtom { int pred, sVar, oVar; };

struct GpuRule {
    int headPred, hS, hO;
    std::vector<GpuBodyAtom> body;
    int numVars;
};

static GpuRule convert_rule(const FinalRule& rule) {
    GpuRule gr;
    gr.headPred = rule.head.predicate;

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

    gr.hS = rule.head.subject.isVariable() ? rule.head.subject.value : nextSlot++;
    gr.hO = rule.head.object.isVariable()  ? rule.head.object.value  : nextSlot++;

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
 *  Uses CSR maxfan instead of sorted-array maxfan.
 * ═══════════════════════════════════════════════ */

struct Ordered { int pred; int keySlot, valSlot, chkSlot; bool useSPO; };

static std::vector<Ordered> order_body(
        const GpuRule& r,
        const std::unordered_map<int, PredCSR>& csrIdx)
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
            auto it = csrIdx.find(b.pred);
            if (it != csrIdx.end()) {
                if (sh == 2) {
                    c = (int)it->second.spo.nnz;
                } else {
                    bool sB = bound[b.sVar];
                    c = sB ? it->second.spo.maxfan : it->second.pos.maxfan;
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

        int nEntities = (int)indexes.mapper.size();
        printf("Entity count: %d\n", nEntities);

        /* ── Build CSR index (O(1) row access per entity) ── */
        auto csrIdx = build_all_csr(indexes, nEntities);
        auto T2 = Clock::now();
        printf("CSR index built: %zu predicates  (%.0f ms)\n",
               csrIdx.size(), ms_between(T1, T2));

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

        /* cache compact head triples per head predicate */
        std::unordered_map<int, CompactHead> compactHeads;

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

            /* build or reuse compact head triples */
            int headPred = rule.head.predicate;
            if (compactHeads.find(headPred) == compactHeads.end()) {
                if (pi)
                    compactHeads[headPred] = build_compact_head(*pi);
                else
                    compactHeads[headPred] = CompactHead{};
            }
            const CompactHead& cHead = compactHeads[headPred];
            int headN = cHead.nPairs;

            if (nb < 1 || nb > MAX_BODY || headN == 0 || gr.numVars > MAXVAR) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* order body atoms */
            auto ord = order_body(gr, csrIdx);
            if ((int)ord.size() != nb) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* check all body predicates exist in CSR index */
            bool skip = false;
            for (auto& o : ord) {
                auto di = csrIdx.find(o.pred);
                if (di == csrIdx.end()) { skip = true; break; }
                const DevCSR& csr = o.useSPO ? di->second.spo : di->second.pos;
                if (csr.nnz == 0) { skip = true; break; }
            }
            if (skip) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* decide: standard vs warp-cooperative */
            bool useCoop = false;
            if (ord[0].valSlot >= 0) {
                auto& pc = csrIdx[ord[0].pred];
                int mf = ord[0].useSPO ? pc.spo.maxfan : pc.pos.maxfan;
                useCoop = (mf > COOP_FAN);
            }

            /* upload atom descriptors to __constant__ memory */
            AtomGPU atoms[MAX_BODY];
            for (int i = 0; i < nb; ++i) {
                auto& o   = ord[i];
                auto& pc  = csrIdx[o.pred];
                const DevCSR& csr = o.useSPO ? pc.spo : pc.pos;
                atoms[i].rowPtr  = csr.d_rowPtr;
                atoms[i].colInd  = csr.d_colInd;
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
                launch_coop(nb, grd, blk, cHead.d_pairs, headN,
                            gr.hS, gr.hO, ssl, d_res);
                ++nCoop;
            } else {
                int grd = std::min((headN + blk - 1) / blk, 2048);
                launch_std(nb, grd, blk, cHead.d_pairs, headN,
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
        auto csrMs     = ms_between(T1, T2);
        auto rulesMs   = ms_between(T2, T4);
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
        std::cout << "CSR build+upload time:  " << csrMs << " ms\n";
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
        for (auto& [_, ch] : compactHeads) free_compact_head(ch);
        free_all_csr(csrIdx);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
