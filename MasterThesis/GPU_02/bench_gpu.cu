/*  bench_gpu.cu — GPU_02: sparse-matrix support counting (TensorLog / RLvLR)
 *
 *  For a chain rule  q1(x,z1) ∧ q2(z1,z2) ∧ ... ∧ qn(zn-1,y) ⇒ p(x,y)
 *
 *    support = nnz( M_q1 × M_q2 × ... × M_qn  .*  M_head )
 *
 *  Predicates are stored as CSR adjacency matrices (standard sparse matrix
 *  format per TensorLog/RLvLR).  A fused GPU kernel evaluates the matrix
 *  product · head-intersection without materializing intermediate M_body.
 *  Filter atoms (non-chain body atoms sharing hS→hO variables) are checked
 *  per head triple via binary search in the filter's CSR.
 *
 *  Compile:
 *    nvcc -O3 -std=c++17 bench_gpu.cu FinalRule.cpp RdfIndexes.cpp \
 *         RuleParser.cpp $(pkg-config --cflags --libs raptor2) -o bench_gpu
 *
 *  Run:
 *    ./bench_gpu <data.ttl> <rules.txt>
 */

#include "gpu_sparse.cuh"
#include "FinalRule.hpp"
#include "RuleParser.hpp"

#include <cmath>
#include <climits>
#include <map>
#include <set>
#include <numeric>
#include <algorithm>
#include <iostream>

/* ═══════════════════════════════════════════════
 *  Convert FinalRule to GPU-friendly format
 * ═══════════════════════════════════════════════ */

struct GpuBodyAtom {
    int pred, sVar, oVar;
    bool sConst, oConst;       /* whether subject/object is a constant */
    int sConstVal, oConstVal;  /* actual entity ID if constant */
};

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
    for (auto& b : rule.body) { track(b.subject); track(b.object); }
    int nextSlot = maxId + 1;

    gr.hS = rule.head.subject.isVariable() ? rule.head.subject.value : nextSlot++;
    gr.hO = rule.head.object.isVariable()  ? rule.head.object.value  : nextSlot++;

    for (auto& atom : rule.body) {
        GpuBodyAtom ba;
        ba.pred = atom.predicate;
        ba.sConst = atom.subject.isConstant();
        ba.oConst = atom.object.isConstant();
        ba.sConstVal = ba.sConst ? atom.subject.value : -1;
        ba.oConstVal = ba.oConst ? atom.object.value  : -1;
        ba.sVar = ba.sConst ? nextSlot++ : atom.subject.value;
        ba.oVar = ba.oConst ? nextSlot++ : atom.object.value;
        gr.body.push_back(ba);
    }
    gr.numVars = nextSlot;
    return gr;
}

/* ═══════════════════════════════════════════════════════
 *  Chain extraction via DFS on the small variable graph.
 *  Finds a path from hS → hO through body atoms.
 *
 *  Each chain step records:
 *    pred      — predicate ID
 *    transpose — true = use M^T (POS), false = use M (SPO)
 *
 *  Atoms NOT on the chain path are returned as "filters".
 * ═══════════════════════════════════════════════════════ */

struct ChainStep { int pred; bool transpose; };

/* recursive DFS on the variable graph (≤ 8 atoms → trivial) */
static bool find_chain_dfs(int current, int target, int nb,
        const std::vector<GpuBodyAtom>& body,
        std::vector<bool>& used,
        std::vector<ChainStep>& chain)
{
    if (current == target) return true;

    for (int i = 0; i < nb; ++i) {
        if (used[i]) continue;
        int next = -1;
        bool trans = false;

        if (body[i].sVar == current)      { next = body[i].oVar; trans = false; }
        else if (body[i].oVar == current) { next = body[i].sVar; trans = true;  }
        else continue;

        used[i] = true;
        chain.push_back({body[i].pred, trans});

        if (find_chain_dfs(next, target, nb, body, used, chain))
            return true;

        chain.pop_back();
        used[i] = false;
    }
    return false;
}

struct ChainResult {
    std::vector<ChainStep> chain;    /* SpGEMM sequence              */
    std::vector<ChainStep> filters;  /* element-wise (s,o) checks    */
    /* extra-check atoms: row/col source + constant values */
    struct Extra {
        int pred;
        bool transpose;   /* use SPO or POS matrix */
        int rowSrc;       /* 0=hS, 1=hO, 2=constant */
        int colSrc;       /* 0=hS, 1=hO, 2=constant, 3=free */
        int rowConst;
        int colConst;
    };
    std::vector<Extra> extras;
    bool ok = false;
};

/*  Build the variable "path" from hS to hO.
 *  Remaining unused atoms are classified as:
 *    - filters  (both endpoints hS, hO)
 *    - extras   (one/both endpoints are constants or free variables) */
static ChainResult extract_chain(const GpuRule& rule) {
    ChainResult cr;
    int nb = (int)rule.body.size();
    std::vector<bool> used(nb, false);

    cr.ok = find_chain_dfs(rule.hS, rule.hO, nb, rule.body, used, cr.chain);
    if (!cr.ok) {
        /* no chain path exists — try treating entire rule as extras
         * (e.g. star patterns where hO isn't reached by any chain) */
        cr.chain.clear();
        cr.ok = true;  /* still process: chain is empty, extras do the work */
    }

    /* collect all chain-path variable IDs */
    std::set<int> chainVars;
    chainVars.insert(rule.hS);
    chainVars.insert(rule.hO);
    /* intermediate chain variables are implicitly on the path */

    /* classify unused atoms */
    for (int i = 0; i < nb; ++i) {
        if (used[i]) continue;
        auto& b = rule.body[i];

        /* resolve what each endpoint maps to */
        auto resolve = [&](int var, bool isConst, int constVal,
                           int& src, int& cval) {
            if (var == rule.hS)       { src = 0; cval = 0; }
            else if (var == rule.hO)  { src = 1; cval = 0; }
            else if (isConst)         { src = 2; cval = constVal; }
            else                      { src = 3; cval = 0; } /* free variable */
        };

        int rSrc, rConst, cSrc, cConst;
        resolve(b.sVar, b.sConst, b.sConstVal, rSrc, rConst);
        resolve(b.oVar, b.oConst, b.oConstVal, cSrc, cConst);

        /* if both are hS/hO → classic filter */
        if (rSrc == 0 && cSrc == 1) {
            cr.filters.push_back({b.pred, false});
        } else if (rSrc == 1 && cSrc == 0) {
            cr.filters.push_back({b.pred, true});
        } else {
            /* extra check — use SPO matrix (subject→object) */
            cr.extras.push_back({b.pred, false, rSrc, cSrc, rConst, cConst});
        }
    }
    return cr;
}

/* ═══════════════════════════════════════════════
 *  Printing helpers (same format as CPU_01)
 * ═══════════════════════════════════════════════ */

static std::string termToString(const Term& t, const RdfIndexes& indexes) {
    if (t.isVariable()) return "?" + std::to_string(t.value);
    return indexes.mapper.getValue(t.value);
}

static void printRule(const FinalRule& rule, const RdfIndexes& indexes) {
    for (std::size_t i = 0; i < rule.body.size(); ++i) {
        const Atom& a = rule.body[i];
        std::cout << "( "
                  << termToString(a.subject, indexes) << " "
                  << indexes.mapper.getValue(a.predicate) << " "
                  << termToString(a.object, indexes) << " )";
        if (i + 1 < rule.body.size()) std::cout << " ^ ";
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
    std::string ttlFile   = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";
    if (argc > 1) ttlFile   = argv[1];
    if (argc > 2) rulesFile = argv[2];

    try {
        auto T0 = Clock::now();

        /* ── Load RDF ── */
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

        /* ── Build per-predicate CSR matrices (SPO + POS) ── */
        auto csrIdx = build_all_csr(indexes, nEntities);
        auto T2 = Clock::now();
        printf("CSR matrices built & uploaded: %zu predicates  (%.0f ms)\n",
               csrIdx.size(), ms_between(T1, T2));

        /* ── Parse rules ── */
        RuleParser parser(indexes, ttlFile);
        std::vector<FinalRule> rules = parser.parseRuleFile(rulesFile);
        auto T4 = Clock::now();
        int nRules = (int)rules.size();
        std::cout << "\n=== Rules loaded ===\n";
        std::cout << "Rule count: " << nRules << "\n\n";

        /* ── Convert to GPU format ── */
        std::vector<GpuRule> gpuRules(nRules);
        for (int i = 0; i < nRules; ++i)
            gpuRules[i] = convert_rule(rules[i]);

        /* ── Process each rule ── */
        auto supportStart = Clock::now();
        std::vector<double> ruleTimesMs(nRules, 0.0);
        int nChainOk = 0, nSkipped = 0;

        /* cache compact head triples per head predicate */
        std::unordered_map<int, CompactHead> compactHeads;

        for (int ri = 0; ri < nRules; ++ri) {
            auto ruleStart = Clock::now();

            FinalRule& rule = rules[ri];
            GpuRule& gr = gpuRules[ri];
            int nb = (int)gr.body.size();

            /* headSize / headSupport (same as CPU_01) */
            const PredIndex* pi = indexes.getPred(rule.head.predicate);
            int headSize = pi ? pi->totalPairs() : 0;
            int headSupport = 0;
            if (pi) {
                bool sB = rule.head.subject.isConstant();
                bool oB = rule.head.object.isConstant();
                if (!sB && !oB)       headSupport = pi->totalPairs();
                else if (sB && !oB)  { int c = 0; pi->spoRange(rule.head.subject.value, c); headSupport = c; }
                else if (!sB && oB)  { int c = 0; pi->posRange(rule.head.object.value, c);  headSupport = c; }
                else                   headSupport = pi->hasTriple(rule.head.subject.value,
                                                                    rule.head.object.value) ? 1 : 0;
            }

            /* quick skip checks */
            if (nb < 1 || headSize == 0) {
                rule.setMeasures(0, headSize, headSupport);
                auto ruleEnd = Clock::now();
                ruleTimesMs[ri] = ms_between(ruleStart, ruleEnd);
                continue;
            }

            /* head predicate must have a CSR */
            auto headIt = csrIdx.find(rule.head.predicate);
            if (headIt == csrIdx.end() || headIt->second.spo.nnz == 0) {
                rule.setMeasures(0, headSize, headSupport);
                auto ruleEnd = Clock::now();
                ruleTimesMs[ri] = ms_between(ruleStart, ruleEnd);
                continue;
            }

            /* extract chain from hS to hO */
            auto cr = extract_chain(gr);
            if (!cr.ok) {
                fprintf(stderr, "Warning: rule %d not decomposable, skipping\n", ri + 1);
                rule.setMeasures(0, headSize, headSupport);
                ++nSkipped;
                auto ruleEnd = Clock::now();
                ruleTimesMs[ri] = ms_between(ruleStart, ruleEnd);
                continue;
            }

            /* check all predicates exist in CSR index */
            bool missing = false;
            for (auto& step : cr.chain) {
                if (csrIdx.find(step.pred) == csrIdx.end()) { missing = true; break; }
            }
            for (auto& step : cr.filters) {
                if (csrIdx.find(step.pred) == csrIdx.end()) { missing = true; break; }
            }
            for (auto& ex : cr.extras) {
                if (csrIdx.find(ex.pred) == csrIdx.end()) { missing = true; break; }
            }
            if (missing) {
                rule.setMeasures(0, headSize, headSupport);
                ++nSkipped;
                auto ruleEnd = Clock::now();
                ruleTimesMs[ri] = ms_between(ruleStart, ruleEnd);
                continue;
            }

            ++nChainOk;

            /* ─── Fused chain-support (TensorLog/RLvLR approach) ───
             *  support = nnz(M_q1 × M_q2 × ... × M_qn .* M_head)
             *  computed without materializing the product matrix. */

            /* build or reuse compact head triples */
            int headPred = rule.head.predicate;
            if (compactHeads.find(headPred) == compactHeads.end())
                compactHeads[headPred] = build_compact_head(headIt->second.spo);
            const CompactHead& cHead = compactHeads[headPred];

            /* collect chain CSR pointers */
            std::vector<const DevCSR*> chainMats;
            for (auto& step : cr.chain)
                chainMats.push_back(step.transpose
                    ? &csrIdx[step.pred].pos : &csrIdx[step.pred].spo);

            /* collect filter CSR pointers */
            std::vector<const DevCSR*> filterMats;
            for (auto& filt : cr.filters)
                filterMats.push_back(filt.transpose
                    ? &csrIdx[filt.pred].pos : &csrIdx[filt.pred].spo);

            /* build extra-check descriptors */
            std::vector<DevExtraCheck> extraChecks;
            for (auto& ex : cr.extras) {
                const DevCSR& mat = csrIdx[ex.pred].spo;
                extraChecks.push_back({
                    mat.d_rowPtr, mat.d_colInd,
                    ex.rowSrc, ex.colSrc,
                    ex.rowConst, ex.colConst
                });
            }

            /* fused GPU kernel */
            /* injective mapping: skip head triples where two distinct
               variables would be mapped to the same entity */
            int ssl = (rule.head.subject.isVariable() &&
                       rule.head.object.isVariable() &&
                       rule.head.subject.value != rule.head.object.value) ? 1 : 0;
            unsigned long long sup = fused_chain_support(
                cHead, chainMats, filterMats, extraChecks, ssl);
            rule.setMeasures((int)sup, headSize, headSupport);

            auto ruleEnd = Clock::now();
            ruleTimesMs[ri] = ms_between(ruleStart, ruleEnd);
        }

        auto supportEnd = Clock::now();
        auto totalEnd   = Clock::now();

        /* ── Print per-rule results ── */
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
        auto indexMs  = ms_between(T0, T1);
        auto csrMs    = ms_between(T1, T2);
        auto rulesMs  = ms_between(T2, T4);
        auto supMs    = ms_between(supportStart, supportEnd);
        auto totalMs  = ms_between(T0, totalEnd);
        double avgMs  = nRules > 0 ? sumRuleMs / nRules : 0.0;

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
        std::cout << "Chain rules processed:  " << nChainOk << "\n";
        std::cout << "Skipped (non-chain):    " << nSkipped << "\n";

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
        for (auto& [_, ch] : compactHeads) free_compact_head(ch);
        free_all_csr(csrIdx);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
