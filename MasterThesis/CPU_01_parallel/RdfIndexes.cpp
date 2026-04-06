#include "RdfIndexes.hpp"

#include <iostream>
#include <algorithm>

#include <raptor2.h>

// ---- IdMapper ----

int IdMapper::getOrCreateId(const std::string& value) {
    auto it = strToId_.find(value);
    if (it != strToId_.end()) {
        return it->second;
    }
    int id = static_cast<int>(idToStr_.size());
    strToId_[value] = id;
    idToStr_.push_back(value);
    return id;
}

const std::string& IdMapper::getValue(int id) const {
    return idToStr_.at(static_cast<std::size_t>(id));
}

std::size_t IdMapper::size() const {
    return idToStr_.size();
}

// ---- PredIndex ----

static void buildCSR(
    const std::vector<std::pair<int,int>>& sorted,
    std::vector<int>& keys,
    std::vector<int>& offsets,
    std::vector<int>& vals
) {
    keys.clear();
    offsets.clear();
    vals.clear();
    if (sorted.empty()) {
        offsets.push_back(0);
        return;
    }
    vals.reserve(sorted.size());
    int i = 0;
    int n = static_cast<int>(sorted.size());
    while (i < n) {
        int key = sorted[i].first;
        keys.push_back(key);
        offsets.push_back(static_cast<int>(vals.size()));
        while (i < n && sorted[i].first == key) {
            vals.push_back(sorted[i].second);
            ++i;
        }
    }
    offsets.push_back(static_cast<int>(vals.size()));
}

void PredIndex::build() {
    // Sort and deduplicate spo
    std::sort(spo.begin(), spo.end());
    spo.erase(std::unique(spo.begin(), spo.end()), spo.end());

    // Build CSR for subject->objects
    buildCSR(spo, spoKeys, spoOffsets, spoVals);

    // Build reversed pairs (object, subject), sorted
    std::vector<std::pair<int,int>> posVec;
    posVec.reserve(spo.size());
    for (const auto& p : spo) {
        posVec.emplace_back(p.second, p.first);
    }
    std::sort(posVec.begin(), posVec.end());
    posVec.erase(std::unique(posVec.begin(), posVec.end()), posVec.end());

    // Build CSR for object->subjects
    buildCSR(posVec, posKeys, posOffsets, posVals);
}

const int* PredIndex::spoRange(int subject, int& count) const {
    auto it = std::lower_bound(spoKeys.begin(), spoKeys.end(), subject);
    if (it == spoKeys.end() || *it != subject) {
        count = 0;
        return nullptr;
    }
    int idx = static_cast<int>(it - spoKeys.begin());
    count = spoOffsets[idx + 1] - spoOffsets[idx];
    return spoVals.data() + spoOffsets[idx];
}

const int* PredIndex::posRange(int object, int& count) const {
    auto it = std::lower_bound(posKeys.begin(), posKeys.end(), object);
    if (it == posKeys.end() || *it != object) {
        count = 0;
        return nullptr;
    }
    int idx = static_cast<int>(it - posKeys.begin());
    count = posOffsets[idx + 1] - posOffsets[idx];
    return posVals.data() + posOffsets[idx];
}

bool PredIndex::hasTriple(int s, int o) const {
    int cnt = 0;
    const int* objs = spoRange(s, cnt);
    if (!objs) return false;
    auto it = std::lower_bound(objs, objs + cnt, o);
    return it != objs + cnt && *it == o;
}

// ---- RdfIndexes ----

void RdfIndexes::addTriple(const std::string& s, const std::string& p, const std::string& o) {
    const int sid = mapper.getOrCreateId(s);
    const int pid = mapper.getOrCreateId(p);
    const int oid = mapper.getOrCreateId(o);

    predIndexes[pid].spo.emplace_back(sid, oid);
}

void RdfIndexes::buildIndexes() {
    for (auto& [pred, pi] : predIndexes) {
        pi.build();
    }
}

const PredIndex* RdfIndexes::getPred(int p) const {
    auto it = predIndexes.find(p);
    return it == predIndexes.end() ? nullptr : &it->second;
}

void RdfIndexes::printStats() const {
    int totalPairs = 0;
    for (const auto& [pred, pi] : predIndexes) {
        totalPairs += pi.totalPairs();
    }
    std::cout << "Triples (deduped): " << totalPairs << "\n";
    std::cout << "Unique RDF terms: " << mapper.size() << "\n";
    std::cout << "Predicates: " << predIndexes.size() << "\n";
}

// ---- Raptor parsing (unchanged) ----

static std::string termToString(const raptor_term* term) {
    if (!term) {
        return "";
    }

    switch (term->type) {
        case RAPTOR_TERM_TYPE_URI:
            return std::string(reinterpret_cast<const char*>(raptor_uri_as_string(term->value.uri)));

        case RAPTOR_TERM_TYPE_LITERAL: {
            std::string value =
                std::string(reinterpret_cast<const char*>(term->value.literal.string));

            if (term->value.literal.datatype) {
                value += "^^";
                value += reinterpret_cast<const char*>(
                    raptor_uri_as_string(term->value.literal.datatype));
            } else if (term->value.literal.language) {
                value += "@";
                value += reinterpret_cast<const char*>(term->value.literal.language);
            }
            return value;
        }

        case RAPTOR_TERM_TYPE_BLANK:
            return "_:" + std::string(reinterpret_cast<const char*>(term->value.blank.string));

        default:
            return "";
    }
}

static void statementHandler(void* user_data, raptor_statement* statement) {
    auto* indexes = static_cast<RdfIndexes*>(user_data);
    if (!indexes || !statement) {
        return;
    }

    const std::string s = termToString(statement->subject);
    const std::string p = termToString(statement->predicate);
    const std::string o = termToString(statement->object);

    if (!s.empty() && !p.empty() && !o.empty()) {
        indexes->addTriple(s, p, o);
    }
}

bool RdfIndexes::parseTurtleFile(const std::string& filePath) {
    raptor_world* world = raptor_new_world();
    if (!world) {
        std::cerr << "Failed to create Raptor world\n";
        return false;
    }

    raptor_world_open(world);

    raptor_parser* parser = raptor_new_parser(world, "turtle");
    if (!parser) {
        std::cerr << "Failed to create Turtle parser\n";
        raptor_free_world(world);
        return false;
    }

    raptor_parser_set_statement_handler(parser, this, statementHandler);

    unsigned char* fileUriString =
        raptor_uri_filename_to_uri_string(filePath.c_str());
    if (!fileUriString) {
        std::cerr << "Failed to convert file path to file:// URI\n";
        raptor_free_parser(parser);
        raptor_free_world(world);
        return false;
    }

    raptor_uri* fileUri = raptor_new_uri(world, fileUriString);
    raptor_free_memory(fileUriString);

    if (!fileUri) {
        std::cerr << "Failed to create file URI\n";
        raptor_free_parser(parser);
        raptor_free_world(world);
        return false;
    }

    const int rc = raptor_parser_parse_file(parser, fileUri, fileUri);

    raptor_free_uri(fileUri);
    raptor_free_parser(parser);
    raptor_free_world(world);

    if (rc != 0) {
        std::cerr << "Raptor failed to parse file: " << filePath << "\n";
        return false;
    }

    // Build flat sorted indexes after parsing
    buildIndexes();

    return true;
}