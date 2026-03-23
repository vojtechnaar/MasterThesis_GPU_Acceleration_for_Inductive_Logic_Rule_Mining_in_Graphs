#include "RdfIndexes.hpp"

#include <iostream>

#include <raptor2.h>

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

int IdMapper::getIdIfExists(const std::string& value) const {
    auto it = strToId_.find(value);
    return (it == strToId_.end()) ? -1 : it->second;
}

const std::string& IdMapper::getValue(int id) const {
    return idToStr_.at(static_cast<std::size_t>(id));
}

std::size_t IdMapper::size() const {
    return idToStr_.size();
}

void RdfIndexes::addTriple(const std::string& s, const std::string& p, const std::string& o) {
    const int sid = mapper.getOrCreateId(s);
    const int pid = mapper.getOrCreateId(p);
    const int oid = mapper.getOrCreateId(o);

    triples.push_back({sid, pid, oid});
    pso[pid][sid].insert(oid);
    pos[pid][oid].insert(sid);
}

bool RdfIndexes::hasTriple(int s, int p, int o) const {
    auto pit = pso.find(p);
    if (pit == pso.end()) return false;

    auto sit = pit->second.find(s);
    if (sit == pit->second.end()) return false;

    return sit->second.find(o) != sit->second.end();
}

const RdfIndexes::IntSet* RdfIndexes::getObjects(int predicate, int subject) const {
    auto pit = pso.find(predicate);
    if (pit == pso.end()) return nullptr;

    auto sit = pit->second.find(subject);
    if (sit == pit->second.end()) return nullptr;

    return &sit->second;
}

const RdfIndexes::IntSet* RdfIndexes::getSubjects(int predicate, int object) const {
    auto pit = pos.find(predicate);
    if (pit == pos.end()) return nullptr;

    auto oit = pit->second.find(object);
    if (oit == pit->second.end()) return nullptr;

    return &oit->second;
}

void RdfIndexes::printStats() const {
    std::cout << "Triples: " << triples.size() << "\n";
    std::cout << "Unique RDF terms: " << mapper.size() << "\n";
    std::cout << "Predicates in PSO: " << pso.size() << "\n";
    std::cout << "Predicates in POS: " << pos.size() << "\n";
}

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

    // base URI = file URI is fine for local Turtle files
    const int rc = raptor_parser_parse_file(parser, fileUri, fileUri);

    raptor_free_uri(fileUri);
    raptor_free_parser(parser);
    raptor_free_world(world);

    if (rc != 0) {
        std::cerr << "Raptor failed to parse file: " << filePath << "\n";
        return false;
    }

    return true;
}