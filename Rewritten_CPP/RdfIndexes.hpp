#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct Triple {
    int s = -1;
    int p = -1;
    int o = -1;
};

class IdMapper {
public:
    int getOrCreateId(const std::string& value);
    int getIdIfExists(const std::string& value) const;
    const std::string& getValue(int id) const;
    std::size_t size() const;

private:
    std::unordered_map<std::string, int> strToId_;
    std::vector<std::string> idToStr_;
};

class RdfIndexes {
public:
    using IntSet = std::unordered_set<int>;
    using SubjectToObjects = std::unordered_map<int, IntSet>;
    using ObjectToSubjects = std::unordered_map<int, IntSet>;

    // PSO: predicate -> subject -> {objects}
    std::unordered_map<int, SubjectToObjects> pso;

    // POS: predicate -> object -> {subjects}
    std::unordered_map<int, ObjectToSubjects> pos;

    std::vector<Triple> triples;
    IdMapper mapper;

    void addTriple(const std::string& s, const std::string& p, const std::string& o);
    bool parseTurtleFile(const std::string& filePath);

    bool hasTriple(int s, int p, int o) const;
    const IntSet* getObjects(int predicate, int subject) const;
    const IntSet* getSubjects(int predicate, int object) const;

    void printStats() const;
};