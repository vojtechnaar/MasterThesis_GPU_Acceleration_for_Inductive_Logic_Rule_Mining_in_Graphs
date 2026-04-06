#include "RuleParser.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

RuleParser::RuleParser(RdfIndexes& indexes)
    : indexes_(indexes),
      prefixes_({
          {"biolink", "https://w3id.org/biolink/vocab/"},
          {"rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
          {"rdfs", "http://www.w3.org/2000/01/rdf-schema#"},
          {"owl", "http://www.w3.org/2002/07/owl#"},
          {"obo", "http://purl.obolibrary.org/obo/"},
          {"wd", "https://www.wikidata.org/wiki/"},
          {"dc1", "http://purl.org/dc/terms/"},
          {"oio", "http://www.geneontology.org/formats/oboInOwl#"}
      }) {}

std::string RuleParser::trim(const std::string& s) const {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }

    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(start, end - start);
}

std::vector<std::string> RuleParser::splitBodyAtoms(const std::string& bodyText) const {
    std::vector<std::string> atoms;
    std::string current;
    int depth = 0;

    for (char ch : bodyText) {
        if (ch == '(') {
            ++depth;
            current.push_back(ch);
        } else if (ch == ')') {
            --depth;
            current.push_back(ch);
        } else if (ch == '^' && depth == 0) {
            std::string t = trim(current);
            if (!t.empty()) {
                atoms.push_back(t);
            }
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    std::string t = trim(current);
    if (!t.empty()) {
        atoms.push_back(t);
    }

    return atoms;
}

std::string RuleParser::expandPrefixedName(const std::string& token) const {
    if (token.empty()) {
        return token;
    }

    if (token.front() == '<' && token.back() == '>') {
        return token.substr(1, token.size() - 2);
    }

    if (token.front() == '?') {
        return token;
    }

    if (token.front() == '"') {
        return token;
    }

    std::size_t pos = token.find(':');
    if (pos == std::string::npos) {
        return token;
    }

    std::string prefix = token.substr(0, pos);
    std::string local = token.substr(pos + 1);

    auto it = prefixes_.find(prefix);
    if (it != prefixes_.end()) {
        return it->second + local;
    }

    return token;
}

Term RuleParser::parseTerm(
    const std::string& token,
    std::unordered_map<std::string, int>& varMap,
    int& nextVarId
) const {
    std::string clean = trim(token);
    if (clean.empty()) {
        throw std::runtime_error("Empty term token");
    }

    if (clean.front() == '?') {
        auto it = varMap.find(clean);
        if (it != varMap.end()) {
            return Term(TermType::Variable, it->second);
        }

        int newId = nextVarId++;
        varMap[clean] = newId;
        return Term(TermType::Variable, newId);
    }

    std::string expanded = expandPrefixedName(clean);
    int id = indexes_.mapper.getOrCreateId(expanded);
    return Term(TermType::Constant, id);
}

FinalRule RuleParser::parseRuleLine(const std::string& line) const {
    std::string clean = trim(line);
    if (clean.empty()) {
        throw std::runtime_error("Empty rule line");
    }

    std::size_t barPos = clean.find('|');
    if (barPos != std::string::npos) {
        clean = trim(clean.substr(0, barPos));
    }

    std::size_t arrowPos = clean.find("=>");
    if (arrowPos == std::string::npos) {
        throw std::runtime_error("Rule line missing => : " + line);
    }

    std::string bodyText = trim(clean.substr(0, arrowPos));
    std::string headText = trim(clean.substr(arrowPos + 2));

    std::unordered_map<std::string, int> varMap;
    int nextVarId = 0;

    std::vector<Atom> body;
    std::vector<std::string> bodyAtoms = splitBodyAtoms(bodyText);

    for (const auto& atomStr : bodyAtoms) {
        std::string s = trim(atomStr);
        if (s.empty() || s.front() != '(' || s.back() != ')') {
            throw std::runtime_error("Invalid body atom: " + atomStr);
        }

        s = trim(s.substr(1, s.size() - 2));

        std::istringstream iss(s);
        std::string subjTok, predTok, objTok;
        iss >> subjTok >> predTok >> objTok;

        if (subjTok.empty() || predTok.empty() || objTok.empty()) {
            throw std::runtime_error("Invalid body atom: " + atomStr);
        }

        Term subj = parseTerm(subjTok, varMap, nextVarId);
        Term obj = parseTerm(objTok, varMap, nextVarId);
        std::string expandedPred = expandPrefixedName(predTok);
        int predId = indexes_.mapper.getOrCreateId(expandedPred);

        body.emplace_back(predId, subj, obj);
    }

    std::string hs = trim(headText);
    if (hs.empty() || hs.front() != '(' || hs.back() != ')') {
        throw std::runtime_error("Invalid head atom: " + headText);
    }

    hs = trim(hs.substr(1, hs.size() - 2));

    std::istringstream hss(hs);
    std::string headSubjTok, headPredTok, headObjTok;
    hss >> headSubjTok >> headPredTok >> headObjTok;

    if (headSubjTok.empty() || headPredTok.empty() || headObjTok.empty()) {
        throw std::runtime_error("Invalid head atom: " + headText);
    }

    Term headSubj = parseTerm(headSubjTok, varMap, nextVarId);
    Term headObj = parseTerm(headObjTok, varMap, nextVarId);
    std::string expandedHeadPred = expandPrefixedName(headPredTok);
    int headPredId = indexes_.mapper.getOrCreateId(expandedHeadPred);

    Atom head(headPredId, headSubj, headObj);
    return FinalRule(head, body);
}

std::vector<FinalRule> RuleParser::parseRuleFile(const std::string& filePath) const {
    std::ifstream in(filePath);
    if (!in) {
        throw std::runtime_error("Cannot open rule file: " + filePath);
    }

    std::vector<FinalRule> rules;
    std::string line;

    while (std::getline(in, line)) {
        std::string clean = trim(line);
        if (clean.empty()) {
            continue;
        }
        rules.push_back(parseRuleLine(clean));
    }

    return rules;
}