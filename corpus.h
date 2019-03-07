# pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <cassert>

namespace treebank {
using namespace std;

struct Sentence {
	// leave it as a POD data type
	vector<string> form;
	vector<unsigned> formi;
	vector<string> pos;
	vector<unsigned> posi;
	vector<int> head;
	vector<string> deprel;
	vector<unsigned> depreli;
	vector<string> transitions;
	vector<unsigned> transitionsi;

	void clear() {
		form.clear(); formi.clear();
		pos.clear(); posi.clear();
		head.clear();
		deprel.clear(); depreli.clear();
		transitions.clear(); transitionsi.clear();
	}
};

struct Vocab {

public:
	// some literals for vocabs to use if they need
	// inclass initialization is only valid for literal type
	static constexpr const char* UNK {"<unk>"};
	static constexpr const char* PAD {"<pad>"};
	static constexpr const char* START {"<s>"};
	static constexpr const char* END {"</s>"};

public:
	bool use_freq = true;
	bool frozen = false;
	vector<string> specials;
	map<string, unsigned> stoi;
	map<unsigned, string> itos;
	map<string, unsigned> stofreq;
	map<unsigned, unsigned> itofreq;

public:
	explicit Vocab(vector<string> t_specials, bool t_use_freq):
		use_freq(t_use_freq), specials(t_specials) {
		if (specials.size()) {
			if (use_freq) {
				for (const string& s : specials) {
					unsigned i = stoi.size();
					stoi[s] = i; itos[i] = s;
					stofreq[s] = 1; itofreq[i] = 1;
				}
			} else {
				for (const string& s : specials) {
					unsigned i = stoi.size();
					stoi[s] = i; itos[i] = s;
				}
			}
		}
	}

	inline unsigned get_or_add(const string& word) {
		auto iter = stoi.find(word);
		if (iter == stoi.end()) {
			if (!frozen) {
				unsigned id = stoi.size();
				itos[id] = word; stoi[word] = id;
				stofreq[word] = 1; itofreq[id] = 1;
				return id;
			} else
				return stoi[UNK];
		} else {
			unsigned id = iter->second;
			if (!frozen) {
				++(stofreq[word]); ++(itofreq[id]);
			}
			return id;
		}
	}

	// to support operator <<
	friend ostream& operator << (ostream& os, const Vocab& v) {
		assert(v.stoi.size() == v.itos.size());
		cerr << "vocab's size is " << v.stoi.size() << endl;

		bool large = false; unsigned i = 0;
		for (const auto& entry : v.stoi) {
			if (i > 50) {
				large = true;
				break;
			}
			cerr << entry.first << ": " << entry.second << endl; ++i;
		}

		if (large) {
			cerr << "..." << endl;
		}
		return os;
	}

};

struct Corpus {

public:
	bool USE_SPELLING = false;
	Vocab form;
	Vocab pos;
	Vocab deprel;
	Vocab transition;
	Vocab chars;
	vector<Sentence> train_sentences;
	vector<Sentence> dev_sentences;
	vector<Sentence> test_sentences;

public:
	Corpus():
		form(vector<string> {Vocab::UNK}, true),
		pos(vector<string> {Vocab::UNK}, false),
		deprel(vector<string> {}, false),
		transition(vector<string> {}, false),
		chars(vector<string> {}, false) {
	}
	
	Corpus(Vocab t_form, Vocab t_pos, Vocab t_deprel, Vocab t_transition, Vocab t_chars) :
		form(t_form), pos(t_pos), deprel(t_deprel), transition(t_transition), chars(t_chars) {
	}

	// 给定首字节， 判断这个 utf8 字符的长度
	unsigned UTF8Len(unsigned char x) {
		if (x < 0x80) return 1;
		else if ((x >> 5) == 0x06) return 2;
		else if ((x >> 4) == 0x0e) return 3;
		else if ((x >> 3) == 0x1e) return 4;
		else if ((x >> 2) == 0x3e) return 5;
		else if ((x >> 1) == 0x7e) return 6;
		else return 0;
	}

	void freeze_vocabs() {
		form.frozen = true;
		pos.frozen = true;
		deprel.frozen = true;
		transition.frozen = true;
		chars.frozen = true;
	}

	// load corpus of my specific format, this function can be re-written to adapt to
	// different format
	void load_corpus(vector<Sentence>& sents, const string& file) {
		ifstream corpus_file(file);
		string line;
		Sentence s;
		while (getline(corpus_file, line)) {
			if (line.empty()) {
				sents.push_back(s);
				s.clear();
			} else {
				istringstream iss(line);
				string field;
				iss >> field;
				if (field == "form:") {
					string word; unsigned id;
					while (iss >> word) {
						if (word.empty()) continue;
						id = form.get_or_add(word);
						s.form.push_back(word); s.formi.push_back(id);
					}
				} else if (field == "pos:") {
					string word; unsigned id;
					while (iss >> word) {
						if (word.empty()) continue;
						id = pos.get_or_add(word);
						s.pos.push_back(word); s.posi.push_back(id);
					}
				} else if (field == "deprel:") {
					string word; unsigned id;
					while (iss >> word) {
						if (word.empty()) continue;
						id = deprel.get_or_add(word);
						s.deprel.push_back(word); s.depreli.push_back(id);
					}
				} else if (field == "transitions:") {
					string word; unsigned id;
					while (iss >> word) {
						if (word.empty()) continue;
						id = transition.get_or_add(word);
						s.transitions.push_back(word); s.transitionsi.push_back(id);
					}
				} else {
					unsigned id;
					while (iss >> id) {
						s.head.push_back(id);
					}
				}
			}
		}
		corpus_file.close();
		cerr << "Read in " << sents.size() << " sentences from " << file << endl;
	}

	void load_corpus_dev(vector<Sentence>& sents, const string& file) {
		freeze_vocabs();
		load_corpus(sents, file);
	}

};

} //namespace

