#include <iostream>
#include "corpus.h"

using namespace std;
using namespace treebank;

template <typename T>
ostream& operator<< (ostream& os, const vector<T>& vec) {
	for (const T& x: vec)
		cerr << x << ", ";
	return os;
}

int main(){
	cout << "hello world!" << endl;
	Corpus corpus;

	corpus.load_corpus(corpus.train_sentences, "../unlabeled_experiment/arc-hybrid/train.txt");
	corpus.load_corpus_dev(corpus.dev_sentences, "../unlabeled_experiment/arc-hybrid/dev.txt");
	corpus.load_corpus_dev(corpus.test_sentences, "../unlabeled_experiment//arc-hybrid/test.txt");
	cout << corpus.form << endl << corpus.pos << endl 
		<< corpus.deprel << endl << corpus.transition << endl
		<< corpus.chars << endl;
		
	Vocab action{vector<string>{}, false};
	action.get_or_add("Shift");
	for (unsigned i = 0; i < corpus.deprel.stoi.size(); ++i) {
		action.get_or_add(string{"LeftReduce-"} + corpus.deprel.itos.at(i));
		action.get_or_add(string{"RightReduce-"} + corpus.deprel.itos.at(i));
	}
	
	cout << "action assembled:\n" << action << endl;
	
	
	cout << "There are " << corpus.train_sentences.size() << " train sentences in the corpus" << endl;
	cout << "There are " << corpus.dev_sentences.size() << " dev sentences in the corpus" << endl;
	cout << "There are " << corpus.test_sentences.size() << " test sentences in the corpus" << endl;
	cout << corpus.train_sentences[0].form << endl;
	

	return 0;
}

