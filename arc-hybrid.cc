#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>

#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/model.h>
#include <dynet/io.h>
#include <dynet/lstm.h>
#include <dynet/mp.h>

#include "corpus.h"

namespace po = boost::program_options;
using namespace std;
using namespace dynet;
using namespace dynet::mp;
using namespace treebank;
using namespace boost::posix_time;

void InitCommandLine(int argc, char** argv, po::variables_map& conf) {
	using namespace std;
	// very typical function that uses side-effect, output to output parameter and returns nothing
	po::options_description opts("Configuration options");
	opts.add_options()
	("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
	("dev_data,d", po::value<string>(), "Development corpus")
	("test_data,p", po::value<string>(), "Test corpus")
	("epoches,e", po::value<unsigned>()->default_value(20), "Training epoches")
	("init_learning_rate,l", po::value<double>()->default_value(0.2), "Init learning rate")
	("resume,r", "Whether to resume training")
	("param,m", po::value<string>(), "Load/ save save model from/ to this file")
	("trainer_state,s", po::value<string>(), "Load/ save trainer state from/ to this file")
	("use_pos_tags,P", po::value<bool>(), "make POS tags visible to parser")
	("use_head", po::value<bool>(), "use head word embedding as a feature for a position")
	("use_rl", po::value<bool>(), "use right and left children as features for a position")
	("use_rl_most", po::value<bool>(), "use right- and left-most children as feature for a position")
	("window_size", po::value<unsigned>()->default_value(3), "windows size on the stack")
	("exp_prob,a", po::value<float>()->default_value(0.1), "probability of aggresive exploration")
	("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
	("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
	("pretrained_dim", po::value<unsigned>()->default_value(0), "pretrained embedding size")
	("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
	("hidden1_dim", po::value<unsigned>()->default_value(64), "hidden dimension of layer1")
	("hidden2_dim", po::value<unsigned>()->default_value(0), "hidden dimension of layer2")
	("lstm_dim", po::value<unsigned>()->default_value(60), "LSTM hidden dimension")
	("train,t", "Should training be run?")
	("words,w", po::value<string>(), "Pretrained word embeddings")
	("config", po::value<string>(), "Config file")
	("help,h", "Help");
	po::options_description dcmdline_options;
	dcmdline_options.add(opts);

	// allow unregistered options, pass them through to dynet
	po::store(po::command_line_parser(argc, argv).options(dcmdline_options).allow_unregistered().run(), conf);

	if (conf.count("config")) {
		ifstream ifs {conf["config"].as<string>()};
		if (ifs)
			po::store(po::parse_config_file(ifs, dcmdline_options), conf);
	}
	po::notify(conf);

	if (conf.count("help")) {
		cerr << dcmdline_options << endl;
		exit(1);
	}
	if (conf.count("training_data") == 0) {
		cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
		exit(1);
	}
}

struct Result {
	vector<unsigned> actions;
	Expression loss;
	size_t num_prediction;
	unsigned right;
};

struct Element {
	int id;
	int lc;
	int rc;
};

struct ArcHybridLSTM {
public:
	bool use_pos;
	bool use_pretrained;
	bool use_head;
	bool use_rl;
	bool use_rlmost;
	unsigned window_size;
	unsigned nnvecs;
	unsigned hidden2_dim;
	double p_agg;
	Vocab& form;
	Vocab& transition; //only 3 transition
	Vocab& deprel;
	Vocab& action;
	LSTMBuilder sflstm; // (1, input, lstm_dim, pc)
	LSTMBuilder sblstm;
	LSTMBuilder bflstm; // (1, 2lstm_dim, lstm_dim, pc)
	LSTMBuilder bblstm;
	LookupParameter p_w; // word embeddings
	LookupParameter p_t; // pretrained word embeddings
	LookupParameter p_p; // pos tag embeddings
	Parameter p_p2h1;
	Parameter p_h1b;
	Parameter p_h1h2;
	Parameter p_h2b;
	Parameter p_h2a;
	Parameter p_abias;
	Parameter p_rp2h1;
	Parameter p_rh1b;
	Parameter p_rh1h2;
	Parameter p_rh2b;
	Parameter p_rh2a;
	Parameter p_rabias;
	Parameter p_empty;
	ParameterCollection& pc;  // parameter collection

public:
	explicit ArcHybridLSTM(ParameterCollection& model,
	                       Vocab& t_form, Vocab& t_transition, Vocab& t_deprel, Vocab& t_action,
	                       unsigned input_dim, unsigned pos_dim,
	                       unsigned layers, unsigned lstm_dim, unsigned t_window_size,
	                       unsigned hidden1_dim, unsigned t_hidden2_dim,
	                       unsigned vocab_size, unsigned pos_size,
	                       bool t_use_head, bool t_use_rl, bool t_use_rl_most,
	                       bool t_use_pos, bool t_use_pretrained, double t_p_agg,
	                       const unordered_map<unsigned, vector<float>>& pretrained) :
		use_pos(t_use_pos), use_pretrained(t_use_pretrained), use_head(t_use_head), use_rl(t_use_rl),
		use_rlmost(t_use_rl_most), window_size(t_window_size), hidden2_dim(t_hidden2_dim),
		p_agg(t_p_agg), form(t_form), transition(t_transition),
		deprel(t_deprel), action(t_action), pc(model) {
		// lookup parameters
		p_w = model.add_lookup_parameters(vocab_size, {input_dim});
		if (use_pos)
			p_p = model.add_lookup_parameters(pos_size, {pos_dim});
		unsigned pretrained_dim = 0;
		if (use_pretrained) { // we do not handle specials here, do it outside
			assert(pretrained_dim == pretrained.at(0).size());
			p_t = model.add_lookup_parameters(pretrained.size(), {pretrained_dim});
			for (const auto& it : pretrained)
				p_t.initialize(it.first, it.second);
		}

		// well, well, since I know that LSTMBuilder is move/copy constructable/assignable;
		nnvecs = (use_head ? 1 : 0) + (use_rl || use_rlmost ? 2 : 0);
		unsigned dims = input_dim + (use_pos ? pos_dim : 0) + (use_pretrained ? pretrained_dim : 0);
		sflstm = LSTMBuilder(layers, dims, lstm_dim, model);
		sblstm = LSTMBuilder(layers, dims, lstm_dim, model);
		bflstm = LSTMBuilder(1, 2 * lstm_dim, lstm_dim, model);
		bblstm = LSTMBuilder(1, 2 * lstm_dim, lstm_dim, model);

		// mlp classifier
		unsigned state_dim = 2 * lstm_dim * nnvecs * (t_window_size + 1); // bilstm * nvec * (stack + buffer)
		p_p2h1 = model.add_parameters({hidden1_dim, state_dim});
		p_h1b = model.add_parameters({hidden1_dim});
		p_rp2h1 = model.add_parameters({hidden1_dim, state_dim});
		p_rh1b = model.add_parameters({hidden1_dim});
		if (hidden2_dim > 0) {
			p_h1h2 = model.add_parameters({hidden2_dim, hidden1_dim});
			p_h2b = model.add_parameters({hidden2_dim});
			p_rh1h2 = model.add_parameters({hidden2_dim, hidden1_dim});
			p_rh2b = model.add_parameters({hidden2_dim});
		}
		unsigned hidden_dim = hidden2_dim > 0 ? hidden2_dim : hidden1_dim;
		p_h2a = model.add_parameters({3, hidden_dim});
		p_abias = model.add_parameters({3});
		unsigned action_size = 2 * deprel.stoi.size() + 1; // shift, left(rel), right(rel)
		p_rh2a = model.add_parameters({action_size, hidden_dim});
		p_rabias = model.add_parameters({action_size});
		p_empty = model.add_parameters({2 * lstm_dim});
	}


	// LEGAL FUNCTION for hybrid, now for simplicity only consider unlabeled transition
	static bool IsActionFeasible(const string& a, unsigned bsize, unsigned ssize) {
		if (a[0] == 'S') {
			if (bsize > 0) return true;
		} else if (a[0] == 'L') {
			if (bsize > 0 && ssize > 1) return true;
		} else if (a[0] == 'R') {
			if ((ssize > 2) || ((ssize == 2) && (bsize == 0))) return true;
		}
		return false;
	}

	// Dynamic Oracle for hybrid
	static bool isActionZeroCost(const string& a, const vector<int>& stacki, const vector<int>& bufferi,
	                             const vector<int>& gold_head, map<int, vector<int>>& gold_children) {
		if (a[0] == 'S') {
			int buffer_top = bufferi.back();
			int buffer_top_head = gold_head.at(buffer_top);
			const vector<int>& buffer_top_children = gold_children[buffer_top];
			auto pos = --(stacki.end());
			auto in_stack = [&stacki](int i) -> bool {
				return find(stacki.begin(), stacki.end(), i) != stacki.end();
			}; // oh my lambda
			if ((find(stacki.begin(), pos, buffer_top_head) != pos)
			    || (any_of(buffer_top_children.begin(), buffer_top_children.end(), in_stack)))
				return false;
		} else if (a[0] == 'L') {
			int stack_top = stacki.back();
			int stack_top_head = gold_head.at(stack_top);
			const vector<int>& stack_top_children = gold_children[stack_top];
			auto bpos = --(bufferi.end());
			auto spos = --(stacki.end());
			auto in_buffer = [&bufferi](int i) -> bool {
				return find(bufferi.begin(), bufferi.end(), i) != bufferi.end();
			}; // oh my lambda
			if ((find(stacki.begin(), spos, stack_top_head) != spos)
			    || (find(bufferi.begin(), bpos, stack_top_head) != bpos)
			    || (any_of(stack_top_children.begin(), stack_top_children.end(), in_buffer))) {
				return false;
			}
		} else if (a[0] == 'R') {
			int stack_top = stacki.back();
			int stack_top_head = gold_head.at(stack_top);
			const vector<int>& stack_top_children = gold_children[stack_top]; // WARNING: here must use operator[], cause some nodes have no children
			auto in_buffer = [&bufferi](int i) -> bool {
				return find(bufferi.begin(), bufferi.end(), i) != bufferi.end();
			}; // oh my lambda
			if ((find(bufferi.begin(), bufferi.end(), stack_top_head) != bufferi.end())
			    || (any_of(stack_top_children.begin(), stack_top_children.end(), in_buffer)))
				return false;
		}
		return true;
	}

	bool isActionZeroCost(const string& a, const vector<int>& stacki, const vector<int>& bufferi,
	                      const vector<int>& gold_head, const vector<unsigned>& gold_deprel,
	                      map<int, vector<int>>& gold_children) {
		bool zerocost_transition = isActionZeroCost(a, stacki, bufferi, gold_head, gold_children);
		if (a[0] == 'S') {
			return zerocost_transition;
		} else {
			assert(a[0] == 'L' || a[0] == 'R');
			int stack_top = stacki.back();
			size_t first_char_in_rel = a.find('-') + 1;
			string hyp_rel = a.substr(first_char_in_rel);
			bool zerocost_deprel = deprel.stoi.at(hyp_rel) == gold_deprel.at(stack_top);
			return zerocost_transition && zerocost_deprel;
		}
	}

	// take a vector of actions and return a parse tree (labeling of every
	// word position with its head's position) heads & rels, inclu that for <root>
	tuple<vector<int>, vector<string>> compute_heads(unsigned sent_len, const vector<unsigned>& actions) {
		vector<int> heads(sent_len, -1);
		vector<string> rels(sent_len, "");

		vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
		for (unsigned i = 0; i < sent_len; ++i)
			bufferi[sent_len - i] = i;
		bufferi[0] = -999;
		stacki.push_back(bufferi.back()); bufferi.pop_back();

		for (auto aci : actions) { // loop over transitions for sentence
			const string& actionString = action.itos.at(aci);
			const char ac = actionString[0];
			if (ac == 'S') { // SHIFT
				assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
				stacki.push_back(bufferi.back());
				bufferi.pop_back();
			} else { // LEFT or RIGHT
				assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
				assert(ac == 'L' || ac == 'R');

				size_t first_char_in_rel = actionString.find('-') + 1;
				string hyp_rel = actionString.substr(first_char_in_rel);
				unsigned depi = 0, headi = 0;
				depi  = stacki.back();
				stacki.pop_back();
				headi = ac == 'R' ? stacki.back() : bufferi.back();
				heads[depi] = headi;
				rels[depi] = hyp_rel;
			}
		}
		assert(bufferi.size() == 1);
		assert(stacki.size() == 2);
		return make_tuple(heads, rels);
	}

	static tuple<unsigned, unsigned> compute_correct(const vector<int>& ref, const vector<int>& hyp,
	    const vector<string>& rel_ref, const vector<string>& rel_hyp,
	    unsigned len) {
		unsigned correct_head = 0, correct_rel = 0;
		for (unsigned i = 1; i < len; ++i) {
			if (hyp[i] == ref[i]) {
				++ correct_head;
				if (rel_hyp[i] == rel_ref[i]) ++correct_rel;
			}
		}
		return make_tuple(correct_head, correct_rel);
	}

	void output_conll(const vector<string>& sent, const vector<string>& sent_pos,
	                  const vector<int>& hyp, const vector<string>& rel_hyp) {
		for (unsigned i = 1; i < sent.size(); ++i) {
			string wit = sent[i];
			string pit = sent_pos[i];
			int hyp_head = hyp[i];
			string hyp_rel = rel_hyp[i];
			cout << i << '\t'          // 1. ID
			     << wit << '\t'         // 2. FORM
			     << "_" << '\t'         // 3. LEMMA
			     << "_" << '\t'         // 4. CPOSTAG
			     << pit << '\t'         // 5. POSTAG
			     << "_" << '\t'         // 6. FEATS
			     << hyp_head << '\t'    // 7. HEAD
			     << hyp_rel << '\t'     // 8. DEPREL
			     << "_" << '\t'         // 9. PHEAD
			     << "_" << endl;        // 10. PDEPREL
		}
		cout << endl;
	}


	Expression collect_state(const vector<Element>& stack, const vector<Element>& buffer,
	                         const vector<Expression>& lstm_out, const Expression& empty) {
		Element empty_element { -999, -999, -999};
		const Element* cur = nullptr;
		vector<int> indices; indices.reserve((window_size + 1) * nnvecs);
		for (size_t n = 0; n < window_size; ++n) {
			cur = (stack.size() > n) ? &(stack.at(stack.size() - 1 - n)) : &empty_element;
			if (use_head) indices.push_back(cur->id);
			if (use_rl || use_rlmost) {
				indices.push_back(cur->rc);
				indices.push_back(cur->rc);
			}
		}
		cur = (!buffer.empty()) ? &buffer.at(buffer.size() - 1) : &empty_element;
		if (use_head) indices.push_back(cur->id);
		if (use_rl || use_rlmost) {
			indices.push_back(cur->rc);
			indices.push_back(cur->rc);
		}
		vector<Expression> lstm_vecs; lstm_vecs.reserve((window_size + 1) * nnvecs);
		for (auto i : indices) {
			lstm_vecs.push_back(i == -999 ? empty : lstm_out.at(i));
		}
		Expression state = concatenate(lstm_vecs);
		return state;
	}

	Result hinge_parser(ComputationGraph& hg, const vector<unsigned>& sent,
	                    const vector<unsigned>& sent_pos, const vector<int>& gold_head,
	                    const vector<unsigned>& gold_deprel) {
		const bool build_training_graph = (!gold_head.empty()) && (!gold_deprel.empty());
		vector<unsigned> results;
		vector<Expression> step_losses;
		step_losses.push_back(dynet::zeros(hg, {1})); // in-case there is nothing here
		unsigned right = 0;

		// engraph
		sflstm.new_graph(hg); sflstm.start_new_sequence();
		sblstm.new_graph(hg); sblstm.start_new_sequence();
		bflstm.new_graph(hg); bflstm.start_new_sequence();
		bblstm.new_graph(hg); bblstm.start_new_sequence();

		Expression p2h1 = parameter(hg, p_p2h1);
		Expression h1b = parameter(hg, p_h1b);
		Expression h1h2, h2b;
		if (hidden2_dim) {
			h1h2 = parameter(hg, p_h1h2);
			h2b = parameter(hg, p_h2b);
		}
		Expression h2a = parameter(hg, p_h2a);
		Expression abias = parameter(hg, p_abias);

		Expression rp2h1 = parameter(hg, p_rp2h1);
		Expression rh1b = parameter(hg, p_rh1b);
		Expression rh1h2, rh2b;
		if (hidden2_dim) {
			rh1h2 = parameter(hg, p_rh1h2);
			rh2b = parameter(hg, p_rh2b);
		}
		Expression rh2a = parameter(hg, p_rh2a);
		Expression rabias = parameter(hg, p_rabias);
		Expression empty = parameter(hg, p_empty);

		// unk-replacement & compute gold_children
		vector<unsigned> tsent = sent;
		map<int, vector<int>> gold_children;
		if (build_training_graph) {
			const unsigned unk_id = form.stoi.at(Vocab::UNK);
			for (size_t i = 0; i < sent.size(); ++i) {
				unsigned c = form.itofreq.at(sent[i]);
				if (dynet::rand01() > c / (c + 0.25))
					tsent[i] = unk_id;
			}
			for (unsigned i = 1; i < gold_head.size(); ++i)
				gold_children[gold_head[i]].push_back(i);
		}

		// building embeddings
		vector<Expression> inputs; inputs.reserve(sent.size()); // no guard no dummy staffs
		for (size_t i = 0; i < sent.size(); ++i) {
			vector<Expression> vecs;
			vecs.push_back(lookup(hg, p_w, tsent[i]));
			if (use_pos)
				vecs.push_back(lookup(hg, p_p, sent_pos[i]));
			if (use_pretrained)
				vecs.push_back(lookup(hg, p_t, sent[i]));
			Expression vec = concatenate(vecs);
			inputs.push_back(vec);
		}

		// pass through slstm
		vector<Expression> slstm_out; slstm_out.reserve(inputs.size());
		{
			vector<Expression> sflstm_out; sflstm_out.reserve(inputs.size());
			vector<Expression> sblstm_out; sblstm_out.reserve(inputs.size());
			for (size_t i = 0; i < inputs.size(); ++i) {
				sflstm_out.push_back(sflstm.add_input(inputs[i]));
				sblstm_out.push_back(sblstm.add_input(inputs[inputs.size() - 1 - i]));
			}
			for (size_t i = 0; i < inputs.size(); ++i)
				slstm_out.push_back(concatenate({sflstm_out[i], sblstm_out[i]}));
		}

		// pass through bsltm
		vector<Expression> blstm_out; blstm_out.reserve(inputs.size());
		{
			vector<Expression> bflstm_out; bflstm_out.reserve(inputs.size());
			vector<Expression> bblstm_out; bblstm_out.reserve(inputs.size());
			for (size_t i = 0; i < slstm_out.size(); ++i) {
				bflstm_out.push_back(bflstm.add_input(slstm_out[i]));
				bblstm_out.push_back(bblstm.add_input(slstm_out[slstm_out.size() - 1 - i]));
			}
			for (size_t i = 0; i < slstm_out.size(); ++i)
				blstm_out.push_back(concatenate({bflstm_out[i], bblstm_out[i]}));
		}

		// prepare stack and buffer, only indices
		vector<Element> buffer; buffer.reserve(sent.size() - 1);
		vector<int> bufferi(sent.size() - 1);
		int slen = static_cast<int>(sent.size());
		for (int i = 1; i < slen; ++i) {
			buffer.push_back(Element {slen - 1 - i, slen - 1 - i, slen - 1 - i});
			bufferi[slen - i - 1] = i;
		} // mind that buffer is reversed

		vector<Element> stack {Element{0, 0, 0}};
		vector<int> stacki {0};

		// main loop
		while (stack.size() > 1 || buffer.size() > 0) {
			// get list of possible actions for the current parser state
			vector<unsigned> current_valid_actions;
			for (const auto& act : action.stoi)
				if (IsActionFeasible(act.first, buffer.size(), stack.size()))
					current_valid_actions.push_back(act.second);
			assert(current_valid_actions.size() > 0);

			// compute scores
			Expression state = collect_state(stack, buffer, blstm_out, empty);
			Expression layer1 = tanh(affine_transform({h1b, p2h1, state}));
			Expression rlayer1 = tanh(affine_transform({rh1b, rp2h1, state}));
			Expression out = affine_transform({abias, h2a,
			                                   (hidden2_dim) ? affine_transform({h2b, h1h2, layer1}) : layer1
			                                  });
			Expression rout = affine_transform({rabias, rh2a,
			                                    (hidden2_dim) ? affine_transform({rh2b, rh1h2, rlayer1}) : rlayer1
			                                   });
			vector<Expression> score_frags;
			const int shift_id = transition.stoi.at("Shift");
			const int left_id = transition.stoi.at("LeftReduce");
			const int right_id = transition.stoi.at("RightReduce");
			const int action_size = 2 * deprel.stoi.size() + 1;
			score_frags.push_back(pick(out, shift_id) + pick(rout, shift_id));
			score_frags.push_back(pick(out, left_id) + strided_select(rout, {2}, {left_id}, {action_size}));
			score_frags.push_back(pick(out, left_id) + strided_select(rout, {2}, {right_id}, {action_size}));
			Expression scores_e = concatenate(score_frags);
			vector<float> scores = as_vector(hg.incremental_forward(scores_e));

			float best_score = scores[current_valid_actions[0]];
			unsigned best_a = current_valid_actions[0];
			for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
				if (scores[current_valid_actions[i]] > best_score) {
					best_score = scores[current_valid_actions[i]];
					best_a = current_valid_actions[i];
				}
			}
			unsigned act_to_take = best_a; // greedy (when testing)
			if (build_training_graph) {
				// compute dynamic oracle and check if the argmax is right
				vector<unsigned> dynamic_oracle;
				string act;
				for (unsigned aci : current_valid_actions) {
					act = action.itos.at(aci);
					if (isActionZeroCost(act, stacki, bufferi, gold_head, gold_deprel, gold_children))
						dynamic_oracle.push_back(aci);
				}
				if (find(dynamic_oracle.begin(), dynamic_oracle.end(), best_a) != dynamic_oracle.end())
					++right;

				// accumulate loss, well multi-label hinge has to be computed manually
				unsigned worst_oraclei = dynamic_oracle[0];
				float worst_oracle = std::numeric_limits<float>::max();
				unsigned best_wrongi = current_valid_actions[0];
				float best_wrong = -std::numeric_limits<float>::max();
				for (unsigned aci : current_valid_actions) {
					if (find(dynamic_oracle.begin(), dynamic_oracle.end(), aci) != dynamic_oracle.end()) {
						if (scores[aci] < worst_oracle) {
							worst_oracle = scores[aci]; worst_oraclei = aci;
						}
					} else {
						if (scores[aci] > best_wrong) {
							best_wrong = scores[aci]; best_wrongi = aci;
						}
					}
				}

				if (worst_oracle < best_wrong + 1.0) {
					Expression loss_t = pick(scores_e, best_wrongi) - pick(scores_e, worst_oraclei);
					step_losses.push_back(loss_t);
				}

				// choose action
				if (rand01() < p_agg) {
					act_to_take = best_wrongi;
				} else {
					act_to_take = worst_oraclei;
				}
			}

			// apply action
			results.push_back(act_to_take);
			const string& actionString = action.itos.at(act_to_take);
			if (actionString[0] == 'S') {
				stacki.push_back(bufferi.back()); bufferi.pop_back();
				stack.push_back(buffer.back()); buffer.pop_back();
			} else if (actionString[0] == 'R') {
				Element& head = stack.at(stack.size() - 2);
				Element& dep = stack.back();
				if (use_rl) {
					head.rc = dep.id;
				} else if (use_rlmost) {
					head.rc = dep.rc;
				}
				stack.pop_back(); stacki.pop_back();
			} else {
				Element& head = buffer.back();
				Element& dep = stack.back();
				if (use_rl) {
					head.lc = dep.id;
				} else if (use_rlmost) {
					head.lc = dep.lc;
				}
				stack.pop_back(); stacki.pop_back();
			}
		}
		// assert everythion is done
		assert(stacki.size() == 1 && bufferi.empty());
		assert(stack.size() == 1 && buffer.empty());

		// return a result
		Expression loss = sum(step_losses);
		return Result {results, loss, results.size(), right};
	}
};

struct Stat {
public:
	size_t num_prediction = 0;
	unsigned right = 0;
	double tot_loss = 0;
	size_t num_token = 0;
	unsigned right_head = 0;
	unsigned right_rel = 0;

public:
	Stat() {}
	Stat(size_t t_num_pred, unsigned t_right, double t_loss,
	     size_t t_num_token, unsigned t_head, unsigned t_rel):
		num_prediction(t_num_pred), right(t_right), tot_loss(t_loss),
		num_token(t_num_token), right_head(t_head), right_rel(t_rel) {}

	Stat& operator+=(const Stat& rhs) {
		num_prediction += rhs.num_prediction;
		right += rhs.right;
		tot_loss += rhs.tot_loss;
		num_token += rhs.num_token;
		right_head += rhs.right_head;
		right_rel += rhs.right_rel;
		return *this;
	}

	friend Stat operator+(Stat lhs, const Stat& rhs) {
		lhs += rhs;
		return lhs;
	}

	bool operator<(const Stat& rhs) {
		// FIX: smaller loss translates into better uas or las
		return double(right_head) / num_token > double(rhs.right_head) / rhs.num_token;
	}

	friend ostream& operator<<(ostream& stream, const Stat& stat) {
		// since there is no probabilistic interpretation, ppl is meaningless
		stream << "loss per action: " << double(stat.tot_loss) / stat.num_prediction
		       << "\tuas: " << double(stat.right_head) / stat.num_token
		       << "\tlas: " << double(stat.right_rel) / stat.num_token;
		return stream;
	}
};

class Learner : public ILearner<Sentence, Stat> {
public:
	explicit Learner(ArcHybridLSTM& t_parser, Trainer& t_trainer,
	                 string t_model_params, string t_trainer_state, unsigned t_data_size) :
		parser(t_parser), trainer(t_trainer), model_params(t_model_params), trainer_state(t_trainer_state), data_size(t_data_size) {
	}

	~Learner() {}

	Stat LearnFromDatum(const Sentence& sent, bool learn) {
		ComputationGraph hg;
		Result result = parser.hinge_parser(hg, sent.formi, sent.posi,
		                                    learn ? sent.head : vector<int>(),
		                                    learn ? sent.depreli : vector<unsigned>());
		double loss = as_scalar(hg.forward(result.loss));
		unsigned h = 0, r = 0;
		if (learn) {
			hg.backward(result.loss);
		} else {
			vector<int> hyp; vector<string> rels_hyp;
			tie(hyp, rels_hyp) = parser.compute_heads(sent.form.size(), result.actions);
			tie(h, r) = parser.compute_correct(sent.head, hyp, sent.deprel, rels_hyp, sent.form.size());
		}
		return Stat {result.actions.size(), result.right, loss, sent.formi.size() - 1, h, r};
	}

	void SaveModel() {
		TextFileSaver s(model_params); s.save(parser.pc);
		ofstream os(trainer_state); trainer.save(os);
	}

private:
	ArcHybridLSTM& parser;
	Trainer& trainer;
	string model_params;
	string trainer_state;
	unsigned data_size;
};

template<class D, class S>
void run_single_process2(ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
    const std::vector<D>& dev_data, unsigned num_iterations, unsigned dev_frequency, unsigned report_frequency, unsigned batch_size) {
  std::vector<unsigned> train_indices(train_data.size());
  std::iota(train_indices.begin(), train_indices.end(), 0);

  std::vector<unsigned> dev_indices(dev_data.size());
  std::iota(dev_indices.begin(), dev_indices.end(), 0);
	
	float init_lr = trainer->learning_rate;

  S best_dev_loss = S();
  bool first_dev_run = true;
  unsigned batch_counter = 0;
  for (unsigned iter = 0; iter < num_iterations && !stop_requested; ++iter) {
    // Shuffle the training data indices
    std::shuffle(train_indices.begin(), train_indices.end(), *rndeng);

    S train_loss = S();

    unsigned data_processed = 0;
    unsigned data_until_report = report_frequency;
    std::vector<unsigned>::iterator begin = train_indices.begin();
    while (begin != train_indices.end()) {
      std::vector<unsigned>::iterator end = begin + dev_frequency;
      if (end > train_indices.end()) {
        end = train_indices.end();
      }
      S batch_loss;
      for (auto it = begin; it != end; ++it) {
        unsigned i = *it;
        DYNET_ASSERT(i < train_data.size(), "Out-of-bounds ID in train set for multiprocessing");
        const D& datum = train_data[i];
        S datum_loss = learner->LearnFromDatum(datum, true);
        batch_loss += datum_loss;
        train_loss += datum_loss;
        if (++batch_counter == batch_size) {
          // TODO: The scaling was originally this
          // trainer->update(1.0 / batch_size); 
          trainer->update(); 
          batch_counter = 0;
        }
        data_processed++;

        if (--data_until_report == 0) {
					trainer->status();
          data_until_report = report_frequency;
          double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
          std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << std::endl;
          batch_loss = S();
        }
      }
      
      trainer ->learning_rate = init_lr / (1 + 0.1 * iter);

      S dev_loss = S();
      for (auto it = dev_indices.begin(); it != dev_indices.end(); ++it) {
        unsigned i = *it;
        DYNET_ASSERT(i < dev_data.size(), "Out-of-bounds ID in dev set for multiprocessing");
        const D& datum = dev_data[i];
        S datum_loss = learner->LearnFromDatum(datum, false);
        dev_loss += datum_loss;
      }
      bool new_best = (first_dev_run || dev_loss < best_dev_loss);
      first_dev_run = false;
      double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
      std::cerr << fractional_iter << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
      if (stop_requested) {
        break;
      }
      if (new_best) {
        learner->SaveModel();
        best_dev_loss = dev_loss;
      }

      begin = end;
    }
  }
}

int main(int argc, char** argv) {
	// parse command line args and prepare args for ParserBuilder
	po::variables_map conf;
	InitCommandLine(argc, argv, conf);

	auto training_data = conf["training_data"].as<string>();
	auto dev_data = conf["dev_data"].as<string>();
	auto test_data = conf["test_data"].as<string>();

	// Read in corpus
	cerr << "Reading corpus..." << endl;
	Corpus corpus;
	corpus.load_corpus(corpus.train_sentences, training_data);
	corpus.load_corpus_dev(corpus.dev_sentences, dev_data);
	corpus.load_corpus_dev(corpus.test_sentences, test_data);
// 	cerr << corpus.form << endl << corpus.pos << endl
// 		<< corpus.deprel << endl << corpus.transition << endl
// 		<< corpus.chars << endl;

	// Special operation to make a symmetric action vocab
	Vocab action {vector<string>{}, false};
	action.get_or_add("Shift");
	for (unsigned i = 0; i < corpus.deprel.stoi.size(); ++i) {
		action.get_or_add(string {"LeftReduce-"} + corpus.deprel.itos.at(i));
		action.get_or_add(string {"RightReduce-"} + corpus.deprel.itos.at(i));
	}
	cerr << "action assembled:\n" << action << endl;

	const auto epoch = conf["epoches"].as<unsigned>();
	const auto lr = conf["init_learning_rate"].as<double>();
	bool resume = conf.count("resume");
	const auto param = conf["param"].as<string>();
	const auto trainer_state = conf["trainer_state"].as<string>();

	const auto layers = conf["layers"].as<unsigned>();
	const auto lstm_dim = conf["lstm_dim"].as<unsigned>();
	const auto hidden1_dim = conf["hidden1_dim"].as<unsigned>();
	const auto hidden2_dim = conf["hidden2_dim"].as<unsigned>();
	const auto input_dim = conf["input_dim"].as<unsigned>();

	// const auto transition_size = corpus.transition.stoi.size();
	const auto vocab_size = corpus.form.stoi.size();

	const auto use_head = conf["use_head"].as<bool>();
	const auto use_rl = conf["use_rl"].as<bool>();
	const auto use_rl_most = conf["use_rl_most"].as<bool>();

	const auto window_size = conf["window_size"].as<unsigned>();
	const auto exp_prob = conf["exp_prob"].as<float>();

	// read in pretrained word embedding, I love iostream, stringstream, string, so good
	unordered_map<unsigned, vector<float>> pretrained;
	bool use_pretrained = conf.count("words");
	if (use_pretrained) {
		auto pretrained_dim = conf["pretrained_dim"].as<unsigned>();
		const unsigned unk_id = corpus.form.get_or_add(Vocab::UNK);
		unsigned max_id = corpus.form.stoi.size();
		pretrained[unk_id] = vector<float>(pretrained_dim, 0);
		cerr << "Loading from " << conf["words"].as<string>() << " with" << pretrained_dim << " dimensions\n";
		ifstream in(conf["words"].as<string>());
		string line;
		getline(in, line);
		vector<float> v(pretrained_dim, 0);
		string word;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> word;
			for (unsigned i = 0; i < pretrained_dim; ++i) lin >> v[i];
			unsigned id = corpus.form.get_or_add(word);
			if (id != unk_id)
				pretrained[id] = v;
			else
				pretrained[max_id++] = v; // pretrained has possibly more word entries than corpus.form.itos
		}
	}

	bool use_pos = conf["use_pos_tags"].as<bool>();
	unsigned pos_size = 0;
	unsigned pos_dim = 0;
	if (use_pos) {
		pos_size = corpus.pos.stoi.size();
		pos_dim = conf["pos_dim"].as<unsigned>();
	}
	const unsigned status_every_i_iterations = 100;

	// print conf
	cerr << "======Configuration is=====" << endl
	     << boolalpha
	     << "training_data: " << training_data << endl
	     << "dev_data: " << dev_data << endl
	     << "test_data" << test_data << endl
	     << "training epoches: " << epoch << endl
	     << "init learning rate: " << lr << endl
	     << "resume: " << resume << endl
	     << "param path: " << param << endl
	     << "trainer state: " << trainer_state << endl
	     << "input_dim: " << input_dim << endl
	     << "vocab_size: " << vocab_size << endl
	     << "layers: " << layers << endl
	     << "lstm_dim: " << lstm_dim << endl
	     << "hidden1_dim: " << hidden1_dim << endl
	     << "hidden2_dim: " << hidden2_dim << endl
	     << "use_pos_tags: " << use_pos << endl;
	if (use_pos)
		cerr << "pos_dim: " << pos_dim << endl
		     << "pos_size: " << pos_size << endl;
	cerr << "use_pretrained: " << use_pretrained << endl;
	if (use_pretrained) {
		cerr << "pretrained embeddings size is: " << pretrained.size() << endl;
	}

	cerr << "use_head: " << use_head << endl
	     << "use_rl: " << use_rl << endl
	     << "use_rl_most: " << use_rl_most << endl
	     << "window_size: " << window_size << endl;

	cerr << "====================" << endl;

	// dynet build model
	dynet::initialize(argc, argv, true);
	ParameterCollection model;
	ArcHybridLSTM parser(model, corpus.form, corpus.transition, corpus.deprel, action,
	                     input_dim, pos_dim, layers, lstm_dim, window_size, hidden1_dim,
	                     hidden2_dim, vocab_size, pos_size, use_head, use_rl, use_rl_most,
	                     use_pos, use_pretrained, exp_prob, pretrained);
	if (conf.count("train")) {
		SimpleSGDTrainer trainer(model);
		if (conf.count("resume")) {
			cerr << "Loading model parameters from " << param << endl 
					<< "Loading trainer state from " << trainer_state << endl;
			TextFileLoader l(param); l.populate(model);
			ifstream is(trainer_state); trainer.populate(is);
		}
		Learner learner(parser, trainer, param, trainer_state, corpus.train_sentences.size());
		run_single_process2(&learner, &trainer, corpus.train_sentences, corpus.dev_sentences,
		                   epoch, corpus.train_sentences.size(), status_every_i_iterations, 1);
// 		run_multi_process(4, &learner, &trainer, corpus.train_sentences, corpus.dev_sentences,
// 		                   epoch, corpus.train_sentences.size(), status_every_i_iterations);
	} else {
		cerr << "Loading model parameters from " << param << endl;
		TextFileLoader l(param); l.populate(model);
		for (auto& sent : corpus.test_sentences) {
			ComputationGraph hg;
			// Result result = parser.hinge_parser(hg, sent.formi, sent.posi, sent.head, sent.depreli);
			Result result = parser.hinge_parser(hg, sent.formi, sent.posi, {}, {});
// 			cerr << result.actions << endl
// 			     << as_scalar(hg.incremental_forward(result.loss)) << endl
// 			     << result.num_prediction << " predictions" << endl
// 			     << result.right << "right" << endl;

			vector<int> hyp; vector<string> hyp_rel;
			tie(hyp, hyp_rel) = parser.compute_heads(sent.form.size(), result.actions);
//			unsigned correct_head = 0; unsigned correct_rel = 0;
// 			tie(correct_head, correct_rel) = parser.compute_correct(sent.head, hyp,
// 			                                 sent.deprel, hyp_rel, sent.form.size());
			parser.output_conll(sent.form, sent.pos, hyp, hyp_rel);
		}
	}
	return 0;
}
