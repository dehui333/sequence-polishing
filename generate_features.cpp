#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <utility>
#include <random>
#include <set>
#include <string>
#include <cmath>
#include <iostream>
#include <chrono>

#include "edlib.h"
#include "generate_features.h"

// For reverse strand add +6
std::unordered_map<Bases, uint8_t, EnumClassHash> ENCODED_BASES = {
    {Bases::A, 0},
    {Bases::C, 1},
    {Bases::G, 2},
    {Bases::T, 3},
    {Bases::GAP, 4},
    {Bases::UNKNOWN, 5}
};

// Changes char to int, where 0,1,2,3,4,5 represent ACGT*N
static constexpr uint8_t CHAR_TO_FORWARD_INT_MAP[] = {
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 0, 5, 1, 5, 5,
    5, 2, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 3, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5
};

// changes int values (forward strand, 0-5) to char
static constexpr char FORWARD_INT_TO_CHAR_MAP[] = {'A', 'C', 'G', 'T', '*', 'N'};

FeatureGenerator::FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict, uint16_t median, uint16_t mad): draft(ref) {
    
    bam = readBAM(filename);
    // normalized_median_cov  // x3
    /*auto pileup_inclusive = bam->pileup(region, true);
    
    uint32_t i = 0;
    while (pileup_inclusive->has_next()) {
        auto column = pileup_inclusive->next();
        long rpos = column->position;
        if (rpos < pileup_inclusive->start()) continue;
        if (rpos >= pileup_inclusive->end()) break;
        auto& stat = stats_info[std::make_pair(rpos, 0)];
        stat.normalized_cov = (column->count() - (float) median) / mad;
    } // x3 */

    pileup_iter = bam->pileup(region);
    if (dict == Py_None) { 
        has_labels = false;
    } else {
        convert_py_labels_dict(dict);
        has_labels = true;
    }
    
}

void FeatureGenerator::convert_py_labels_dict(PyObject *dict) {

    Py_ssize_t pos = 0;
    PyObject *key = NULL;
    PyObject *value = NULL;    

    if (! PyDict_Check(dict)) {
        PyErr_Format(PyExc_TypeError, 
                "Argument \"dict\" to %s must be dict not \"%s\"", 
                __FUNCTION__, Py_TYPE(dict)->tp_name);	       
    }
    while (PyDict_Next(dict, &pos, &key, &value)) {
        if (! PyTuple_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "A key of dict is not a tuple!");
            labels.clear();
        } 
        if (PyTuple_Size(key) != static_cast<Py_ssize_t>(2)) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict is not a pair!");
            labels.clear();
        }
        PyObject *pair_item0 = PyTuple_GetItem(key, 0);
        PyObject *pair_item1 = PyTuple_GetItem(key, 1);
        if ((!PyLong_Check(pair_item0)) || (!PyLong_Check(pair_item1))) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict does contain two longs!");
            labels.clear();
        }
        if (! PyLong_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "A value of dict is not of long type!");
            labels.clear();
        }
        pos_index_t pair_item0_c = PyLong_AsUnsignedLong(pair_item0);
        pos_index_t pair_item1_c = PyLong_AsUnsignedLong(pair_item1);
        uint8_t value_c = PyLong_AsUnsignedLong(value);
        if (PyErr_Occurred()) {
            labels.clear();
        }
        labels.emplace(std::make_pair(pair_item0_c, pair_item1_c), value_c);
    }
}


void FeatureGenerator::increment_base_count(std::pair<pos_index_t, pos_index_t>& index, PosInfo& pos_info) {
    Bases b = pos_info.base;
    uint8_t mq = pos_info.mq;
    auto& s = stats_info[index];
    s.n_total += mq;
    Bases draft_base = Bases::GAP;
    // index here is a pair of indices: index.first = draft_base index, index.second = insertion index
    if (index.second == 0) {
        draft_base = char_to_base(draft[index.first]);
    }
    bool diff = draft_base != b;
    switch(b) {
        case Bases::A:
            s.n_A += mq;
            if (diff && s.n_A > s.largest_diff) {
               s.largest_diff = s.n_A;
            }
            break;
        case Bases::C:
            s.n_C += mq;
            if (diff && s.n_C > s.largest_diff) {
                s.largest_diff = s.n_C;
            }
            break;
        case Bases::G:
            s.n_G += mq;
            if (diff && s.n_G > s.largest_diff) {
                s.largest_diff = s.n_G;
            }
            break;
        case Bases::T:
            s.n_T += mq;
            if (diff && s.n_T > s.largest_diff) {
                s.largest_diff = s.n_T;
            }
            break;
        case Bases::GAP:
            s.n_GAP += mq;
            if (diff && s.n_GAP > s.largest_diff) {
                s.largest_diff = s.n_GAP;
            }
            break;
        case Bases::UNKNOWN:
            std::cout << "Unknown base!" << std::endl;
    }
}


Bases FeatureGenerator::char_to_base(char c) {

    return static_cast<Bases>(static_cast<int>(CHAR_TO_FORWARD_INT_MAP[static_cast<uint8_t>(c)]));
}

char FeatureGenerator::base_to_char(Bases b) {

    return FORWARD_INT_TO_CHAR_MAP[static_cast<int>(b)];
}

char FeatureGenerator::forward_int_to_char(uint8_t i) {

    return FORWARD_INT_TO_CHAR_MAP[i];
}

uint8_t FeatureGenerator::char_to_forward_int(char c) {

    return CHAR_TO_FORWARD_INT_MAP[static_cast<uint8_t>(c)];
}


// this function will be called for each index
void FeatureGenerator::pos_queue_push(std::pair<pos_index_t, pos_index_t>& index) {
    bool is_uncertain = false;
    auto& s = stats_info[index];
    uint16_t num_total = s.n_total;
    if (index.second != 0) { // if this is an inserted base
        if (num_total == 0 || (float) s.largest_diff/ num_total < NON_GAP_THRESHOLD) { // NON_GAP_THRESHOLD = 0.01
            align_info.erase(index);
            return;
        }
    }
    if ((float) s.largest_diff/num_total >= UNCERTAIN_POSITION_THRESHOLD) { // UNCERTAIN_POSITION_THRESHOLD = 0.15
        is_uncertain = true;
    } 
    if (is_uncertain) {
        pos_queue.push_back(index);
        distances.push(counter);
        counter = 0;

    } else {
        pos_queue.push_back(index);
        counter++;
    }
}

void FeatureGenerator::pos_queue_pop(uint16_t num) {
    for (auto it = pos_queue.begin(), end = pos_queue.begin() + num; it != end; ++it) {
        align_info.erase(*it);
    }
    
    pos_queue.erase(pos_queue.begin(), pos_queue.begin() + num);
    
    while (distances.size() > 0 && num >=(distances.front()+1)) {
        num -= distances.front() + 1; 
        distances.pop();
    }
    if (distances.empty()) { 
        counter -= num; 
    } else {
        distances.front() -= num;
    }

}

// base_index = rpos, segments = ins_segments (includes one truth segment), target_index = best segment index, no_ins_reads = no_ins_reads
void FeatureGenerator::align_to_target(pos_index_t base_index, std::vector<segment>& segments, int target_index, 
        std::vector<segment>& no_ins_reads) {

    std::vector<segment*> non_label_seqs;
    non_label_seqs.reserve(segments.size());

    segment& target = segments[target_index];

    std::unordered_map<uint32_t, PosInfo> target_positions[target.sequence.size()];
    
    std::vector<std::unordered_map<uint32_t, PosInfo>> ins_positions[target.sequence.size()+1];
    uint8_t target_positions_labels[target.sequence.size()];
    for (unsigned int i = 0; i < target.sequence.size(); i ++) {
        target_positions_labels[i] = ENCODED_BASES[Bases::GAP];
    }
    
    std::vector<uint8_t> ins_positions_labels[target.sequence.size()+1];
    
    int total_ins_pos = 0;
    for (auto& s : segments) {
        if (s.index != LABEL_SEQ_ID) non_label_seqs.push_back(&s);
        if (s.index != target.index) {
            EdlibAlignResult result = edlibAlign(s.sequence.c_str(), s.sequence.size(), target.sequence.c_str(),
                    target.sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_PATH, NULL, 0)); 
            int ref_pos = -1;
            int query_pos = -1;
            unsigned int ins_index = 0;
            char char_at_pos;
            Bases base_at_pos;
            for (int i = 0; i < result.alignmentLength; i++) {
                switch (result.alignment[i]) {
                    case 0: // match
                        ins_index = 0;
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == LABEL_SEQ_ID) {
                            target_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else { // else it is just an insertion segment (not target, not ground truth)
                            target_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos, s.mq));
                        }
                        break;
                    case 1: // insertion [has base on insertion segment s, del on target]
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        // store the insertion segment
                        if (ins_positions[ref_pos+1].size() < ins_index + 1) { // if not enough maps to record sequence at that position
                            ins_positions[ref_pos+1].push_back(std::unordered_map<uint32_t, PosInfo>{});
                            ins_positions_labels[ref_pos+1].push_back(ENCODED_BASES[Bases::GAP]);
                            total_ins_pos++;
                        }
                        if (s.index == LABEL_SEQ_ID) {
                            ins_positions_labels[ref_pos+1][ins_index] = char_to_forward_int(char_at_pos);
                        } else {
                            ins_positions[ref_pos+1][ins_index].emplace(s.index, PosInfo(base_at_pos, s.mq));
                        }
                        ins_index++;
                        break;
                    case 2: // deletion has base on target, del on insertion segment]
                        ins_index = 0;
                        ref_pos++;
                        break;
                    case 3: // mismatch, same as match
                        ins_index = 0; 
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == LABEL_SEQ_ID) {
                            target_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else {
                            target_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos, s.mq));
                        }
                        break;
                    default:
                        std::cout << "Unknown alignment result!\n";
                }
            }
            edlibFreeAlignResult(result); // cleaning
        } else { // if it is target, also might be the ground truth
            for (unsigned int i = 0; i < s.sequence.size(); i++) {
                const char char_at_pos = s.sequence[i];
                Bases base_at_pos = char_to_base(char_at_pos);               
                if (s.index == LABEL_SEQ_ID) {
                    target_positions_labels[i] = char_to_forward_int(char_at_pos);
                } else {
                    target_positions[i].emplace(s.index, PosInfo(base_at_pos, s.mq));
                }
            }
        }
    }
    // saved all segments into non_lalbels_seqs, target_positions, ins_positions, target_posisions_labels and ins_positions_labels
    
    pos_index_t count = 1;
    // insertions before the first position of target
    for (unsigned int i = 0; i < ins_positions[0].size(); i++) {
        auto& map = ins_positions[0][i];
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
        
        count++;
        
        for (uint16_t k = 0; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (map.find(s->index) == map.end()) {
                map.emplace(s->index, PosInfo(Bases::GAP, s->mq));
            }
        }
        for (auto& s: no_ins_reads) { 
            map.emplace(s.index, PosInfo(Bases::GAP, s.mq));
        }
        for (auto& pair: map) {
            increment_base_count(index, pair.second);
        }
        align_info[index] = map;
        if (has_labels) {
            labels_info[index] = ins_positions_labels[0][i];
        }
        pos_queue_push(index);
    }

    // target sequence before aligning, and insertions after target (the inner loop)
    for (unsigned int i = 0; i < target.sequence.size(); i++) {
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);

        count++;
        
        for (uint16_t k = 0 ; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (target_positions[i].find(s->index) == target_positions[i].end()) {
                target_positions[i].emplace(s->index, PosInfo(Bases::GAP, s->mq));
            }
        }
        for (auto& s: no_ins_reads) {
            target_positions[i].emplace(s.index, PosInfo(Bases::GAP, s.mq));
        }
        for (auto& pair: target_positions[i]) {
            increment_base_count(index, pair.second);  
        }

        align_info[index] = target_positions[i];
        pos_queue_push(index); 
        if (has_labels) {
            labels_info[index] = target_positions_labels[i];
        }

        for (unsigned int j = 0; j < ins_positions[i+1].size(); j++) {
            auto& map = ins_positions[i+1][j];
            auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
          
            count++;
            
            for (uint16_t k = 0; k< non_label_seqs.size(); k++) {
                auto& s = non_label_seqs[k];
                if (map.find(s->index) == map.end()) {
                    map.emplace(s->index, PosInfo(Bases::GAP, s->mq));
                }
            }
            for (auto& s: no_ins_reads) {
                map.emplace(s.index, PosInfo(Bases::GAP, s.mq));
            }
            for (auto& pair: map) {
                increment_base_count(index, pair.second);  
            }

            align_info[index] = map;
            if (has_labels) {
                labels_info[index] = ins_positions_labels[i+1][j];
            }
            pos_queue_push(index);
        }
    }
}

int FeatureGenerator::find_center(std::vector<segment>& segments) {
    int dists[segments.size()]{0}; 
    for (unsigned int i = 0; i < segments.size(); i++) { 
        for (unsigned int j = i + 1; j < segments.size(); j++) { 
            EdlibAlignResult result = edlibAlign(segments[i].sequence.c_str(), segments[i].sequence.size(), segments[j].sequence.c_str(),
                    segments[j].sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, NULL, 0));
            dists[i] += result.editDistance;
            dists[j] += result.editDistance;
            
            edlibFreeAlignResult(result); // cleaning
        }
    }
    int best_pos_index = 0;
    for (unsigned int i = 0; i < segments.size(); i++) {
        if (segments[i].index == LABEL_SEQ_ID) continue;
        if (dists[i] < dists[best_pos_index]) { // get the smallest sum of edit distance, the "center"
            best_pos_index = i;
        }
    }
    return best_pos_index;

}

int FeatureGenerator::find_longest(std::vector<segment>& segments) {
    int best_index = 0;
    int highest_len = 0;
    for (int i = 0; i < segments.size(); i++) {
        if (segments[i].index == LABEL_SEQ_ID) continue;
        int len = segments[i].sequence.size();
        if (len > highest_len) {
            best_index = i;
            highest_len = len;
        }
    }
    return best_index;
}


// base_index = rpos
void FeatureGenerator::align_ins_longest(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int longest_index = find_longest(ins_segments); // the index of the longest read in ins_segments
    align_to_target(base_index, ins_segments, longest_index, no_ins_reads);

}

void FeatureGenerator::align_ins_center(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int center_index = find_center(ins_segments);
    align_to_target(base_index, ins_segments, center_index, no_ins_reads);

}

std::unique_ptr<Data> FeatureGenerator::generate_features() {
    npy_intp dims[2]; // dimensions of X1 (2d)
    npy_intp dims2[2]; // dimensions of X2 (2d)
    npy_intp dims3[2]; // x3 // dimentions of X3 (1d)
    npy_intp labels_dim[1]; // dimensions of labels (1d)
    labels_dim[0] = dimensions[1]; // labels_dim[0] = S
    for (int i = 0; i < 2; i++) {
        dims[i] = dimensions[i]; // dimensions[0] = R, dimensions[1] = S
        dims2[i] = dimensions2[i]; // dimensions2[0] = 5, dimensions2[1] = S
        dims3[i] = dimensions3[i]; // x3 // dimensions3[0] = 1, dimensions3[1] = S
    }
    unsigned seed = 42;
    std::mt19937 g (seed); 
    auto data = std::unique_ptr<Data>(new Data()); // positions, X, Y, X2 // X3
    
    // for each position in draft
    while (pileup_iter->has_next()) {
        auto column = pileup_iter->next(); // get the column
        long rpos = column->position;
        if (rpos < pileup_iter->start()) continue;
        if (rpos >= pileup_iter->end()) break;
        std::vector<segment> ins_segments;
        std::vector<segment> no_ins_reads;
        
        // if labels are provided by truth2draft: 
        std::string s;
        if (has_labels) {
            std::pair<pos_index_t, pos_index_t> index {rpos, 0};
            labels_info[index] = labels[index];
            pos_index_t ins_count = 1;
            index = std::make_pair(rpos, ins_count); // first insertion index: (rpos, 1)
            auto found = labels.find(index);
            while (found != labels.end()) { 
                // while the consecutive items can be found -> there are insertions in truth thats not present in draft
                char c = forward_int_to_char(labels[index]);
                // put it into the insertion segment s
                s.push_back(c); 
                ins_count++;
                index = std::make_pair(rpos, ins_count);
                found = labels.find(index);
            }
        }


        if (s.size() > 0) { // at this position, the draft is wrong, truth has a segment not present in draft
            ins_segments.emplace_back(std::move(s), LABEL_SEQ_ID); // LABEL_SEQ_ID = -1
        } // not gonna happen in inference mode, only training

        // now start from (rpos,0) again, check every rpos, so next one will be (rpos+1,0)
        std::pair<pos_index_t, pos_index_t> base_index(rpos, 0);
        uint32_t num_low_mq = 0;
        std::uint32_t ins_count = 0;

        // time to check each read at this position
        while(column->has_next()) { // a column is made up of bases from many reads at one position
            auto r = column->next(); // column -> next() goes down the column of bases, r is one of the reads in this column 
            if (r->is_refskip()) continue;
            if (r->mqual() < 10) {
                if (num_low_mq >= 100) continue;
                num_low_mq++; // reads with mapQ < 10 are low_mq reads, if a column has more than 100 low_mq reads then skip this column
            }
            if (align_bounds.find(r->query_id()) == align_bounds.end()) {
                align_bounds.emplace(r->query_id(), std::make_pair(r->ref_start(), r->ref_end()));
            }
            strand.emplace(r->query_id(), !r->rev()); // strand information of this read
            
            if (r->is_del()) {
                // DELETION
                auto pos_info = PosInfo(Bases::GAP, r->mqual());
                align_info[base_index].emplace(r->query_id(), pos_info);
                increment_base_count(base_index, pos_info);
            } else {
                // POSITION
                auto qbase = r->qbase(0);
                auto pos_info = PosInfo(qbase, r->mqual());
                align_info[base_index].emplace(r->query_id(), pos_info);
                increment_base_count(base_index, pos_info);
                // INSERTION
                if (r-> indel() > 0) {
                    std::string s;
                    s.reserve(r->indel());
                    for (int i = 1, n = r->indel(); i <= n; ++i) { // store the insertion segment
                        qbase = r->qbase(i);
                        s.push_back(base_to_char(qbase));
                    }
                    ins_segments.emplace_back(std::move(s), r->query_id(), r->mqual());
                    ins_count++;
                } else {
                    no_ins_reads.emplace_back("", r->query_id(), r->mqual());
                }
           }
        } // done with all the reads at one draft position

        pos_queue_push(base_index); // this base_index is (rpos,0)
        // push this index into pos_queue and check if this position is uncertain

        if (ins_count > 0) { // at this position, there exists at least one read with insertion
            //align_ins_longest(rpos, ins_segments, no_ins_reads); // use align_ins_center
            align_ins_center(rpos, ins_segments, no_ins_reads);
        }

        //BUILD FEATURE MATRIX
        // enter this while loop after adding one window (dimension[1] positions) to pos_queue
        while (pos_queue.size() >= dimensions[1]) {
            if (distances.empty())  {
                // if all are certain, remove 25% of positions in pos_queue 
                // only keep 75% of window size
                pos_queue_pop(pos_queue.size() - dimensions[1]/4 * 3);
                continue;
                
            } else if (distances.front() >= dimensions[1]) {
                uint16_t a = distances.front() - dimensions[1]/4 * 3;
                uint16_t b = pos_queue.size();
                pos_queue_pop(std::min(a, b));
                continue;
            }
            struct id_mq {
                uint32_t id;
                uint8_t mq;
                id_mq(uint32_t id, uint8_t mq) : id(id), mq(mq) {}
            };
            struct comp {
                bool operator() (const id_mq& lhs, const id_mq& rhs) {
                    return lhs.id < rhs.id; // sorts elements by their index
                }
            };

            std::set<id_mq, comp> valid_aligns;
            const auto it = pos_queue.begin();

            for (auto s = 0; s < dimensions[1]; s++) {    
                auto curr = it + s;
                
                for (auto& align : align_info[*curr]) {
                    if (align.second.base != Bases::UNKNOWN) { // the read does not have unknown base
                        valid_aligns.emplace(align.first, align.second.mq); // align.first = index, align.second = PosInfo, align.second.mq = mapq
                    }
                }
            }

           std::vector<id_mq> valid(valid_aligns.begin(), valid_aligns.end());
            
            uint32_t valid_size = valid.size();
            std::uint32_t sum = 0;
            for (auto x : valid) {
                sum += x.mq;
            }
            double avg = (double) sum / valid_size;
            struct alias {
                std::uint32_t idx1;
                std::uint32_t idx2;
                double share1;
                double share2;
                alias(std::uint32_t idx1, std::uint32_t idx2, double share1, double share2) : idx1(idx1), idx2(idx2), share1(share1), share2(share2) {}
            };
            std::vector<alias> done;
            done.reserve(valid_size);
            std::vector<std::pair<std::uint32_t, double>> large;
            large.reserve(valid_size);
            std::vector<std::pair<std::uint32_t, double>> small;
            small.reserve(valid_size);
            for (std::uint32_t i = 0; i < valid_size; i++) {
                double value = valid[i].mq;
                if (value == avg) {
                    done.emplace_back(i, i, avg, avg);
                } else if (value < avg) {
                    small.emplace_back(i, value);
                } else {
                    large.emplace_back(i, value);
                }
            }
            while (!large.empty() && !small.empty()) {
                auto large_one = large.back();
                auto small_one = small.back();
                large.pop_back();
                small.pop_back();
                double diff = avg -  small_one.second;
                done.emplace_back(large_one.first, small_one.first, diff, small_one.second);
                large_one.second -= diff;
                if (large_one.second == avg) {
                    done.emplace_back(large_one.first, large_one.first, large_one.second, large_one.second);
                } else if (large_one.second < avg) {
                    small.push_back(large_one);
                } else {
                    large.push_back(large_one);
                }
            }
            if (!large.empty()) {
                for (auto& x : large) {
                    done.emplace_back(x.first, x.first, avg, avg);            
                }
            } else if (!small.empty()) {
                for (auto& x: small) {
                    done.emplace_back(x.first, x.first, avg, avg);
                }
            }
            std::uniform_real_distribution<double> dist (0.0, (double) sum);


            auto X = PyArray_SimpleNew(2, dims, NPY_UINT8);
            auto X2 = PyArray_SimpleNew(2, dims2, NPY_UINT32);
            auto X3 = PyArray_SimpleNew(2, dims3, NPY_FLOAT);
            auto Y = PyArray_SimpleNew(1, labels_dim, NPY_UINT8);
            
            uint8_t* value_ptr;
            uint32_t *value_ptr_32;
            float* value_ptr_float;

            // First handle assembly (REF_ROWS)
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s; uint8_t value;

                if (curr->second != 0) value = ENCODED_BASES[Bases::GAP];
                else value = ENCODED_BASES[get_base(draft[curr->first])];

                for (int r = 0; r < REF_ROWS; r++) {
                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = value; // Forward strand - no +6
                }
            }
            //fill up X2 
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s;
                auto& pos_stats = stats_info[*curr];
	        if (curr->second == 0) {	
                    value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 0, s);
                    *value_ptr_32 = pos_stats.n_GAP;
		    value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 5, s);
		    *value_ptr_32 = 0;
		} else {
		    value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 0, s);
                    *value_ptr_32 = 0;
		    value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 5, s);
		    *value_ptr_32 = pos_stats.n_GAP;
		}
                value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 1, s);
                *value_ptr_32 = pos_stats.n_A;
                value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 2, s);
                *value_ptr_32 = pos_stats.n_C;
                value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 3, s);
                *value_ptr_32 = pos_stats.n_G;
                value_ptr_32 = (uint32_t*) PyArray_GETPTR2(X2, 4, s);
                *value_ptr_32 = pos_stats.n_T;

                // normalized_median_cov // x3
                value_ptr_float = (float*) PyArray_GETPTR2(X3, 0, s);
                
                if (curr->second == 0) {
                    *value_ptr_float = pos_stats.normalized_cov;
    
                } else {
                    auto& stats = stats_info[std::make_pair(curr->first, 0)];
                    *value_ptr_float = stats.normalized_cov;
                } // x3
                
            }

            // fill up X
            for (int r = REF_ROWS; r < dimensions[0]; r++) {

                uint8_t base;

                double value = dist(g);
                uint16_t idx = value/avg;
                auto& chosen = done[idx];
                uint32_t random_num;

                if (chosen.idx1 == chosen.idx2) {
                    random_num = chosen.idx1;
                } else {
                    if (value > idx * avg + chosen.share1) {
                        random_num = chosen.idx2;
                    } else {
                        random_num = chosen.idx1;
                    }
                }
                uint32_t query_id = valid[random_num].id;

                auto& fwd = strand[query_id];


                auto it = pos_queue.begin();
                for (auto s = 0; s < dimensions[1]; s++) {
                    auto curr = it + s;

                    auto pos_itr = align_info[*curr].find(query_id);
                    auto& bounds = align_bounds[query_id];
                    if (pos_itr == align_info[*curr].end()) {
                        if (curr->first < bounds.first || curr->first > bounds.second) {
                            base = ENCODED_BASES[Bases::UNKNOWN];
                        } else {
                            base = ENCODED_BASES[Bases::GAP];
                        }
                    } else {
                        base = ENCODED_BASES[pos_itr->second.base];
                    }

                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = fwd ? base : (base + 6);
                }
            }

            if (has_labels) {
                for (auto s = 0; s < dimensions[1]; s++) {
                    auto curr = it + s;
                    uint8_t value = labels_info[*curr];
                    value_ptr = (uint8_t*) PyArray_GETPTR1(Y, s);
                    *value_ptr = value;
                }
            }
            data->X.push_back(X);
            data->X2.push_back(X2);
            data->Y.push_back(Y);
            data->X3.push_back(X3); // x3
            data->positions.emplace_back(pos_queue.begin(), pos_queue.begin() + dimensions[1]);
            pos_queue_pop(WINDOW);
        }
    }
    return data;
}
