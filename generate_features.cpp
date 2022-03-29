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
// A char has 8 bits, so there are 256 (0-255) values. 
// The first 128 (0-127) values can be corresponded to the ascii table 
// (eg. 43th char is * in ascii table so the 43th int in this map is 4)
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

// ref here is draft. FeatureGenerator class has a constant attribute [const char* draft] that cant be changed after construction
// here after the construction there is a ': draft(ref)'. this is initialization list in C++, setting draft = ref.
// see line 90 of generate_features.h and (https://www.cprogramming.com/tutorial/initialization-lists-c++.html)
// dict is --Y for training features. It's a dictinary of {(pos, ins_pos): label}
FeatureGenerator::FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict): draft(ref) {
    
    bam = readBAM(filename);
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
        PyErr_Format(PyExc_TypeError, // dict should be a dictionary
                "Argument \"dict\" to %s must be dict not \"%s\"", 
                __FUNCTION__, Py_TYPE(dict)->tp_name);	       
    }
    while (PyDict_Next(dict, &pos, &key, &value)) {
        if (! PyTuple_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "A key of dict is not a tuple!"); // key should be a tuple
            labels.clear();
        } 
        if (PyTuple_Size(key) != static_cast<Py_ssize_t>(2)) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict is not a pair!"); // key should be a pair
            labels.clear();
        }
        PyObject *pair_item0 = PyTuple_GetItem(key, 0);
        PyObject *pair_item1 = PyTuple_GetItem(key, 1);
        if ((!PyLong_Check(pair_item0)) || (!PyLong_Check(pair_item1))) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict does contain two longs!"); // means it should contain 2 longs
            labels.clear();
        }
        if (! PyLong_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "A value of dict is not of long type!"); // means the value should be of long type
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

void FeatureGenerator::add_bq_sample(std::pair<pos_index_t, pos_index_t>& index, float bq) {
    auto& info = stats_info[index];
    info.avg_bq = info.avg_bq + (bq - info.avg_bq)/ ++info.n_bq; // calculate new average
}

void FeatureGenerator::add_mq_sample(std::pair<pos_index_t, pos_index_t>& index, uint8_t mq) {
    auto& info = stats_info[index];
    info.avg_mq = info.avg_mq + (float) (mq - info.avg_mq)/ ++info.n_mq; // calculate new average
}

void FeatureGenerator::increment_base_count(std::pair<pos_index_t, pos_index_t>& index, Bases b) {
    auto& s = stats_info[index]; // create a pointer to be used to update stats_info at index
    s.n_total++; // total number of reads at this position specified by index
    Bases draft_base = Bases::GAP; // assume this is an insertion in the read, so draft has gap at this position
    // index here is a pair of indices: index.first = draft_base index, index.second = insertion index
    if (index.second == 0) { // if this base is not an insertion
        draft_base = char_to_base(draft[index.first]);
    }
    bool diff = draft_base != b;
    switch(b) {
        case Bases::A:
            s.n_A++;
            if (diff && s.n_A > s.largest_diff) {
               s.largest_diff = s.n_A;
            }
            break;
        case Bases::C:
            s.n_C++;
            if (diff && s.n_C > s.largest_diff) {
                s.largest_diff = s.n_C;
            }
            break;
        case Bases::G:
            s.n_G++;
            if (diff && s.n_G > s.largest_diff) {
                s.largest_diff = s.n_G;
            }
            break;
        case Bases::T:
            s.n_T++;
            if (diff && s.n_T > s.largest_diff) {
                s.largest_diff = s.n_T;
            }
            break;
        case Bases::GAP:
            s.n_GAP++;
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
    //std::cout << "counter in " << counter << std::endl;
    bool is_uncertain = false;
    auto& s = stats_info[index];
    //char draft_base = draft[index.first];
    uint16_t num_total = s.n_total;
    //uint16_t num_same; //same as draft
    if (index.second != 0) { // if this is an inserted base
        //num_same = s.n_GAP;
        //uint16_t num_not_gap = num_total - s.n_GAP;
        //std::cout << "index " << index.first << ", " << index.second << " " << (float) s.largest_diff/num_total << std::endl;
        if ((float) s.largest_diff/ num_total < NON_GAP_THRESHOLD) { // NON_GAP_THRESHOLD = 0.01
            // the frequency of the most common alternative base is represented by less than 1% of reads
            // so this position is thrown away, index not added to pos_queue
            // therefore the result feature's insertion index might not be continuous
            // e.g. if (45,3) does not have enough supporting reads but (45,4) does then (45,3) is skipped
            //num_filter++;
            return;
        }
    } 
    if ((float) s.largest_diff/num_total >= UNCERTAIN_POSITION_THRESHOLD) { // UNCERTAIN_POSITION_THRESHOLD = 0.15
        is_uncertain = true;
        // if the frequency of the most common alternative base is represented by more than 15% of reads
        // this base is uncertain (likely to be wrong)
    } 
   // std::cout << index.first << ", " << index.second << " : " << (float) s.largest_diff/num_total << std::endl;
    if (is_uncertain) {
        pos_queue.push_back(index); // pos_queue is a deque object, push_back(val) puts val at the end of the queue
        distances.push(counter); // distances records all numbers of certain positions between uncertain positions
        counter = 0;

    } else {
        pos_queue.push_back(index);
        counter++; // counter counts the number of certain positions between any 2 uncertain positions
    }
    //std::cout << "counter out " << counter << std::endl;
}

// should have at least num elements in the deque container
void FeatureGenerator::pos_queue_pop(uint16_t num) {
    
    pos_queue.erase(pos_queue.begin(), pos_queue.begin() + num); // iterator erase (iterator first, iterator last) -> Removes from the deque container a range of elements [first,last)
    // erase the first num elements from pos_queue
    while (distances.size() > 0 && num >=(distances.front()+1)) { // queue.size() = number of elements in a queue
        // if there are elements (number of certain positions between uncertain positions)
        // and the element being checked is smaller than num
        num -= distances.front() + 1; // remove the certain positions from num
        distances.pop();
    }
    if (distances.empty()) { // if all elements in distances are popped
        counter -= num; // what is this counter?
    } else {
        // since distances is not empty, the second condition of the while loop must be false (i.e. num <(distances.front()+1))
        distances.front() -= num;
    }

}

// base_index = rpos, segments = ins_segments (includes one truth segment), star_pos_index = best segment index, no_ins_reads = no_ins_reads
void FeatureGenerator::align_center_star(pos_index_t base_index, std::vector<segment>& segments, int star_pos_index, 
        std::vector<segment>& no_ins_reads) {

    std::vector<segment*> non_label_seqs; // create a pointer named non_label_seqs
    non_label_seqs.reserve(segments.size());  // pointing to a vector of segments with the same number of elements as the number of insertion segments

    segment star = segments[star_pos_index]; // get the best insertion segment

    std::unordered_map<uint32_t, PosInfo> star_positions[star.sequence.size()];
    // declare an array of x unordered_maps, x = star.sequence.size() = number of bases in the best insertion segment

    std::vector<std::unordered_map<uint32_t, PosInfo>> ins_positions[star.sequence.size()+1];
    // declare a vector of x unordered_maps, x = star.sequence.size() = number of bases in the best insertion segment + 1

    // stores labels of the star sequence. By default all gaps
    uint8_t star_positions_labels[star.sequence.size()];  
    for (unsigned int i = 0; i < star.sequence.size(); i ++) {
        star_positions_labels[i] = ENCODED_BASES[Bases::GAP];
    }
    
    // stores the labels aligned to the positions where the star has gaps after aligning
    std::vector<uint8_t> ins_positions_labels[star.sequence.size()+1];    
    
    int total_ins_pos = 0;
    for (auto& s : segments) {
        if (s.index != LABEL_SEQ_ID) non_label_seqs.push_back(&s); // if it is not ground truth (therefore not label)
        if (s.index != star.index) { // if it is not the best insertion segment (might be ground truth)
            // calculate the edit distance between this segment and the best insertion segment (align this segment to the best insertion segment)
            EdlibAlignResult result = edlibAlign(s.sequence.c_str(), s.sequence.size(), star.sequence.c_str(),
                    star.sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_PATH, NULL, 0));
/* EdlibAlignResult edlibAlign(const char* const queryOriginal, const int queryLength,
                            const char* const targetOriginal, const int targetLength,
                            const EdlibAlignConfig config) {
                    EdlibAlignResult result;
                    result.status = EDLIB_STATUS_OK;
                    result.editDistance = -1;
                    result.endLocations = result.startLocations = NULL;
                    result.numLocations = 0;
                    result.alignment = NULL;
                    result.alignmentLength = 0;
                    result.alphabetLength = 0; */

            //why use the previous one?
            int ref_pos = -1; // pointing to before next to read ref base?? previous base?
            int query_pos = -1; // pointing to before next to read query base?? previous base?
            unsigned int ins_index = 0; // index of next insertion, 0-based
            char char_at_pos;
            Bases base_at_pos;
            for (int i = 0; i < result.alignmentLength; i++) { // for each aligned pos in the resulting alignment
                switch (result.alignment[i]) {
                    case 0: // match
                        // if it is a match: query_pos increases by 1, ref_pos increases by 1
                        ins_index = 0;	      
                        char_at_pos = s.sequence[++query_pos]; // s is the segment being aligned to star
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++; // ref here should be the star segment
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth, add it to the label
                            star_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else { // else it is just an insertion segment (not star)
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos)); 
                            // insertion segment index = read_id, base
                        }
                        break;
                    case 1: // insertion [has base on insertion segment, del on star]
                        // ref_pos on star is not incremented, query_pos increases by 1, ins_index increases by 1, total_ins_pos increases by 1
                        char_at_pos = s.sequence[++query_pos]; // get the segment's base
                        base_at_pos = char_to_base(char_at_pos);
                        if (ins_positions[ref_pos+1].size() < ins_index + 1) { // if not enough maps to record bases in that position
                            ins_positions[ref_pos+1].push_back(std::unordered_map<uint32_t, PosInfo>{});
                            ins_positions_labels[ref_pos+1].push_back(ENCODED_BASES[Bases::GAP]);
                            total_ins_pos++;
                        }
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth
                            ins_positions_labels[ref_pos+1][ins_index] = char_to_forward_int(char_at_pos);
                        } else {
                            ins_positions[ref_pos+1][ins_index].emplace(s.index, PosInfo(base_at_pos));
                        }
                        ins_index++;
                        break;
                    case 2: // deletion has base on star, del on insertion segment]
                        // ref_pos on star increases by 1, ins_index reset to 0
                        ins_index = 0;
                        ref_pos++;
                        break;
                    case 3: // mismatch
                        // query_pos increases by 1, ref_pos on star increases by 1, ins_index reset to 0, 
                        ins_index = 0; 
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth
                            star_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else {
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos));
                        }
                        break;
                    default:
                        std::cout << "Uknown alignment result!\n";
                }
            }
            edlibFreeAlignResult(result); // cleaning
        } else { // if it is star, also might be the ground truth
            // record bases on the star
            for (unsigned int i = 0; i < s.sequence.size(); i++) {
                const char char_at_pos = s.sequence[i];
                Bases base_at_pos = char_to_base(char_at_pos);               
                if (s.index == LABEL_SEQ_ID) {
                    star_positions_labels[i] = char_to_forward_int(char_at_pos);
                } else {
                    star_positions[i].emplace(s.index, PosInfo(base_at_pos));
                }
            }
        }
    }
  
    uint16_t pos_counts[non_label_seqs.size()] = {0}; 
    // create an array called pos_counts with non_label_seqs.size() elements, initialize all elements to 0
    // [not the same as the reserved size of segments.size(), non_label_seqs does not have ground truth]
    // vector.size() returns actual size, not reserved size (eg. std::vector<int> vec1; vec1.reserve(5); vec1.push_back(1); std:cout << vec1.size(); will output 1 instead of 5)

    pos_index_t count = 1;
    // correspond to positions before the first position of star before aligning
    for (unsigned int i = 0; i < ins_positions[0].size(); i++) {
        auto& map = ins_positions[0][i];
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
        
        count++;
        
        for (uint16_t k = 0; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (map.find(s->index) == map.end()) {
                map.emplace(s->index, PosInfo(Bases::GAP));                
                add_bq_sample(index, ( (float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1]) /2 );
                add_mq_sample(index, s->mq);

            } else {
               
                add_bq_sample(index, s->bqs[++pos_counts[k]]);
                add_mq_sample(index, s->mq);
            }
        }
        for (auto& s: no_ins_reads) { 
            map.emplace(s.index, PosInfo(Bases::GAP));
            add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
            add_mq_sample(index, s.mq);

             
        }
        for (auto& pair: map) {
            auto b = pair.second.base;
            increment_base_count(index, b);  

        }

        align_info[index] = map;
        if (has_labels) {
            labels_info[index] = ins_positions_labels[0][i]; 
        }
        pos_queue_push(index);
    }

    // correspond to positions on star before aligning, and insertions after them(the inner loop)
    for (unsigned int i = 0; i < star.sequence.size(); i++) {
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);

        count++;
        
        for (uint16_t k = 0 ; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (star_positions[i].find(s->index) == star_positions[i].end()) {
                star_positions[i].emplace(s->index, PosInfo(Bases::GAP));
                add_bq_sample(index, ((float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1])/2);
                add_mq_sample(index, s->mq);

            } else {
                add_bq_sample(index, s->bqs[++pos_counts[k]]);
                add_mq_sample(index, s->mq);

            }
        }
        for (auto& s: no_ins_reads) {
            star_positions[i].emplace(s.index, PosInfo(Bases::GAP));
            add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
            add_mq_sample(index, s.mq);
        }
        for (auto& pair: star_positions[i]) {
            auto b = pair.second.base;
            increment_base_count(index, b);  

        }

        pos_queue_push(index); 
        align_info[index] = star_positions[i];
        if (has_labels) {
            labels_info[index] = star_positions_labels[i];
        }

        for (unsigned int j = 0; j < ins_positions[i+1].size(); j++) {
            auto& map = ins_positions[i+1][j];
            auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
          
            count++;
            
            for (uint16_t k = 0; k< non_label_seqs.size(); k++) {
                auto& s = non_label_seqs[k];
                if (map.find(s->index) == map.end()) {
                    map.emplace(s->index, PosInfo(Bases::GAP));
                   
                    add_bq_sample(index, ((float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1])/2  );
                    add_mq_sample(index, s->mq);

                } else {
                   
                    add_bq_sample(index, s->bqs[++pos_counts[k]]);
                    add_mq_sample(index, s->mq);
                }
            }
            for (auto& s: no_ins_reads) {
                map.emplace(s.index, PosInfo(Bases::GAP));	
                add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
                add_mq_sample(index, s.mq);

            }
            for (auto& pair: map) {
                auto b = pair.second.base;
                increment_base_count(index, b);  

            }

            align_info[index] = map;
            if (has_labels) {
                labels_info[index] = ins_positions_labels[i+1][j];
            }
            pos_queue_push(index);
        }
    } 
}

int FeatureGenerator::find_center(std::vector<segment>& segments) { // segments here are insertion segments
    int dists[segments.size()]{0}; // creates an array named dists with segments.size() number of elements, initialize each element to 0
    // creates dists with x 0s, x = number of insertion segments
    for (unsigned int i = 0; i < segments.size(); i++) { // for each insertion segment
        for (unsigned int j = i + 1; j < segments.size(); j++) { // compare with the insertion segments after the i-th one

            // string::c_str() Returns a pointer to an array that contains a null-terminated sequence of characters 
            // (i.e., a C-string) representing the current value of the string object.
            // the array will be an array of char: [char1 char2 char3 null]
            EdlibAlignResult result = edlibAlign(segments[i].sequence.c_str(), segments[i].sequence.size(), segments[j].sequence.c_str(),
                    segments[j].sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, NULL, 0));
            dists[i] += result.editDistance; // add this distance to i
            dists[j] += result.editDistance; // add this distance to j
            // comparing i-th segment to all other segments once
            // eg if there are 5 elements: 0-th 1-th 2-th 3-th 4-th, dists will have 5 entries
            // then comparison is done between 0-1, 0-2, 0-3, 0-4
            //                                 1-2, 1-3, 1-4
            //                                 2-3, 2-4
            //                                 3-4.      for each comparison, result is saved to both segments' entries
            // edit distance measures the similarity between segments (the smaller the more similar)
            /*// Example program
                #include <iostream>
                #include <string>

                int main() {
                    int dists[5] {0};
                    int segments[5] {2, 4, 1, 7, 9};
                    for (int i = 0; i < 5; i++) { // for each insertion segment
                        for (int j = i + 1; j < 5; j++) { // compare with all other insertion segments
                        dists[i] += segments[j] - segments[i];
                        dists[j] += segments[j] - segments[i];
                        std::cout << "\ni, j, j-i distance \n";
                        std::cout << i << " " << j << " " << segments[j] - segments[i];
                        std::cout << "\n";

                        for (int k = 0; k < 5; k++) {
                            std::cout << "dists " << dists[k] << " ";
                            }
                        }
                    }
                }*/
                /*[output of example code]
                i, j, j-i distance 
                0 1 2
                dists 2 dists 2 dists 0 dists 0 dists 0 
                i, j, j-i distance 
                0 2 -1
                dists 1 dists 2 dists -1 dists 0 dists 0 
                i, j, j-i distance 
                0 3 5
                dists 6 dists 2 dists -1 dists 5 dists 0 
                i, j, j-i distance 
                0 4 7
                dists 13 dists 2 dists -1 dists 5 dists 7 
                i, j, j-i distance 
                1 2 -3
                dists 13 dists -1 dists -4 dists 5 dists 7 
                i, j, j-i distance 
                1 3 3
                dists 13 dists 2 dists -4 dists 8 dists 7 
                i, j, j-i distance 
                1 4 5
                dists 13 dists 7 dists -4 dists 8 dists 12 
                i, j, j-i distance 
                2 3 6
                dists 13 dists 7 dists 2 dists 14 dists 12 
                i, j, j-i distance 
                2 4 8
                dists 13 dists 7 dists 10 dists 14 dists 20 
                i, j, j-i distance 
                3 4 2
                dists 13 dists 7 dists 10 dists 16 dists 22*/

                // in this exaxmple, the distance is difference between the 2 segments
                // this function is kind of like listing the segments out to become 1,2,4,7,9
                // the middle element is the best since its difference to all other segments is the smallest
                // since it is in the middle
                // now change the distance to edit distance, find center hence finds the alignment whose [sum of
                // edit distances to all other segments] is minimal, therefore the best segment
                // dists is the list of [sum of edit distance of each element with all others]
                // eg 0-th element in dists = sum of edit distance between the 0-th segment and all other segments
            edlibFreeAlignResult(result); // cleaning
        }
    }
    int best_pos_index = 0;
    for (unsigned int i = 0; i < segments.size(); i++) {
        if (dists[i] < dists[best_pos_index]) { // get the smallest sum of edit distance, the "center"
            best_pos_index = i;
        }
    }
    return best_pos_index;    

}

int FeatureGenerator::find_longest(std::vector<segment>& segments) { // finds the longest segment
    int best_index = 0;
    int highest_len = 0;
    for (int i = 0; i < segments.size(); i++) {
        int len = segments[i].sequence.size();
        if (len > highest_len) {
            best_index = i;
            highest_len = len;
        }
    }
    return best_index;
}


// base_index = rpos, ins_segments = ins_segments, no_ins_reads = no_ins_reads
void FeatureGenerator::align_ins_longest_star(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int longest_index = find_longest(ins_segments); // the index of the longest read in ins_segments
    align_center_star(base_index, ins_segments, longest_index, no_ins_reads);

}

void FeatureGenerator::align_ins_center_star(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int center_index = find_center(ins_segments);
    align_center_star(base_index, ins_segments, center_index, no_ins_reads);

}

std::unique_ptr<Data> FeatureGenerator::generate_features() {   
    npy_intp dims[2]; // dimensions of X1 (2d)
    npy_intp dims2[2]; // dimensions of X2 (2d)
    npy_intp labels_dim[1]; // dimensions of labels (1d)
    srand(49);
    labels_dim[0] = dimensions[1]; // labels_dim[0] = S
    for (int i = 0; i < 2; i++) {
        dims[i] = dimensions[i]; // dimensions[0] = R, dimensions[1] = S
        dims2[i] = dimensions2[i]; // dimensions2[0] = 5, dimensions2[1] = S
    }
 
    auto data = std::unique_ptr<Data>(new Data()); // positions, X, Y, X2
    
    // for each position in draft
    while (pileup_iter->has_next()) {
        auto column = pileup_iter->next(); // get the column
        long rpos = column->position;
        if (rpos < pileup_iter->start()) continue;
        if (rpos >= pileup_iter->end()) break;
        std::vector<segment> ins_segments;
        std::vector<segment> no_ins_reads;
        
        // if labels are provided by truth2draft: 
        // put the insertion segment according to truth seq into s
        std::string s;
        if (has_labels) {
            std::pair<pos_index_t, pos_index_t> index {rpos, 0}; // initialize a pair called index to have values(rpos, 0)
            labels_info[index] = labels[index]; // store truth base
            pos_index_t ins_count = 1; // check if there is any insertion (bases present in truth but not in draft)
            index = std::make_pair(rpos, ins_count); // first insertion index: (rpos, 1)
            auto found = labels.find(index); // labels is an unordered_map, use .find(key) to get value
            // using unordered_map.find(key), if item is found, return the item, if item is not found, return unordered_map::end
            while (found != labels.end()) { // while the consecutive items can be found -> there are insertions in truth thats not present in draft
                char c = forward_int_to_char(labels[index]); // convert this base to char
                s.push_back(c); // put it into the insertion segment s -> the ground truth segment that is not present in draft
                ins_count++; // increment insertion count
                index = std::make_pair(rpos, ins_count); // index = next insertion index
                found = labels.find(index); // try to find the next inserted base
            }
        }


        if (s.size() > 0) { // the draft is wrong, truth has a segment not present in draft
            // put this insertion segment to the vector of insertion segments at this position and give it an ID of -1
            ins_segments.emplace_back(std::move(s), LABEL_SEQ_ID); // LABEL_SEQ_ID = -1
        } // not gonna happen in inference mode, only training

        // now start from (rpos,0) again
        std::pair<pos_index_t, pos_index_t> base_index(rpos, 0);

        // time to check each read at this position
        while(column->has_next()) { // a column is made up of bases from many reads at one position
            auto r = column->next(); // column -> next() goes down the column of bases, so here r is one of the reads in this column 
            if (r->is_refskip()) continue; // cigar = N (represents introns which are removed), meaning that this position has no ACGT base
            if (align_bounds.find(r->query_id()) == align_bounds.end()) { // if this read is not found in align_bounds
                align_bounds.emplace(r->query_id(), std::make_pair(r->ref_start(), r->ref_end())); // add it in
            } // align_bounds {query_id:(ref_start, ref_end))}
            strand.emplace(r->query_id(), !r->rev()); // strand information of this read
            
            if (r->is_del()) {
                // DELETION
                align_info[base_index].emplace(r->query_id(), PosInfo(Bases::GAP)); // align_info is a map defined in generate_features.h
                // it maps an index (key A) to another map B (value A), which maps the query id (key B) to the base (value B).
                increment_base_count(base_index, Bases::GAP);
                add_mq_sample(base_index, r->mqual());
                add_bq_sample(base_index, ((float) r->qqual(-1) + r->qqual(0)) /2); // qqual(offset) is defined in models.h
                // if offset = 0 for del, it will get the qual of next base (NOT returning 10)
                // base quality of deletion: (quality of the previous base + quality of the next base)/2
            } else {
                // POSITION
                auto qbase = r->qbase(0); // get read base (query base)
                align_info[base_index].emplace(r->query_id(), PosInfo(qbase));
                increment_base_count(base_index, qbase);
                add_mq_sample(base_index, r->mqual());
                add_bq_sample(base_index,  r->qqual(0));
                // INSERTION
                if (r-> indel() > 0) {
                    std::string s; // declare an insertion string
                    s.reserve(r->indel()); // of size of insertion length
                    std::vector<uint8_t> segment_bqs; // declare a vector named segment base qualities
                    segment_bqs.push_back(r->qqual(0)); // put the base quality of this base into the segment_bqs vector
                    // here qqual(offset = 0) gets its own quality because it is not a deletion
                    for (int i = 1, n = r->indel(); i <= n; ++i) { // store the insertion segment and all base qualities
                        qbase = r->qbase(i); // get inserted base
                        s.push_back(base_to_char(qbase)); // store inserted base into insertion string
                        segment_bqs.push_back(r->qqual(i)); // store inserted base quality into segment_bqs
                    }
                    segment_bqs.push_back(r->qqual(r->indel() + 1)); // store the quality of the next base after the insertion segment
                    // take the next base quality because if there's a gap after the insertion, the gap's quality needs both the left and right bases to be calculated
                    ins_segments.emplace_back(std::move(s), r->query_id(), r->mqual(), std::move(segment_bqs));
                } else {
                    no_ins_reads.emplace_back("", r->query_id(), r->mqual(),std::initializer_list<uint8_t>{r->qqual(0), r->qqual(1)});
                }
           }
        } // done with all the reads at one position

        pos_queue_push(base_index); // this base_index is (rpos,0)

        if (ins_segments.size() > 0) { // at this position, there exists at least one read with insertion
            //align_ins_longest_star(rpos, ins_segments, no_ins_reads); // dehui says change this to align_ins_center_star
            align_ins_center_star(rpos, ins_segments, no_ins_reads);
            // at position rpos, provide the function with reads: those with insertions and those without
            // reads contain information: (insertion segment if any), query_id, mapping quality, base quality (of itself and the next base's)
        }

        //BUILD FEATURE MATRIX
        while (pos_queue.size() >= dimensions[1]) {
            if (distances.empty())  {
                // if all are certain, remove most of positions in pos_queue 
                // only keep 75% of window size
                pos_queue_pop(pos_queue.size() - dimensions[1]/4 * 3);
                continue;
                
            } else if (distances.front() >= dimensions[1]) { // the first certain segment is larger than window size
                uint16_t a = distances.front() - dimensions[1]/4 * 3; // certain segment length - 75% window size
                uint16_t b = pos_queue.size();
                pos_queue_pop(std::min(a, b));
                continue;
            } 

            std::set<uint32_t> valid_aligns;
            const auto it = pos_queue.begin();

            for (auto s = 0; s < dimensions[1]; s++) {    
                auto curr = it + s;
                
                for (auto& align : align_info[*curr]) {
                    if (align.second.base != Bases::UNKNOWN) {
                        valid_aligns.emplace(align.first);
                    }
                }
            }
            std::vector<uint32_t> valid(valid_aligns.begin(), valid_aligns.end());
           
            int valid_size = valid.size();
            // when number of valid reads < threshold, then add draft base as reads for x times (comment this block off later)
            //int num_draft = REF_ROWS;
            //if(valid_size < 3) num_draft = 15;

            auto X = PyArray_SimpleNew(2, dims, NPY_UINT8);
            auto X2 = PyArray_SimpleNew(2, dims2, NPY_UINT16);
            auto Y = PyArray_SimpleNew(1, labels_dim, NPY_UINT8);
            
            uint8_t* value_ptr;
            uint16_t *value_ptr_16;

            // First handle assembly (REF_ROWS)
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s; uint8_t value;

                if (curr->second != 0) value = ENCODED_BASES[Bases::GAP];
                else value = ENCODED_BASES[get_base(draft[curr->first])];

                // change this part back to 'for (int r = 0; r < REF_ROWS; r++)' later
                for (int r = 0; r < REF_ROWS; r++) {
                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = value; // Forward strand - no +6
                }
            }
            //fill up X2 
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s;
                auto pos_stats = stats_info[*curr];

                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 0, s);
                *value_ptr_16 = pos_stats.n_GAP;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 1, s);
                *value_ptr_16 = pos_stats.n_A;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 2, s);
                *value_ptr_16 = pos_stats.n_C;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 3, s);
                *value_ptr_16 = pos_stats.n_G;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 4, s);
                *value_ptr_16 = pos_stats.n_T;
                // value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 5, s);
                //*value_ptr_16 = static_cast<uint16_t>(pos_stats.avg_mq);
                //value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 6, s);
                //*value_ptr_16 = static_cast<uint16_t>(pos_stats.avg_bq);
                
            }

            for (int r = REF_ROWS; r < dimensions[0]; r++) {

                uint8_t base;
                auto random_n = rand();
                auto random_num = random_n  % valid_size;

                uint32_t query_id = valid[random_num];

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
            data->positions.emplace_back(pos_queue.begin(), pos_queue.begin() + dimensions[1]);
            for (auto it = pos_queue.begin(), end = pos_queue.begin() + WINDOW; it != end; ++it) {
                align_info.erase(*it);
            }
            pos_queue_pop(WINDOW);
        }
    } 
    return data;
}


