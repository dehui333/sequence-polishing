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

FeatureGenerator::FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict, bool inference_mode): draft(ref), inference_mode(inference_mode) {
    
    bam = readBAM(filename);
    pileup_iter = bam->pileup(region);
    convert_py_labels_dict(dict);
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
        long pair_item0_c = PyLong_AsLong(pair_item0);
        long pair_item1_c = PyLong_AsLong(pair_item1);
        uint8_t value_c = PyLong_AsLong(value);
        if (PyErr_Occurred()) {
            labels.clear();
        }
        labels.emplace(std::make_pair(pair_item0_c, pair_item1_c), value_c);

    }
}


Bases FeatureGenerator::char_to_base(char c) {
    switch (c) {
        case 'A':
            return Bases::A;
        case 'C':
            return Bases::C;
        case 'G':
            return Bases::G;
        case 'T':
            return Bases::T;
        case '*':
            return Bases::GAP;
        case 'N':
            std::cout << "Unknown base!" << std::endl;
            return Bases::UNKNOWN;
        default:
            std::cout << "Non N unknown!" << std::endl;
            return Bases::UNKNOWN;
    }
}

char FeatureGenerator::base_to_char(Bases b) {
    switch (b) {
        case Bases::A:
            return 'A';
        case Bases::C:
            return 'C';
        case Bases::G:
            return 'G';
        case Bases::T:
            return 'T';
        case Bases::GAP:
            std::cout << "Gap!" << std::endl;
            return '*';
        default:
            std::cout << "Unknown base!" << std::endl;
            return 'N';
    }
}

char FeatureGenerator::int_to_char(uint8_t i) {
    switch (i) {
        case 0:
            return 'A';
        case 1:
            return 'C';
        case 2:
            return 'G';
        case 3:
            return 'T';
        case 4:
            return '*';
        default:
            return 'N';
    }
}

Bases FeatureGenerator::int_to_base(uint8_t i) {

    switch (i) {
        case 0:
            return Bases::A;
        case 1:
            return Bases::C;
        case 2:
            return Bases::G;
        case 3:
            return Bases::T;
        case 4:
            return Bases::GAP;
        default:
            return Bases::UNKNOWN;
    }
}

uint8_t FeatureGenerator::char_to_int(char c) {
    switch (c) {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        case '*':
            return 4;
        default:
            return 5;
    }
}


void FeatureGenerator::align_center_star(long base_index, std::vector<segment>& segments, int star_index, 
        unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads) {
    std::vector<uint32_t> seq_indices(segments.size());
    segment star = segments[star_index];
    std::unordered_map<uint32_t, PosInfo> star_positions[star.len]; //stores bases aligned to original positions on the star
    std::vector<std::unordered_map<uint32_t, PosInfo>> ins_positions[star.len+1]; //stores bases aligned to gaps inserted into the original seq of star
    uint8_t star_positions_labels[star.len];
    for (int i = 0; i < star.len; i ++) {
        star_positions_labels[i] = ENCODED_BASES[Bases::GAP];
    }
    std::vector<uint8_t> ins_positions_labels[star.len+1];
    int total_ins_pos = 0;
    for (auto s: segments) {
        //std::cout << s.index << " " << s.sequence << std::endl;
        if (s.index != -1) seq_indices.push_back(s.index);
        if (s.index != star.index) {
            EdlibAlignResult result = edlibAlign(s.sequence.c_str(), s.len, star.sequence.c_str(),
                    star.len, edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_PATH, NULL, 0));
            int ref_pos = -1; // pointing to before next to read ref base
            int query_pos = -1; // pointing to before next to read query base 
            unsigned int ins_index = 0; // index of next insertion, 0-based
            char char_at_pos;
            Bases base_at_pos;
            for (int i = 0; i < result.alignmentLength; i++) {
                switch (result.alignment[i]) {
                    case 0: // match
                        ins_index = 0;	      
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == -1) {
                            star_positions_labels[ref_pos] = char_to_int(char_at_pos);
                        } else {
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos));
                        }
                        break;
                    case 1: // insertion

                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        if (ins_positions[ref_pos+1].size() < ins_index + 1) { // if not enough maps to record bases in that position
                            ins_positions[ref_pos+1].push_back(std::unordered_map<uint32_t, PosInfo>{});
                            ins_positions_labels[ref_pos+1].push_back(ENCODED_BASES[Bases::GAP]);
                            total_ins_pos++;
                        }
                        if (s.index == -1) {
                            ins_positions_labels[ref_pos+1][ins_index] = char_to_int(char_at_pos);
                        } else {
                            ins_positions[ref_pos+1][ins_index].emplace(s.index, PosInfo(base_at_pos));
                        }
                        ins_index++;
                        break;
                    case 2: // deletion

                        ins_index = 0;
                        ref_pos++;
                        break;
                    case 3: // mismatch

                        ins_index = 0; 
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == -1) {
                            star_positions_labels[ref_pos] = char_to_int(char_at_pos);
                        } else {
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos));
                        }
                        break;
                    default:
                        std::cout << "Uknown alignment result!\n";


                }
            }

            edlibFreeAlignResult(result);

        } else {
            // record bases on the star    
            for (int i = 0; i < s.len; i++) {
                const char char_at_pos = s.sequence[i];
                Bases base_at_pos = char_to_base(char_at_pos);               
                if (s.index == -1) {
                    star_positions_labels[i] = char_to_int(char_at_pos);
                } else {
                    star_positions[i].emplace(s.index, PosInfo(base_at_pos));
                }
            }

        }


    }

    long count = 1;
    for (unsigned int i = 0; i < ins_positions[0].size(); i++) {
        auto& map = ins_positions[0][i];
        auto index = std::pair<long, long>(base_index, count);
        if (map.size() >= threshold_num) {
            pos_queue.emplace_back(base_index, count);
            count++;
        }

        for (auto& id: seq_indices) {
            if (map.find(id) == map.end()) {
                map.emplace(id, PosInfo(Bases::GAP));	
            }	  
        }
        for (auto& id: no_ins_reads) { 
            map.emplace(id, PosInfo(Bases::GAP));	  
            
        }
        for (auto& pair: map) {
            auto b = pair.second.base;
            switch(b) {
                case Bases::A:
                    stats_info[index].n_A++;
                    break;
                case Bases::C:
                    stats_info[index].n_C++;
                    break;
                case Bases::G:
                    stats_info[index].n_G++;
                    break;
                case Bases::T:
                    stats_info[index].n_T++;
                    break;
                case Bases::GAP:
                    stats_info[index].n_del++;
                    break;
                default:
                    std::cout << "SHOULD NOT GET HERE" << std::endl;        
            }   

        }

        align_info[index] = map;
        labels_info[index] = ins_positions_labels[0][i];
        
    }
    for (int i = 0; i < star.len; i++) {
        auto index = std::pair<long, long>(base_index, count);

        if (star_positions[i].size() >= threshold_num) {
            pos_queue.emplace_back(base_index, count);
            count++;
        }

        for (auto& id: seq_indices) {
            if (star_positions[i].find(id) == star_positions[i].end()) {
                star_positions[i].emplace(id, PosInfo(Bases::GAP));	
            }	  
        }
        for (auto& id: no_ins_reads) {
            star_positions[i].emplace(id, PosInfo(Bases::GAP));	  
        }
        for (auto& pair: star_positions[i]) {
            auto b = pair.second.base;
            switch(b) {
                case Bases::A:
                    stats_info[index].n_A++;
                    break;
                case Bases::C:
                    stats_info[index].n_C++;
                    break;
                case Bases::G:
                    stats_info[index].n_G++;
                    break;
                case Bases::T:
                    stats_info[index].n_T++;
                    break;
                case Bases::GAP:
                    stats_info[index].n_del++;
                    break;
                default:
                    std::cout << "SHOULD NOT GET HERE" << std::endl;        
            }   

        }

        align_info[index] = star_positions[i];
        labels_info[index] = star_positions_labels[i]; 

        for (unsigned int j = 0; j < ins_positions[i+1].size(); j++) {
            auto& map = ins_positions[i+1][j];
            auto index = std::pair<long, long>(base_index, count);

            if (map.size() >= threshold_num) {
                pos_queue.emplace_back(base_index, count);
                count++;
            }



            for (auto& id: seq_indices) {
                if (map.find(id) == map.end()) {
                    map.emplace(id, PosInfo(Bases::GAP));	
                }	  
            }
            for (auto& id: no_ins_reads) {
                map.emplace(id, PosInfo(Bases::GAP));	  
            }
            for (auto& pair: map) {
                auto b = pair.second.base;
                switch(b) {
                    case Bases::A:
                        stats_info[index].n_A++;
                        break;
                    case Bases::C:
                        stats_info[index].n_C++;
                        break;
                    case Bases::G:
                        stats_info[index].n_G++;
                        break;
                    case Bases::T:
                        stats_info[index].n_T++;
                        break;
                    case Bases::GAP:
                        stats_info[index].n_del++;
                        break;
                    default:
                        std::cout << "SHOULD NOT GET HERE" << std::endl;        
                 }   

            }

            align_info[index] = map;
            labels_info[index] = ins_positions_labels[i+1][j];
        }
    } 
}

int FeatureGenerator::find_center(std::vector<segment>& segments) {
    int dists[segments.size()]{0};
    for (unsigned int i = 0; i < segments.size(); i++) {
        for (unsigned int j = i + 1; j < segments.size(); j++) {

            EdlibAlignResult result = edlibAlign(segments[i].sequence.c_str(), segments[i].len, segments[j].sequence.c_str(),
                    segments[j].len, edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, NULL, 0));
            dists[i] += result.editDistance;
            dists[j] += result.editDistance;
            edlibFreeAlignResult(result);
        }
    }
    int best_pos_index = 0;
    for (unsigned int i = 0; i < segments.size(); i++) {
        if (dists[i] < dists[best_pos_index]) {
            best_pos_index = i;
        }
    }
    return best_pos_index;    

}

void FeatureGenerator::align_ins_longest_star(long base_index, std::vector<segment>& ins_segments, int longest_index,
        unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads) {
    align_center_star(base_index, ins_segments, longest_index, threshold_num, no_ins_reads);

}

void FeatureGenerator::align_ins_center_star(long base_index, std::vector<segment>& ins_segments,
        unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads) {
    int center_index = find_center(ins_segments);
    align_center_star(base_index, ins_segments, center_index, threshold_num, no_ins_reads);

}

std::unique_ptr<Data> FeatureGenerator::generate_features() {

    npy_intp dims[2];
    npy_intp dims2[2];
    npy_intp labels_dim[1];
    labels_dim[0] = dimensions[1];
    for (int i = 0; i < 2; i++) {
        dims[i] = dimensions[i];
        dims2[i] = dimensions2[i];
    }
 
    auto data = std::unique_ptr<Data>(new Data());
    
    while (pileup_iter->has_next()) {
        auto column = pileup_iter->next();
        long rpos = column->position;
        if (rpos < pileup_iter->start()) continue;
        if (rpos >= pileup_iter->end()) break;
        unsigned int threshold_num = threshold_prop * column->count();
        if (threshold_num == 0) threshold_num = 1;
        std::vector<segment> ins_segments;
        std::vector<uint32_t> no_ins_reads;
        bool col_has_enough_ins = false;
        unsigned int total_ins_len = 0;
        unsigned int max_indel = 0;
        std::string s;
        if (inference_mode) {
            std::pair<long, long> index {rpos, 0};
            labels_info[index] = labels[index];
            long ins_count = 1;
            index = std::make_pair(rpos, ins_count);
            auto found = labels.find(index);
            while (found != labels.end()) {
                char c = int_to_char(labels[index]);
                s.push_back(c);
                ins_count++;
                index = std::make_pair(rpos, ins_count);
                found = labels.find(index);
            }
        }
        if (s.size() > 0) {
            ins_segments.emplace_back(s, s.size(), -1);\
        } 

        while(column->has_next()) {
            auto r = column->next();
            if (r->is_refskip()) continue;
            if (align_bounds.find(r->query_id()) == align_bounds.end()) {
                align_bounds.emplace(r->query_id(), std::make_pair(r->ref_start(), r->ref_end()));
            }
            strand.emplace(r->query_id(), !r->rev());
            std::pair<long, long> index(rpos, 0);
            if (align_info.find(index) == align_info.end()) {
                pos_queue.emplace_back(rpos, 0);
            }
            if (r->is_del()) {
                // DELETION
                align_info[index].emplace(r->query_id(), PosInfo(Bases::GAP));
                stats_info[index].n_del++;
            } else {
                // POSITION

                auto qbase = r->qbase(0);
                align_info[index].emplace(r->query_id(), PosInfo(qbase));
                switch(qbase) {
                    case Bases::A:
                        stats_info[index].n_A++;
                    break;
                    case Bases::C:
                        stats_info[index].n_C++;
                    break;
                    case Bases::G:
                        stats_info[index].n_G++;
                    break;
                    case Bases::T:
                        stats_info[index].n_T++;
                    break;
                    default:
                        std::cout << "SHOULD NOT GET HERE" << std::endl;        
                }   
                // INSERTION
                if (r-> indel() > 0) {
                    std::string s;
                    s.reserve(r->indel());
                    for (int i = 1, n = r->indel(); i <= n; ++i) {
                        qbase = r->qbase(i);
                        s.push_back(base_to_char(qbase));                 
                    }
                    if (static_cast<unsigned int>(r->indel()) > max_indel) max_indel = r->indel();
                    total_ins_len += r->indel();     
                    ins_segments.emplace_back(s, r->indel(), static_cast<int>(r->query_id()));
                } else {
                    no_ins_reads.push_back(r->query_id());

                }

           }


        }
        float avg_ins_len = total_ins_len/ (float) column->count();
        if (avg_ins_len   >= align_len_threshold && ins_segments.size() > 0) col_has_enough_ins = true;

        if (col_has_enough_ins) {

            align_ins_center_star(rpos, ins_segments, threshold_num, no_ins_reads);
            col_has_enough_ins = false;

        } else {
            for (auto& s: ins_segments) {
                if (s.index == -1) continue;
                long count = 1;    
                for (auto& c: s.sequence) {		
                    std::pair<long, long> index(rpos, count);
                    auto b = char_to_base(c);
                    align_info[index].emplace(s.index, PosInfo(b));
                    switch(b) {
                        case Bases::A:
                            stats_info[index].n_A++;
                        break;
                        case Bases::C:
                            stats_info[index].n_C++;
                        break;
                        case Bases::G:
                            stats_info[index].n_G++;
                        break;
                        case Bases::T:
                            stats_info[index].n_T++;
                        break;
                        default:
                            std::cout << "SHOULD NOT GET HERE" << std::endl;        
                    }   

                    if (align_info[index].size() == threshold_num) {
                        pos_queue.emplace_back(rpos, count);	
                    }
                    count++;
                }	
            }

            for (auto& s: ins_segments) {
                if (s.index == -1) continue;
                for (long i = s.len + 1; i <= max_indel; i++) {
                    std::pair<long, long> index(rpos, i);
                    stats_info[index].n_del++; 

                    align_info[index].emplace(s.index, PosInfo(Bases::GAP));		   

                }	

            }	    

            for (auto& id: no_ins_reads) {
                for (long i = 1; i <= max_indel; i++) {
                    std::pair<long, long> index(rpos, i);
                    stats_info[index].n_del++;  

                    align_info[index].emplace(id, PosInfo(Bases::GAP));		   

                }	
            }	
            for (long i = 1; i <= max_indel; i++) {
                std::pair<long, long> index(rpos, i);
                labels_info[index] = ENCODED_BASES[Bases::GAP];
            }

        }
        //BUILD FEATURE MATRIX
        while (pos_queue.size() >= dimensions[1]) {
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
                *value_ptr_16 = pos_stats.n_del;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 1, s);
                *value_ptr_16 = pos_stats.n_A;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 2, s);
                *value_ptr_16 = pos_stats.n_C;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 3, s);
                *value_ptr_16 = pos_stats.n_G;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 4, s);
                *value_ptr_16 = pos_stats.n_T;
                
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

            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s;
                uint8_t value = labels_info[*curr];
                value_ptr = (uint8_t*) PyArray_GETPTR1(Y, s);
                *value_ptr = value;
            }

            data->X.push_back(X);
            data->X2.push_back(X2);
            data->Y.push_back(Y);
            data->positions.emplace_back(pos_queue.begin(), pos_queue.begin() + dimensions[1]);
            for (auto it = pos_queue.begin(), end = pos_queue.begin() + WINDOW; it != end; ++it) {
                align_info.erase(*it);
            }
            pos_queue.erase(pos_queue.begin(), pos_queue.begin() + WINDOW);
        }
    }

    return data;
}


