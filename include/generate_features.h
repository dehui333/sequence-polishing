#ifndef GENERATE_FEATURES_H
#define GENERATE_FEATYRES_H

#include <Python.h>
extern "C" {
    #define NO_IMPORT_ARRAY
    #define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
    #include "numpy/arrayobject.h"
}

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "models.h"


constexpr int dimensions[] = {30, 90}; 
constexpr int dimensions2[] = {5, 90}; // dimensions for second matrix
constexpr int WINDOW = dimensions[1] / 3;
constexpr int REF_ROWS = 1;
constexpr float threshold_prop = 0; // need this proportion of reads to support a base(ACTG) in the position to include it
constexpr unsigned int align_len_threshold = 0; // need avg ins len >= this at the position to align it 


struct Data{
    std::vector<std::vector<std::pair<long, long>>> positions;
    std::vector<PyObject*> X;
    std::vector<PyObject*> Y;
    std::vector<PyObject*> X2;
};

struct PosInfo{
    Bases base;

    PosInfo(Bases b) : base(b) {};
};

struct PosStats {

    uint16_t n_del = 0;
    uint16_t n_A = 0;
    uint16_t n_C = 0;
    uint16_t n_G = 0;
    uint16_t n_T = 0;
    
    //PosStats() : avg_mq(0), n_mq(0), avg_pq(0), n_pq(0) {};
    
    
};

struct EnumClassHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

extern std::unordered_map<Bases, uint8_t, EnumClassHash> ENCODED_BASES;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2;  
    }
};

class FeatureGenerator {

    private:
        // unchanged after construction
        std::unique_ptr<BAMFile> bam;
        std::unique_ptr<PositionIterator> pileup_iter;
        const char* draft;
        const bool inference_mode;
        
        // store progress
        std::unordered_map<std::pair<long, long>, uint8_t, pair_hash> labels;
        std::vector<std::pair<long, long>> pos_queue;
        std::unordered_map<std::pair<long, long>, std::unordered_map<uint32_t, PosInfo>, pair_hash> align_info;
        std::unordered_map<std::pair<long, long>, uint8_t, pair_hash> labels_info;
        std::unordered_map<uint32_t, std::pair<long, long>> align_bounds;
        std::unordered_map<uint32_t, bool> strand;
        std::unordered_map<std::pair<long, long>, PosStats, pair_hash> stats_info;
        struct segment {
            std::string sequence;
            int len;
            int index;
            segment(std::string seq, int l, int id) : sequence(seq), len(l), index(id) {};
        };

        Bases char_to_base(char c);
        char base_to_char(Bases b);
        char int_to_char(uint8_t i);
        Bases int_to_base(uint8_t i);
        uint8_t char_to_int(char c);
            
        void align_center_star(long base_index, std::vector<segment>& segments, int star_index,
            unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads);

        void align_ins_longest_star(long base_index, std::vector<segment>& ins_segments, int longest_index, 
            unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads);

        void align_ins_center_star(long base_index, std::vector<segment>& ins_segments,
            unsigned int& threshold_num, std::vector<uint32_t>& no_ins_reads);

        int find_center(std::vector<segment>& segments);

        void convert_py_labels_dict(PyObject *dict);

    public:
        FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict, bool inference_mode);   

        std::unique_ptr<Data> generate_features();
};


#endif //GENERATE_FEATURES_H
