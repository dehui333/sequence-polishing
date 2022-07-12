#ifndef GENERATE_FEATURES_H
#define GENERATE_FEATURES_H

#include <Python.h>
extern "C" {
    #define NO_IMPORT_ARRAY
    #define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
    #include "numpy/arrayobject.h"
}
#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "models.h"

typedef uint32_t pos_index_t;

constexpr int dimensions[] = {30, 90}; 
constexpr int dimensions2[] = {6, 90};
constexpr int dimensions3[] = {1, 90}; // x3
constexpr int WINDOW = dimensions[1] / 3;
constexpr int REF_ROWS = 1; // ref_rows=1 to include draft in the feature

constexpr float UNCERTAIN_POSITION_THRESHOLD = 0.15;
constexpr float NON_GAP_THRESHOLD = 0.01;
constexpr uint64_t LABEL_SEQ_ID = -1;

struct Data{
    std::vector<std::vector<std::pair<pos_index_t, pos_index_t>>> positions;
    std::vector<PyObject*> X;
    std::vector<PyObject*> Y;
    std::vector<PyObject*> X2;
    std::vector<PyObject*> X3; // x3
};

struct PosInfo{
    Bases base;
    uint8_t mq;
    PosInfo(Bases b, uint8_t mq) : base(b), mq(mq) {};
};

struct PosStats {
    uint32_t n_total = 0;
    uint32_t n_GAP = 0;
    uint32_t n_A = 0;
    uint32_t n_C = 0;
    uint32_t n_G = 0;
    uint32_t n_T = 0;

    uint32_t largest_diff = 0; 
    
    float normalized_cov = 0; // x3
    
    
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
        return h1 ^ h2; // Bitwise XOR (exclusive or), 1 when both bits are different, 0 when they are the same
    }
};

class FeatureGenerator {

    private:
        // unchanged after construction
        std::unique_ptr<BAMFile> bam;
        std::unique_ptr<PositionIterator> pileup_iter;
        const char* draft;

        bool has_labels;
        uint16_t counter = 0;
        
        // store progress
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, uint8_t, pair_hash> labels;
        std::deque<std::pair<pos_index_t, pos_index_t>> pos_queue;
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, std::unordered_map<uint32_t, PosInfo>, pair_hash> align_info;
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, uint8_t, pair_hash> labels_info;
        std::unordered_map<uint32_t, std::pair<pos_index_t, pos_index_t>> align_bounds;
        std::unordered_map<uint32_t, bool> strand;
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, PosStats, pair_hash> stats_info;

        std::queue<uint16_t> distances; 

        struct segment {
            std::string sequence;
            uint64_t index;
            uint8_t mq;
            segment(std::string seq, int id, uint8_t mq) : sequence(seq), index(id), mq(mq) {};
            segment(std::string seq, int id) : sequence(seq), index(id) {};
        };

        Bases char_to_base(char c);
        char base_to_char(Bases b);
        char forward_int_to_char(uint8_t i);
        uint8_t char_to_forward_int(char c);
            
        void align_to_target(pos_index_t base_index, std::vector<segment>& segments, int target_index,
            std::vector<segment>& no_ins_reads);

        void align_ins_longest(pos_index_t base_index, std::vector<segment>& ins_segments, 
            std::vector<segment>& no_ins_reads);

        void align_ins_center(pos_index_t base_index, std::vector<segment>& ins_segments,
            std::vector<segment>& no_ins_reads);

        int find_center(std::vector<segment>& segments);

        int find_longest(std::vector<segment>& segments);

        void convert_py_labels_dict(PyObject *dict);

        void increment_base_count(std::pair<pos_index_t, pos_index_t>& index, PosInfo& pos_info);

        void pos_queue_push(std::pair<pos_index_t, pos_index_t>& index);

        void pos_queue_pop(uint16_t num);

    public:
        FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict, uint16_t median, uint16_t mad);   

        std::unique_ptr<Data> generate_features();
};


#endif //GENERATE_FEATURES_H
