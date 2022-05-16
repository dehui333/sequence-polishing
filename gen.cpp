#include <Python.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <unordered_map>

#define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
#include "numpy/arrayobject.h"

#include "generate_features.h"

struct cov_info {
    uint16_t median = 0;
    uint16_t mad = 0;
    
};

static std::unordered_map<std::string, cov_info> contig_cov_info;

static PyObject* initialize_cpp(PyObject *self, PyObject *args) {
    PyObject *contig_names, *contig_lens, *item, *item2; 
    char* file_name; // reads to draft bam file

    // expects file_name, list of contigs in the draft, and lengths
    if (!PyArg_ParseTuple(args, "sOO", &file_name, &contig_names, &contig_lens)) return NULL; 

    if (!PyList_Check(contig_names) || !PyList_Check(contig_lens)) return NULL;
    if (PyList_Size(contig_names) != PyList_Size(contig_lens)) return NULL;
    uint16_t num_contigs = PyList_Size(contig_names);

    std::vector<std::string> contigs; 
    std::vector<uint32_t> lens;
    contigs.reserve(num_contigs);
    lens.reserve(num_contigs);
    contig_cov_info.reserve(num_contigs);
    if (PyList_Check(contig_names) && PyList_Check(contig_lens) && PyList_Size(contig_names) == PyList_Size(contig_lens)) {
        for (Py_ssize_t i = 0; i < PyList_Size(contig_names); i++) {
            item = PyList_GetItem(contig_names, i);
            item2 = PyList_GetItem(contig_lens, i);
            if (!item || !item2) {
                return NULL;
    
            }
            item = PyUnicode_AsEncodedString(item, "UTF-8", "strict");
            if (!item) {
                return NULL;
            }
            contigs.emplace_back(PyBytes_AsString(item));
            Py_DECREF(item);
            long len = PyLong_AsLong(item2);
            if (len < 0) return NULL;
            lens.push_back(len);
        }
    }
    
    uint64_t total_len = 0;
    for (auto l: lens) {
        total_len += l;
    }
    auto bam = readBAM(file_name);
    for (uint16_t contig_idx = 0; contig_idx < num_contigs; contig_idx++) {
        std::string region_string = contigs[contig_idx] + ':';
        auto contig_pileup = bam->pileup(region_string.c_str(), true);        
        uint32_t i = 0;
        uint32_t len = lens[contig_idx];
        uint16_t* coverages = new uint16_t[len];
        uint16_t* absolute_deviations = new uint16_t[len];
        while (contig_pileup->has_next()) {
            auto column = contig_pileup->next();
            long rpos = column->position;
            if (rpos < contig_pileup->start()) continue; 
            coverages[i++] = column->count();
        }

        uint16_t median_coverage = 0;
        uint16_t MAD = 0;

        std::sort(coverages, coverages + len);
    
        if (len % 2 == 0) {
            median_coverage = (coverages[len/2] + coverages[len/2-1])/2; 
        } else {
            median_coverage = coverages[len/2];
        }

        for (i = 0; i < len; i++) {
            absolute_deviations[i] = std::abs(coverages[i] - median_coverage);
        }

        std::sort(absolute_deviations, absolute_deviations + len);

        if (total_len % 2 == 0) {
            MAD = (absolute_deviations[len/2] + absolute_deviations[len/2-1])/2; 
        } else {
            MAD = absolute_deviations[len/2];
        }
        
        auto& s = contig_cov_info[contigs[contig_idx]];
        s.median = median_coverage;
        s.mad = MAD;
        delete[] coverages;
        delete[] absolute_deviations;

    }

    /*for (auto& p:contig_cov_info) {
        std::cout << p.first << " median " << p.second.median << " mad " << p.second.mad << std::endl;
    }*/
      
    Py_INCREF(Py_None);
    return Py_None;
}


// Module method definitions
static PyObject* generate_features_cpp(PyObject *self, PyObject *args) {
    srand(time(NULL));

    char *filename, *ref, *region;
    PyObject* dict = NULL;
    
    
    if (!PyArg_ParseTuple(args, "sssO", &filename, &ref, &region, &dict)) return NULL;
    
    std::string contig_name;
    for (uint16_t i = 0; i < strlen(region); i++) {
        if (region[i] == ':') break;
        contig_name.push_back(region[i]);
    }
    auto& s = contig_cov_info[contig_name];
    
    FeatureGenerator feature_generator {filename, ref, region, dict, s.median, s.mad};
    auto result = feature_generator.generate_features();
    PyObject* return_tuple = PyTuple_New(5);
    PyObject* pos_list = PyList_New(result->positions.size());
    PyObject* X_list = PyList_New(result->X.size());
    PyObject* X2_list = PyList_New(result->X2.size());
    PyObject* X3_list = PyList_New(result->X3.size());
    PyObject* Y_list = PyList_New(result->Y.size());

    for (int i = 0, size=result->positions.size(); i < size; i++) {
        auto& pos_element = result->positions[i];

        PyObject* inner_list = PyList_New(pos_element.size());
        for (int j = 0, s = pos_element.size(); j < s; j++) {
            PyObject* pos_tuple = PyTuple_New(2);
            PyTuple_SetItem(pos_tuple, 0, PyLong_FromLong(pos_element[j].first));
            PyTuple_SetItem(pos_tuple, 1, PyLong_FromLong(pos_element[j].second));

            PyList_SetItem(inner_list, j, pos_tuple);
        }
        PyList_SetItem(pos_list, i, inner_list);

        PyList_SetItem(X_list, i, result->X[i]);
        PyList_SetItem(X2_list, i, result->X2[i]);
        PyList_SetItem(X3_list, i, result->X3[i]);
        PyList_SetItem(Y_list, i, result->Y[i]);
    }
 
    PyTuple_SetItem(return_tuple, 0, pos_list);
    PyTuple_SetItem(return_tuple, 1, X_list);
    PyTuple_SetItem(return_tuple, 2, Y_list);
    PyTuple_SetItem(return_tuple, 3, X2_list);
    PyTuple_SetItem(return_tuple, 4, X3_list);

    return return_tuple;
}

static PyMethodDef gen_methods[] = {
        {"generate_features", generate_features_cpp, METH_VARARGS, "Generate features for polisher."},
        {"initialize", initialize_cpp, METH_VARARGS, "Initialize median and MAD coverage values."},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef gen_definition = {
        PyModuleDef_HEAD_INIT,
        "gen",
        "Feature generation.",
        -1,
        gen_methods
};


PyMODINIT_FUNC PyInit_gen(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&gen_definition);
}