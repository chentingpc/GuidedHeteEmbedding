#pragma once

//#define int NOT_PA_TASK
#define PA_TASK

typedef float real;                     // Precision of float numbers
typedef unsigned int uint;
typedef long long int64;
typedef unsigned long long uint64;

const float NEG_SAMPLING_POWER = 0.75;  // unigram downweighted
const int hash_table_size = 5e7;        // at least twice the size of num_vertices
const int neg_table_size = 1e8;         // at least the size of sum_w / min_w
const double PATH_NORMALIZED_SUM_DEFAULT = 100000;  // avoid precision overflow

const double LOG_MIN = 1e-15;           // Smoother for log
const int SIGMOID_BOUND = 10;
const int sigmoid_table_size = 1000;
const int MAX_STRING = 2000;
const int MAX_LINE_LEN = 65535;

const int PATH_DIRECTION_NORMAL = 0;
const int PATH_DIRECTION_REVERSE = 1;
const int PATH_DIRECTION_BIDIRECTION = 2;
const int PATH_ORDER_SINGLE = 0;
const int PATH_ORDER_CONTEXT = 1;

struct Vertex {
  // content in this structure can be modified outside, which is not safe
  double degree;
  char *name;
  int type;
};


struct Graph {
  // content in this structure can be modified outside, which is not safe
  Vertex                  *vertex;
  int                     *vertex_type;
  double                  *vertex_degree_of_etype;

  int                     *edge_source_id;
  int                     *edge_target_id;
  double                  *edge_weight;
  int                     *edge_type;
  double                  *edge_type_w;

  bool                    *node_type_to_edge_type;

  int                     *num_vertices_p;
  int64                   *num_edges_p;
  int                     *num_node_type_p;
  int                     *num_edge_type_p;
};
