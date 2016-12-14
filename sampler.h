#pragma once
#include <math.h>
#include <cassert>
#include "./common.h"
#include "./config.h"
#include "./utility.h"
#include "./data_helper.h"


/* high precision unifrom distribution generator */
class GSLRandUniform {
  const gsl_rng_type    *gsl_T;
  gsl_rng               *gsl_r;

 public:
  explicit GSLRandUniform(const int seed = 314159265) {
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, seed);
  }

  double operator()() {
    // This function returns a double precision floating point number
    //  uniformly distributed in the range [0,1)
    return gsl_rng_uniform(gsl_r);
  }
};


// The alias sampling algorithm is used to sample an edge propotional to its weight in O(1) time.
class EdgeSampler {
  const Graph             *graph;
  const double            *edge_weight;
  int64                   num_edges;

  GSLRandUniform          gsl_rand;

  int64                   *alias;
  double                  *prob;

  void init_alias_table() {
    alias = new int64[num_edges];
    prob = new double[num_edges];
    if (alias == NULL || prob == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }
    memset(alias, 0, sizeof(int64) * num_edges);
    memset(prob, 0, sizeof(double) * num_edges);

    double *norm_prob = new double[num_edges];
    int64 *large_block = new int64[num_edges];
    int64 *small_block = new int64[num_edges];
    if (norm_prob == NULL || large_block == NULL || small_block == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }

    double sum = 0;
    int64 cur_small_block, cur_large_block;
    int64 num_small_block = 0, num_large_block = 0;

    for (int64 k = 0; k != num_edges; k++) sum += edge_weight[k];
    assert(sum != 0);  // either set at least one non-zero path, or drop network embedding part (set omega = 0)
    for (int64 k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

    for (int64 k = num_edges - 1; k >= 0; k--) {
      if (norm_prob[k] < 1)
        small_block[num_small_block++] = k;
      else
        large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block) {
      cur_small_block = small_block[--num_small_block];
      cur_large_block = large_block[--num_large_block];
      prob[cur_small_block] = norm_prob[cur_small_block];
      alias[cur_small_block] = cur_large_block;
      norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
      if (norm_prob[cur_large_block] < 1)
        small_block[num_small_block++] = cur_large_block;
      else
        large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    delete[]norm_prob;
    delete[]small_block;
    delete[]large_block;
  }

 public:
  explicit EdgeSampler(const Graph *graph) :
      graph(graph) {
    printf("Building edge sampler..\r");
    fflush(stdout);
    clock_t start, end;
    start = clock();
    edge_weight = graph->edge_weight;
    num_edges = *graph->num_edges_p;
    init_alias_table();
    end = clock();
    printf("Edge sampler built in %.2f (s)       \n", (double)(end-start) / CLOCKS_PER_SEC);
  }

  inline int64 sample() {
    int64 k = (int64)num_edges * gsl_rand();
    return gsl_rand() < prob[k] ? k : alias[k];
  }
};


class NodeSampler {
  int                 *neg_table;         // based on overall network
  int                 **neg_tables;       // based on edge type and node type
                                          // indexed by i * num_node_type + j

  const Graph         *graph;
  const Config        *conf_p;
  const bool          *node_type_to_edge_type;
  const Vertex        *vertex;
  const int           *vertex_type;
  const int           *edge_source_id;
  const int           *edge_target_id;
  const double        *edge_weight;
  const int           *edge_type;

  int                 num_vertices;
  int64               num_edges;
  int                 num_node_type;
  int                 num_edge_type;

  /* Fastly generate a random integer */
  inline int Rand(uint64 &seed) {
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
  }

  // calculate degree of node of n_type using edegs of e_type, store in vertex_degree
  void cal_degree(double *vertex_degree, int e_type, int n_type, bool check_type = true) {
    int src, dst, this_e_type, src_type, dst_type;
    double w, base_deg, min_deg = 1e9;
    memset(vertex_degree, 0, num_vertices * sizeof(double));
    // compute vertex degree of target type
    for (int64 i = 0; i < num_edges; i++) {
      src = edge_source_id[i];
      dst = edge_target_id[i];
      w = edge_weight[i];
      this_e_type = edge_type[i];
      if (check_type && this_e_type != e_type)
        continue;
      if (w < min_deg)
        min_deg = w;
      src_type = vertex_type[src];
      dst_type = vertex_type[dst];
      if (!check_type || src_type == n_type) {
        vertex_degree[src] += w;
      }
      if (!check_type || dst_type == n_type) {
        vertex_degree[dst] += w;
      }
    }
    // add base_deg (in terms of multiple of non-zero min_deg)
    if (min_deg == 0) min_deg = 1;
    if (conf_p->use_path_conf)
      base_deg = conf_p->path_base_deg[e_type];
    else
      base_deg = conf_p->path_base_deg_default;
    base_deg *= min_deg;
    for (int i = 0; i < num_vertices; i++) {
      if (vertex_type[i] == n_type)
        vertex_degree[i] += base_deg;
    }
  }

  // zero degree node will not be sampled
  void set_table(int *table, const double *vertex_degree, float neg_sampling_power) {
    double sum = 0, cur_sum = 0, por = 0, deg;
    double max_w = 0, min_w = 1e8;  // w is a re-weighted degree
    int k = 0;
    for (int i = 0; i != num_vertices; i++) {
        deg = vertex_degree[i];
        if (deg == 0.) continue;
        sum += pow(vertex_degree[i], neg_sampling_power);
    }
    for (int i = 0; i < num_vertices; i++) {
        deg = vertex_degree[i];
        if (deg == 0.) continue;
        double w = pow(deg, neg_sampling_power);
        if (w > max_w) max_w = w;
        if (w < min_w && w > 0) min_w = w;
        cur_sum += w;
        por = cur_sum / sum;
        while ((double)k / neg_table_size < por && k != neg_table_size) {
          table[k++] = i;
        }
    }
    if (k != neg_table_size)
      printf("[ERROR!] k %d, neg_table_size %d. check path sum. try to add base deg.\n", k, neg_table_size);
    assert(k == neg_table_size); // even this equation not hold, they should be close, check precision
    int min_neg_table_size = int(cur_sum / min_w) + 1;
    if (min_neg_table_size > neg_table_size) {
      printf("[WARNING!!!] Some negative nodes with small weights cannot be sampled, max_w %f, min_w %f.", max_w, min_w);
      printf("Set neg_table_size in common.h at least %d, currently its %d\n", min_neg_table_size, neg_table_size);
    }
  }

  void init_neg_table() {
    neg_table = new int[neg_table_size];
    double *vertex_degree = new double [num_vertices];
    // for (int i = 0; i < num_vertices; i++)
    //  vertex_degree[i] = vertex[i].degree;
    cal_degree(vertex_degree, -1, -1, false);  // calculate with edges in case edge re-weighting
    set_table(neg_table, vertex_degree, conf_p->path_sampling_pow_default);
    delete []vertex_degree;
  }

  void init_neg_tables() {
    double *vertex_degree = new double [num_vertices];
    neg_tables = new int*[num_edge_type * num_node_type];
    for (int i = 0; i < num_edge_type; i++) {
      int i_row_start = i * num_node_type;
      for (int j = 0; j < num_node_type; j++) {
        if (node_type_to_edge_type[j * num_edge_type + i]) {
          int *table = new int[neg_table_size];
          cal_degree(vertex_degree, i, j);
          float neg_sampling_power = conf_p->path_sampling_pow_default;
          if (conf_p->use_path_conf)
            neg_sampling_power = conf_p->path_sampling_pow[i];
          set_table(table, vertex_degree, neg_sampling_power);
          neg_tables[i_row_start + j] = table;
        }
        else {
          neg_tables[i_row_start + j] = NULL;
        }
      }
    }
    delete [] vertex_degree;
  }

 public:
  explicit NodeSampler (const Graph *graph, const Config *conf_p) :
      graph(graph),
      conf_p(conf_p) {
    printf("Building node sampler..\r");
    fflush(stdout);
    clock_t start, end;
    start = clock();
    node_type_to_edge_type = graph->node_type_to_edge_type;
    vertex = graph->vertex;
    num_vertices = *graph->num_vertices_p;
    num_edges = *graph->num_edges_p;
    num_node_type = *graph->num_node_type_p;
    num_edge_type = *graph->num_edge_type_p;
    vertex_type = graph->vertex_type;
    edge_source_id = graph->edge_source_id;
    edge_target_id = graph->edge_target_id;
    edge_weight = graph->edge_weight;
    edge_type = graph->edge_type;

    assert(num_vertices > 0);
    assert(num_edges > 0);
    assert(num_node_type > 0);
    assert(num_edge_type > 0);

    init_neg_table();
    init_neg_tables();

    end = clock();
    printf("Node sampler built in %.2f (s)       \n", (double)(end-start) / CLOCKS_PER_SEC);
  }

  // build a sampler from a discrete distribution of size dist_size
  // should only use sample(seed) for sample
  explicit NodeSampler(const double *dist, const int dist_size,
        float neg_sampling_power = NEG_SAMPLING_POWER):
      graph(NULL),
      node_type_to_edge_type(NULL),
      vertex(NULL),
      vertex_type(NULL),
      edge_source_id(NULL),
      edge_target_id(NULL),
      edge_weight(NULL),
      edge_type(NULL)
      {
    num_vertices = dist_size;
    neg_table = new int[neg_table_size];
    set_table(neg_table, dist, neg_sampling_power);
  }

  /* Sample negative vertex samples according to aggregated vertex degrees */
  inline int64 sample(uint64 &seed) {
    return neg_table[Rand(seed)];
  }

  /* Sample negative vertex samples of node type according to vertex degrees under given edge type*/
  inline int64 sample(uint64 &seed, const int &e_type, const int &n_type) {
    return neg_tables[e_type * num_node_type + n_type][Rand(seed)];
  }
};
