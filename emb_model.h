#pragma once
#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <pthread.h>
#include <cassert>
#include <unistd.h>
#include "./common.h"
#include "./config.h"
#include "./utility.h"
#include "./sampler.h"
#include "./data_helper.h"
using namespace std;

class EmbeddingModel {
 protected:
  int                     dim;
  int                     num_negative;
  real                    lr_net_emb;
  real                    lr_net_w;
  real                    lr_net_etype_bias;
  real                    reg_net_emb;
  real                    init_lr_net_emb;
  real                    init_lr_net_w;
  real                    init_lr_net_etype_bias;
  int64                   total_samples;
  int64                   current_sample_count;
  int64                   current_sample_count_emb;  // count only by emb only
  int                     num_threads;
  bool                    using_edge_type_bias;
  bool                    using_transformation_vector;
  bool                    using_transformation_matrix;

  int64                   samples_before_switch_emb;
  int64                   samples_before_switch_other;
  bool                    *task_switchs_for_embedding;

  struct Context {
    EmbeddingModel *model_ptr;
    int id;
  };

  const Config            *conf_p;
  const Graph             *graph;
  const bool              *node_type_to_edge_type;
  const Vertex            *vertex;
  const int               *vertex_type;
  const double            *vertex_degree_of_etype;
  map<string, int>        node_name2type;
  map<int, string>        node_type2name;
  map<string, int>        edge_name2type;
  map<int, string>        edge_type2name;
  int                     num_node_type;
  int                     num_vertices;
  const int               *edge_type;
  const int               *edge_source_id;
  const int               *edge_target_id;
  int64                   num_edges;
  int                     num_edge_type;
  GSLRandUniform          uniform;
  bool                    use_path_conf;
  vector<int>             path_direction;
  vector<int>             path_order;
  const map<int, string>  *non_vertex_id2name;

  bool                    fit_not_finished;
  bool                    *edge_type_using_context;
  int                     ls;
  int                     band_width;

  real                    *emb_vertex;            // main embedding vector for each node
                                                  // indexed by [vid * dim + k]
  real                    *emb_context;           // context embedding vector for certain nodes
                                                  // indexed by [vid * dim + k]
  real                    *weight_edge_type;      // weights for each mlr_net_w-path
  real                    *W_m_band_chuck;        // memory chunk trick
  real                    **W_m_band;             // weight matrix diagonal band for each mlr_net_w-path
  real                    ***w_mn;                // weight vectors for each (path, node-type)
  real                    *bias_edge_type;        // bias for each mlr_net_w-path

  double                  *ll_edge_type;          // log-likelihood under different paths
  int64                   *ll_edge_type_cnt;      // count of times ll_edge_type being added up

  Sigmoid                 *sigmoid;
  DataHelper              *data_helper;
  NodeSampler             *node_sampler;
  EdgeSampler             *edge_sampler;
  GSLRandUniform          gsl_rand;

  /**********************************************
   * Work scheduler and helpers
   *********************************************/

  static void *fit_thread_helper(void* context) {
      Context *c = (Context *)context;
     EmbeddingModel* p = static_cast<EmbeddingModel*>(c->model_ptr);
      p->fit_thread(c->id);
      return NULL;
  }


  /**********************************************
   * Initializations
   *********************************************/

  void init_variables();

  void init_task_schduler();

  // Initialize the vertex embedding and the context embedding
  // There are some over-allocation in here, meaning some parameters might not be used but allocated
  // Not worry as long as memory is not a limit
  void init_vector();

  /**********************************************
   * Training functions
   *********************************************/

  // network embedding training main loop
  void fit_thread(int id);

  // training given edge sample
  inline void train_on_sample(const int &id, int64 &u, int64 &v, const int64 &curedge,
    double &ll, uint64 &seed, real *vec_error, real *e_type_bias_err_vec = NULL,
    real ***w_mn_err = NULL);

  // Update embeddings & return likelihood, skip-gram negative sampling objective
  inline real update_skipgram(real *vec_u, real *vec_v, real *vec_error, const int &label,
    const real &e_type_bias = 0, real *e_type_bias_err = NULL);

  // Update embeddings & return likelihood, max-margin negative sampling objective
  inline real update_maxmargin(real *src_vec, real *dst_vec_pos, real *dst_vec_neg, real *vec_error);

  // update embedding with vector weighting for embeddings
  inline real update_with_weight(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err,
    real *w_mn_u, real *w_mn_v, real *w_mn_err_u, real *w_mn_err_v);

 public:
  EmbeddingModel(DataHelper *data_helper,
                 NodeSampler *node_sampler,
                 EdgeSampler *edge_sampler,
                 int dim, const Config *conf_p) :
                dim(dim),
                lr_net_emb(0),
                lr_net_w(0),
                lr_net_etype_bias(0),
                reg_net_emb(0),
                using_edge_type_bias(false),
                using_transformation_vector(false),
                using_transformation_matrix(false),
                samples_before_switch_emb(0),
                samples_before_switch_other(0),
                conf_p(conf_p),
                data_helper(data_helper),
                node_sampler(node_sampler),
                edge_sampler(edge_sampler) {
    init_variables();
    init_task_schduler();
    init_vector();
  }

  virtual ~EmbeddingModel(){}

  virtual void fit();

  virtual void save(string embedding_file, bool is_binary, string pred_file = string());

  virtual void load(string embedding_file, bool is_binary);
};
