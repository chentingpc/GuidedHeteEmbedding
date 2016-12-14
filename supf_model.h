#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <pthread.h>
#include <cassert>
#include <utility>
#include <vector>
#include <unistd.h>
#include "./common.h"
#include "./utility.h"
#include "./sampler.h"
#include "./data_helper.h"
#include "./emb_model.h"
#include "./metrics.h"
using namespace std;

typedef vector<pair<int, real> > vecOfIntReal;

const int IS_SRC_TRAIN = 0;         // constants for indicating condition
const int IS_SRC_TEST = 1;
const int IS_DST_TRAIN = 2;
const int IS_DST_TEST = 3;
const int NEG_BY_SAMPLING_TOP = 100;

class SupervisedFeatureModel: public EmbeddingModel {
  // parameters
  real                          *bias_node;               //
  real                          *weights_node;            // shared by src/dst in train/test pairs
  real                          *weights_node_type;       // shared by src/dst in train/test pairs

  bool                          *train_src_feat_has_node_type;  // if features have that node type
  bool                          *train_dst_feat_has_node_type;  // each is an array
  bool                          *test_src_feat_has_node_type;   // n_type -> ture/false
  bool                          *test_dst_feat_has_node_type;
  real                           *train_src_feat_node_type_cnt;  // number of feature instances in each node type,
  real                           *train_dst_feat_node_type_cnt;  // for a given node in the train/test features
  real                           *test_src_feat_node_type_cnt;   // matrix of i-th node to j-th n_type
  real                           *test_dst_feat_node_type_cnt;   // indexed by [i * num_node_type + j]
  vector<real>                  pred;
  GSLRandUniform                gsl_rand;

  real                          dropout;
  real                          lr_supf_emb;
  real                          reg_supf_emb;
  real                          lr_supf_ntype_w;
  real                          lr_supf_nbias;
  real                          init_lr_supf_emb;
  real                          init_lr_supf_ntype_w;
  real                          init_lr_supf_nbias;
  int64                         current_sample_count_supf;  // count only by supf only

  // traing data
  const vector<int>             *train_group_p;
  const vector<pair<int, int> > *train_pairs_p;
  const vector<real>            *train_pairs_label_p;
  const map<int, vecOfIntReal>  *train_src_features_p;
  const map<int, vecOfIntReal>  *train_dst_features_p;
  vector<int>                   train_pairs_pos;            // element is entry for pos train_pairs
  map<int, vector<int> >        train_pairs_neg_by_src;     // given src, return a train_pairs_neg
  NodeSampler                   *neg_author_sampler;
  NodeSampler                   *test_neg_author_sampler;
  int                           negative_by_sampling;       // negative by sample or by provided in label
                                                            // if true, negatives are sampled
                                                            // if false, will choose from labels
  vector<int>                   train_group_new;            // auxilliary variables
  vector<pair<int, int> >       train_pairs_new;
  vector<real>                  train_pairs_label_new;
  int                           max_src_feat_size;          // max of source node features size

  // test data
  const vector<int>             *test_group_p;
  const vector<pair<int, int> > *test_pairs_p;
  vector<pair<int, int> >       test_pairs_dup;
  const vector<real>            *test_pairs_label_p;
  const map<int, vecOfIntReal>  *test_src_features_p;
  const map<int, vecOfIntReal>  *test_dst_features_p;

  /**********************************************
   * Work scheduler and helpers
   *********************************************/

  static void *train_helper(void* context) {
      Context *c = (Context *)context;
      SupervisedFeatureModel* p = static_cast<SupervisedFeatureModel*>(c->model_ptr);
      p->train_with_weight(c->id);
      return NULL;
  }

  static void *eval_helper(void* context) {
      Context *c = (Context *)context;
      SupervisedFeatureModel* p = static_cast<SupervisedFeatureModel*>(c->model_ptr);
      p->eval_thread(c->id);
      return NULL;
  }

  void eval_thread(int map_topk);

  void eval(int map_topk = -1);

  /**********************************************
   * Initializations
   *********************************************/

  // only keep train_percent of groups in train data
  void _down_sample_train_data(real train_percent);

  void _build_feature_node_type_cnt(const map<int, vecOfIntReal > *features_p,
      bool *&feat_has_node_type, real *&feat_node_type_cnt);

  void init_variables();

  void init_vector();

  void init_runtime();

  /**********************************************
   * Training functions
   *********************************************/

  // first sample a positive (src, dst_pos), then sample dst_neg, either randomly or by pre-given
  // negative candidates
  inline void _fetch_train_triple(int &src, int &dst_pos, int &dst_neg,
                                  uint &r_seed, uint64 &n_seed, bool sample_pos = true);

  // triple based training main function
  void train_with_weight(int id);

  // choice: 0 for average, 1 for weighted
  const vector<real> &predict(int choice = 1);

  // compute the feature vector of a node by (1) first average all embeddings of same node type for
  //  each node type, (2) then weighted combine embeddings of each node type
  // is_src_train: 0 -> src & train, 1 -> src & test, 2 -> dst & train, 3 -> dst & test, see constants
  inline void _get_weighted_node_vector(const int &node, const int &is_src_train,
    real *vec_int, real *vec, bool *ignore_feats = NULL);

  // simply compute the feature vector of a node by averaging all its features' embedding
  // features: node's features (other nodes in network)
  // vec: node's feature vector
  inline void _get_averaged_node_vector(const vecOfIntReal &features, real *vec);

  inline void _update_fpa(const int &src, const int &dst, const real &score_err,
    const real *src_vec, const real *src_vec_int, const real *dst_vec, const real *dst_vec_int,
    real *src_err, real *dst_err, bool *ignore_feats = NULL);

  // choice 0: update node weight
  // choice 1: update embedding
  // is_src_train: 0 src & train, 1: src & test, 2: dst & train, 3: dst & test, see constants
  inline void _update_node_weight_or_embedding(const int &node, const int &is_src_train,
    real *pre_vec_err, int choice = 1, bool *ignore_feats = NULL);

 public:
  SupervisedFeatureModel(DataHelper *data_helper,
                         NodeSampler *node_sampler,
                         EdgeSampler *edge_sampler,
                         int dim, const Config *conf_p):
                         EmbeddingModel(data_helper, node_sampler, edge_sampler, dim, conf_p),
                         dropout(0),
                         lr_supf_emb(0),
                         reg_supf_emb(0),
                         lr_supf_ntype_w(0),
                         lr_supf_nbias(0),
                         current_sample_count_supf(0),
                         neg_author_sampler(NULL),
                         test_neg_author_sampler(NULL),
                         max_src_feat_size(0) {
    clock_t start, end;
    printf("Start to initialize %s...\n", "SupervisedFeatureModel");
    fflush(stdout);
    start = clock();
    init_variables();
    init_vector();
    init_runtime();
    end = clock();
    printf("%s initialized in %.2f (s).\n", "SupervisedFeatureModel",
      (double)(end - start) / CLOCKS_PER_SEC);
  }

  void fit();

  void load(string embedding_file, bool is_binary) {
    printf("[WARNING] load function is not implemented for SupervisedFeatureModel.\n");
  };

  void save(string embedding_file, bool is_binary, string pred_file);
};
