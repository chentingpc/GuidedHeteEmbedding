#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <cassert>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include "./common.h"
#include "./config.h"
#include "./utility.h"

using namespace std;

class DataHelper : public VertexHashTable {
  Graph                   graph;              // this compact struct is used for convinience
  Config                  *conf_p;
  bool                    *node_type_to_edge_type;
                                              // this is a matrix indexed by i * num_edge_type + j,
                                              // 1 if a node type involved with a edge type

  /* network vertex data struct */
  Vertex                  *vertex;
  int                     *vertex_type;       // init to 0
  double                  *vertex_degree_of_etype;
                                              // indexed by [vid * num_edge_type + e_type]
  int                     num_vertices;
  map<int, string>        node_type2name;     // type (int) => type (name, str)
  map<string, int>        node_name2type;     // type (name, str) => type (int)
  int                     num_node_type;      // init to 1

  /* network edge data struct */
  int                     *edge_source_id;    // edge info is stored separted for performance sake
  int                     *edge_target_id;
  double                  *edge_weight;
  int                     *edge_type;         // init to 0
  double                  *edge_type_w;       // edge weights sum for each meta-path
  int64                   num_edges;
  map<int, string>        edge_type2name;
  map<string, int>        edge_name2type;
  int                     num_edge_type;      // init to 1

  /* train/test data struct */
  vector<int>             train_group;
  vector<pair<int, int> > train_pairs;
  vector<real>            train_pairs_label;
  map<int, vector<int> >  train_src_features;
  map<int, vector<int> >  train_dst_features;
  vector<int>             test_group;         // each element specify the group size for ranking
  vector<pair<int, int> > test_pairs;         // test pairs, can be nodes or features
  vector<real>            test_pairs_label;   // labels for test pairs, if any
  vector<string>          test_pairs_type;    // task type of test pairs, if any
  vector<int>             test_pairs_etype;   // task type as edge type of test pairs
  map<int, vector<int> >  test_src_features;  // features (node indexes) for src nodes (LHS)
  map<int, vector<int> >  test_dst_features;  // features (node indexes) for dst nodes (RHS)
  map<string, pair<int, int> >
                          test_task_group_start_end;

  /* other auxilliary data */
  int                     max_num_vertices;
  map<string, int>        non_vertex_name2id;
  map<int, string>        non_vertex_id2name;

  /*
   * Read edges, indexing edges/nodes, map edge types, pre-process network
   */
  void load_network();

  /*
   * Read node to type_name, map node types
   * If not load, assuming only one node type
   */
  void load_node_type();

  /*
   * Construct the schema relation between node type and edge type from network/node_type files
   */
  void reload_node_type_to_edge_type(bool printing = true);

 public:
  explicit DataHelper(Config *conf_p = NULL) :
      VertexHashTable(),
      conf_p(conf_p),
      node_type_to_edge_type(NULL),
      num_vertices(0),
      num_node_type(0),
      num_edges(0),
      num_edge_type(0),
      max_num_vertices(10000) {
    load_network(); // in future optimization, load_network can be put after load_node_type

    if (conf_p->node_type_file.size() > 0) {
      load_node_type();
    } else {
      printf("Number of node type: %d          \n", num_node_type);
      printf("[node_name2type/node_type2name not presented]\n");
    }

    if (hash_table_size < 2 * num_vertices) {
      printf("[WARNING!!!] You should set a bigger hash_table_size to speed-up data loading.\n");
      printf("You may resize in common.h and comment out in data_helper.h\n");
      printf("Suggested hash_table_size > %lld\n", 2 * static_cast<int64>(num_vertices));
    }
  }

  /*
   * Loading paper-author train or test (w/ feature) file
   *
   * pa_file: paper-author train or test file
   * po_file: paper-feature train or test file
   *
   * require paper to be in int, and author and all features being in the network
   */
  void load_pa_trainortest(string pa_file, string po_file, bool is_training);

  // constructing group information from train/test pairs
  // require train_pairs, test_pairs, and organized according to paper
  void construct_group(bool test_only = true);

  void load_test(string test_file);

  void _check_pa_test_data();

  /**********************
   * attribute getters
   **********************/

  int get_num_vertices() {
    return num_vertices;
  }

  int get_num_edges() {
    return num_edges;
  }

  int get_vertex_id(char *name) {
    return search_hash_table(name, vertex);
  }

  const Graph * get_graph() {
    return &graph;
  }

  const Vertex * get_vertex() {
    return vertex;
  }

  const int * get_vertex_type() {
    return vertex_type;
  }

  const map<int, string> & get_node_type2name() {
    return node_type2name;
  }

  const map<string, int> & get_node_name2type() {
    return node_name2type;
  }

  const map<int, string> & get_edge_type2name() {
    return edge_type2name;
  }

  const map<string, int> & get_edge_name2type() {
    return edge_name2type;
  }

  const int * get_edge_type() {
    return edge_type;
  }

  const double * get_edge_weight() {
    return edge_weight;
  }

  const int * get_edge_source_id() {
    return edge_source_id;
  }

  const int * get_edge_target_id() {
    return edge_target_id;
  }

  const vector<int> * get_train_group() {
    return &train_group;
  }

  const vector<pair<int, int> > * get_train_pairs() {
    return &train_pairs;
  }

  const vector<real> * get_train_pairs_label() {
    return &train_pairs_label;
  }

  const map<int, vector<int> > * get_train_src_features() {
    return &train_src_features;
  }

  const map<int, vector<int> > * get_train_dst_features() {
    return &train_dst_features;
  }

  const vector<int> * get_test_group() {
    return &test_group;
  }

  const vector<pair<int, int> > * get_test_pairs() {
    return &test_pairs;
  }

  const vector<real> * get_test_pairs_label() {
    return &test_pairs_label;
  }

  const vector<string> * get_test_pairs_type() {
    return &test_pairs_type;
  }

  const vector<int> * get_test_pairs_etype() {
    return &test_pairs_etype;
  }

  const map<int, vector<int> > * get_test_src_features() {
    return &test_src_features;
  }

  const map<int, vector<int> > * get_test_dst_features() {
    return &test_dst_features;
  }

  const map<string, pair<int, int> > * get_test_task_group_start_end() {
    return &test_task_group_start_end;
  }

  const map<int, string> * get_non_vertex_id2name() {
    return &non_vertex_id2name;
  }
};
