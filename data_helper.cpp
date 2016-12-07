#include "data_helper.h"

void DataHelper::load_network() {
  FILE *fin;
  char name_v1[MAX_STRING], name_v2[MAX_STRING], type_name[MAX_STRING], line_buffer[MAX_LINE_LEN];
  vector<string> &valid_paths = conf_p->valid_paths;
  int vid, type, num_separator = 0;
  set<string> valid_paths_set;
  bool path_screening = false;
  double weight;
  clock_t start, end;
  start = clock();

  // count number of edges
  fin = fopen(conf_p->network_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: network file not found!\n");
    exit(-1);
  }
  num_edges = 0;
  while (fgets(line_buffer, sizeof(line_buffer), fin)) num_edges++;
  fclose(fin);

  // initial edges structure
  edge_source_id = new int[num_edges];
  edge_target_id = new int[num_edges];
  edge_weight = new double[num_edges];
  if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
    printf("Error: memory allocation failed!\n");
    exit(-1);
  }
  memset(edge_source_id, 0, sizeof(int) * num_edges);
  memset(edge_target_id, 0, sizeof(int) * num_edges);
  memset(edge_weight, 0, sizeof(double) * num_edges);
  edge_type = (int *)malloc(num_edges*sizeof(int));
  for (int64 i = 0; i < num_edges; i++) edge_type[i] = 0;
  vertex = (struct Vertex *)calloc(max_num_vertices, sizeof(struct Vertex));

  // edge type screening
  if (valid_paths.size() > 0) {
    path_screening = true;
    for (vector<string>::const_iterator it = valid_paths.begin(); it != valid_paths.end();
        it ++) {
      valid_paths_set.insert(*it);
    }
  }

  // load edge and vertex
  fin = fopen(conf_p->network_file.c_str(), "rb");
  num_vertices = 0;
  for (int64 k = 0; k != num_edges; k++) {
    type_name[0] = '\0';
    fgets(line_buffer, sizeof(line_buffer), fin);
    if (num_separator == 0) {
      // read one line to find out the separator, and be consistent
      for (size_t i = 0; i < MAX_LINE_LEN; i++) {
        if (line_buffer[i] == '\0') break;
        else if (line_buffer[i] == ' ' || line_buffer[i] == '\t') num_separator++;
      }
    }
    if (num_separator == 2) {
      sscanf(line_buffer, "%s %s %lf", name_v1, name_v2, &weight);
    }
    else if (num_separator == 3) {
      sscanf(line_buffer, "%s %s %lf %s", name_v1, name_v2, &weight, type_name);
    }
    else {
      printf("ERROR: separator mis-match, check network file format..\n");
      exit(-1);
    }

    /* edge type screening */
    bool go = false;
    // expecting the set finder to achieve better results with much larger paths
    // if (!path_screening || valid_paths_set.find(type_name) != valid_paths_set.end())
    //    go = true;
    if (path_screening) {
      for (vector<string>::const_iterator it = valid_paths.begin(); it != valid_paths.end();
          it ++) {
        if (strcmp(type_name, it->c_str()) == 0) {
          go = true;
          break;
        }
      }
    } else {
      go = true;
    }
    if (!go) {
      // still add the vertex
      vid = search_hash_table(name_v1, vertex);
      if (vid == -1) vid = add_vertex(name_v1, vertex, num_vertices, max_num_vertices);
      vid = search_hash_table(name_v2, vertex);
      if (vid == -1) vid = add_vertex(name_v2, vertex, num_vertices, max_num_vertices);
      k--;
      num_edges--;
      continue;
    }

    if (k % 10000 == 0) {
      printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
      fflush(stdout);
    }

    vid = search_hash_table(name_v1, vertex);
    if (vid == -1) vid = add_vertex(name_v1, vertex, num_vertices, max_num_vertices);
    vertex[vid].degree += weight;
    edge_source_id[k] = vid;

    vid = search_hash_table(name_v2, vertex);
    if (vid == -1) vid = add_vertex(name_v2, vertex, num_vertices, max_num_vertices);
    vertex[vid].degree += weight;
    edge_target_id[k] = vid;

    edge_weight[k] = weight;

    if (type_name[0] != '\0') {
      if (edge_name2type.find(type_name) == edge_name2type.end()) {
        type = num_edge_type++;
        edge_name2type[type_name] = type;
        edge_type2name[type] = type_name;
      } else {
        type = edge_name2type[type_name];
      }
      edge_type[k] = type;
    }
  }
  fclose(fin);
  printf("Number of (unique) edges (in edge file): %lld          \n", num_edges);
  printf("Number of nodes (in edge file): %d          \n", num_vertices);

  // edge weight computation
  edge_type_w = new double[num_edge_type];
  memset(edge_type_w, 0, sizeof(double) * num_edge_type);
  for (int64 i = 0; i < num_edges; i++) edge_type_w[edge_type[i]] += edge_weight[i];

  /* processing node type, edge type */
  vertex_type = (int *)calloc(max_num_vertices, sizeof(int));
  for (int i = 0; i < num_vertices; i++) {
    vertex[i].type = 0;
    vertex_type[i] = 0;
  }
  num_node_type = 1;
  if (num_edge_type > 0) {
    printf("Number of edge type: %d          \n", num_edge_type);
    printf("[edge_name2type] edge type name => type index, edge_type_w\n");
    for (map<string, int>::const_iterator it = edge_name2type.begin();
        it != edge_name2type.end(); it++) {
      printf("\t%s => %d, %f\n", it->first.c_str(), it->second, edge_type_w[it->second]);
    }
  } else {
    num_edge_type = 1;
    printf("Number of edge type: %d          \n", num_edge_type);
    printf("[edge_name2type/edge_type2name not presented]\n");
  }

  node_type_to_edge_type = new bool[num_node_type * num_edge_type];
  for (int i = 0; i < num_node_type; i++) {
    int i_row_start = i * num_edge_type;
    for (int j = 0; j < num_edge_type; j++) {
      node_type_to_edge_type[i_row_start + j] = 1;
    }
  }

  // re-order edge type configuration with path conf
  if (conf_p->use_path_conf) {
    vector<float> path_weight;
    vector<int> path_direction;
    vector<int> path_order;
    vector<float> path_sampling_pow;
    vector<float> path_base_deg;
    for (int i = 0; i < num_edge_type; i++) {
      string path_name = edge_type2name[i];
      size_t j;
      for (j = 0; j < valid_paths.size(); j++) {
        if (valid_paths[j] == path_name)
          break;
      }
      assert(j != valid_paths.size());
      path_weight.push_back(conf_p->path_weight[j]);
      path_direction.push_back(conf_p->path_direction[j]);
      path_order.push_back(conf_p->path_order[j]);
      path_sampling_pow.push_back(conf_p->path_sampling_pow[j]);
      path_base_deg.push_back(conf_p->path_base_deg[j]);
    }
    conf_p->path_weight = path_weight;
    conf_p->path_direction = path_direction;
    conf_p->path_order = path_order;
    conf_p->path_sampling_pow = path_sampling_pow;
    conf_p->path_base_deg = path_base_deg;

    printf("[Edge type configurations] edge_type weight direction proximity/order sampling_pow base_deg\n");
    for (int i = 0; i < num_edge_type; i++) {
      printf("\t%s %f %d %d %f %f\n", edge_type2name[i].c_str(), conf_p->path_weight[i],
        conf_p->path_direction[i], conf_p->path_order[i], conf_p->path_sampling_pow[i], conf_p->path_base_deg[i]);
    }
  }

  // re-weighting each row for each network
  if (conf_p->row_reweighting) {
    double **_edge_type_degree = new double*[num_edge_type];
    for (int i = 0; i < num_edge_type; i++) {
      _edge_type_degree[i] = new double[num_vertices];
      memset(_edge_type_degree[i], 0, sizeof(double) * num_vertices);
    }
    for (int64 i = 0; i < num_edges; i++) {
      _edge_type_degree[edge_type[i]][edge_source_id[i]] += edge_weight[i];
    }
    for (int64 i = 0; i < num_edges; i++) {
      double deg = _edge_type_degree[edge_type[i]][edge_source_id[i]];
      edge_weight[i] = pow(edge_weight[i] / deg, NEG_SAMPLING_POWER);  // reweighting function
    }
    for (int i = 0; i < num_edge_type; i++) delete []_edge_type_degree[i];
  }

  // normalize over networks
  // only the relative values (instead of absolute values) of each meta-path matter
  bool use_path_conf = conf_p->use_path_conf;
  double path_sum = conf_p->path_sum_default;
  if (conf_p->path_normalization) {
    double *_edge_type_w = new double[num_edge_type];
    memset(_edge_type_w, 0, sizeof(double) * num_edge_type);
    for (int64 i = 0; i < num_edges; i++) _edge_type_w[edge_type[i]] += edge_weight[i];
    for (int64 i = 0; i < num_edges; i++) {
      int etype = edge_type[i];
      if (use_path_conf) path_sum = conf_p->path_weight[etype];
      edge_weight[i] *= path_sum / _edge_type_w[etype];
    }
    delete [] _edge_type_w;
    if (use_path_conf)
      printf("each meta-path edge weight is normalized to pre-define\n");
    else
      printf("each meta-path edge weight is normalized to %f\n", path_sum);
  }

  // compute vertex_degree_of_etype
  vertex_degree_of_etype = new double[num_vertices * num_edge_type];
  memset(vertex_degree_of_etype, 0, sizeof(double) * num_vertices * num_edge_type);
  for (int64 i = 0; i < num_edges; i++) {
    int src = edge_source_id[i];
    int dst = edge_target_id[i];
    int e_type = edge_type[i];
    double w = edge_weight[i];
    vertex_degree_of_etype[src * num_edge_type + e_type] += w;
    vertex_degree_of_etype[dst * num_edge_type + e_type] += w;
  }

  graph.vertex = vertex;
  graph.vertex_type = vertex_type;
  graph.vertex_degree_of_etype = vertex_degree_of_etype;
  graph.edge_source_id = edge_source_id;
  graph.edge_target_id = edge_target_id;
  graph.edge_weight = edge_weight;
  graph.edge_type = edge_type;
  graph.edge_type_w = edge_type_w;
  graph.node_type_to_edge_type = node_type_to_edge_type;
  graph.num_vertices_p = &num_vertices;
  graph.num_edges_p = &num_edges;
  graph.num_node_type_p = &num_node_type;
  graph.num_edge_type_p = &num_edge_type;

  end = clock();
  printf("network loaded in %.2f (s)\n", (double)(end-start) / CLOCKS_PER_SEC);
}

void DataHelper::load_node_type() {
  char line_buffer[MAX_LINE_LEN], node_name[MAX_STRING], type_name[MAX_STRING];
  int vid, type;
  int *num_node_in_type;
  int vertex_type_size = max_num_vertices;
  FILE *fin = fopen(conf_p->node_type_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: node type file not exist..\n");
    exit(-1);
  }
  num_node_type = 0;
  while (fgets(line_buffer, sizeof(line_buffer), fin)) {
    sscanf(line_buffer, "%s %s", node_name, type_name);
    vid = search_hash_table(node_name, vertex);
    if (vid == -1) {
      vid = add_vertex(node_name, vertex, num_vertices, max_num_vertices);
      vertex[vid].degree = 0;
      if (vertex_type_size < max_num_vertices) {
        vertex_type_size = max_num_vertices;
        vertex_type = (int *)realloc(vertex_type, vertex_type_size * sizeof(int));
      }
    }
    // if (vid == -1) continue;  // debug
    if (node_name2type.find(type_name) == node_name2type.end()) {
      type = num_node_type++;
      node_name2type[type_name] = type;
      node_type2name[type] = type_name;
    } else {
      type = node_name2type[type_name];
    }
    vertex[vid].type = type;
    vertex_type[vid] = -1; // mark as loaded
  }
  for (int i = 0; i < num_vertices; i++) {
    // make sure every node has a type now
    assert(vertex_type[i] == -1);
    vertex_type[i] = vertex[i].type;
  }
  num_node_in_type = new int[num_node_type];
  memset(num_node_in_type, 0, sizeof(int) * num_node_type);
  for (int i = 0; i < num_vertices; i++)
    num_node_in_type[vertex_type[i]]++;
  printf("Number of nodes (in node file): %d          \n", num_vertices);
  printf("Number of node type: %d          \n", num_node_type);
  printf("[node_name2type] node type name => type index, num_node_in_type\n");
  for (map<string, int>::iterator it = node_name2type.begin(); it != node_name2type.end(); it ++)
    printf("\t%s => %d, %d\n", it->first.c_str(), it->second, num_node_in_type[it->second]);
  fclose(fin);

  reload_node_type_to_edge_type();

  graph.vertex = vertex;
  graph.vertex_type = vertex_type;

  delete [] num_node_in_type;
}

void DataHelper::reload_node_type_to_edge_type(bool printing) {
  if (node_type_to_edge_type != NULL) {
    delete []node_type_to_edge_type;
  }
  node_type_to_edge_type = new bool[num_node_type * num_edge_type];
  memset(node_type_to_edge_type, 0, sizeof(bool) * num_node_type * num_edge_type);
  graph.node_type_to_edge_type = node_type_to_edge_type;

  // set to 1/true if any connectivity from the node type to the edge type is observed in network
  for (int64 i = 0; i < num_edges; i++) {
    int src, dst, e_type, src_type, dst_type;
    src = edge_source_id[i];
    dst = edge_target_id[i];
    src_type = vertex_type[src];
    dst_type = vertex_type[dst];
    e_type = edge_type[i];
    node_type_to_edge_type[src_type * num_edge_type + e_type] = 1;
    node_type_to_edge_type[dst_type * num_edge_type + e_type] = 1;
  }

  // print the connectivity schema
  if (printing) {
    printf("Node type to edge type schema in network: \n");
    for (int i = 0; i < num_node_type; i++) {
      int i_row_start = i * num_edge_type;
      for (int j = 0; j < num_edge_type; j++) {
        if (node_type_to_edge_type[i_row_start + j]) {
          string node_type_name("-");
          string edge_type_name("-");
          if (node_type2name.find(i) != node_type2name.end())
            node_type_name = node_type2name[i];
          if (edge_type2name.find(j) != edge_type2name.end())
            edge_type_name = edge_type2name[j];
          printf("\t%s ~~> %s\n",node_type_name.c_str(), edge_type_name.c_str());
        }
      }
    }
  }
}

void DataHelper::load_pa_trainortest(string pa_file, string po_file, bool is_training) {
  char name_v1[MAX_STRING], name_v2[MAX_STRING], line_buffer[MAX_LINE_LEN];
  int src, dst;
  int64 num_lines = 0;
  float label = 0;
  int max_non_vertices_id = -1;
  time_t start, end;
  start = clock();

  string                  trainortest_name;
  // vector<int>             *group, *target_group;
  vector<pair<int, int> > *pairs;
  vector<real>            *pairs_label;
  map<int, vector<int> >  *src_features, *dst_features;

  if (is_training) {
    trainortest_name = "train";
    pairs = &train_pairs;
    pairs_label = &train_pairs_label;
    src_features = &train_src_features;
    dst_features = &train_dst_features;
  } else {
    trainortest_name = "test";
    pairs = &test_pairs;
    pairs_label = &test_pairs_label;
    src_features = &test_src_features;
    dst_features = &test_dst_features;
  }

  FILE *fin = fopen(pa_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: %s p2a file not found!\n", trainortest_name.c_str());
    exit(-1);
  }
  while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;

  // load paper to authors candidate train or test pairs
  fin = fopen(pa_file.c_str(), "rb");
  pairs->reserve(num_lines);
  pairs_label->reserve(num_lines);
  for (int64 i = 0; i != num_lines; i++) {
    fscanf(fin, "%s %s %f", name_v1, name_v2, &label);
    if (i % 10000 == 0) {
      printf("Reading %s_p2a lines: %.3lf%%%c", trainortest_name.c_str(),
        i / (double)(num_lines + 1) * 100, 13);
      fflush(stdout);
    }

    // if (is_training && label == 0) continue;  // debug
    if (non_vertex_name2id.find(name_v1) == non_vertex_name2id.end()) {
      src = ++max_non_vertices_id;
      non_vertex_name2id[name_v1] = src;
      non_vertex_id2name[src] = name_v1;
    } else {
      src = non_vertex_name2id.at(name_v1);
    }
    dst = search_hash_table(name_v2, vertex);
    assert(dst != -1);
    pairs->push_back(make_pair(src, dst));
    // if (is_training) assert(label > 0);  // only positive pairs are given in training
    pairs_label->push_back(label);
  }
  assert(pairs->size() == pairs_label->size());
  fclose(fin);
  printf("Number of %s p2a pairs: %ld          \n", trainortest_name.c_str(), pairs->size());

  num_lines = 0;
  fin = fopen(po_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: %s p2o file not found!\n", trainortest_name.c_str());
    exit(-1);
  }
  while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;

  // load paper to features
  int count_unseen_features = 0;
  fin = fopen(po_file.c_str(), "rb");
  for (int64 i = 0; i != num_lines; i++) {
    fscanf(fin, "%s %s %f", name_v1, name_v2, &label);
    if (i % 10000 == 0) {
      printf("Reading %s_p2o lines: %.3lf%%%c", trainortest_name.c_str(),
        i / (double)(num_lines + 1) * 100, 13);
      fflush(stdout);
    }
    // map src and dst nodes
    if (non_vertex_name2id.find(name_v1) == non_vertex_name2id.end()) {
      src = ++max_non_vertices_id;
      non_vertex_name2id[name_v1] = src;
      non_vertex_id2name[src] = name_v1;
    } else {
      src = non_vertex_name2id.at(name_v1);
    }
    dst = search_hash_table(name_v2, vertex);
    if (dst == -1 && !is_training) {
      count_unseen_features++;
      continue; // debug, ignore all features only appear in test, could be the new year, hazard
    }
    assert(dst != -1);

    // insert features, taking more than half of the loading time
    /* this is real slow
    if (src_features->find(src) != src_features->end()) {
      src_features->at(src).push_back(dst);
    } else {
      // the key does not exist in the map
      vector<int> f_vec(dst);
      src_features->insert(make_pair(src, f_vec));
    }
    */
    int k = src;
    int f = dst;
    map<int, vector<int> >::iterator lb = src_features->lower_bound(k);

    if(lb != src_features->end() && !(src_features->key_comp()(k, lb->first))) {
      // key already exists
      lb->second.push_back(f);
    }
    else {
      // the key does not exist in the map
      vector<int> f_vec;
      f_vec.push_back(f);
      src_features->insert(lb, map<int, vector<int> >::value_type(k, f_vec));
    }
  }
  fclose(fin);
  if (count_unseen_features > 0)
    printf("[WARNING!!!!!] There are %d unseen features in test feature file. Please check!\n", count_unseen_features);

  /* to delete debug
  for (map<int, vector<int> >::iterator it = src_features->begin(); it != src_features->end(); it ++) {
    printf("%s:\t", non_vertex_id2name[it->first].c_str());
    for (vector<int>::iterator jt = it->second.begin(); jt != it->second.end(); jt ++) {
      printf("\t%s", vertex[*jt].name);
    }
    printf("\n");
  }
  */

  // add author to features
  // also make sure all papers in pairs should have features
  for (vector<pair<int, int> >::const_iterator it = pairs->begin(); it != pairs->end();
      ++it) {
    int paper = it->first;
    int author = it->second;
    if (dst_features->find(author) == dst_features->end()) {
      vector<int> a_vec;
      a_vec.push_back(author);
      (*dst_features)[author] = a_vec;
    }
    assert(src_features->find(paper) != src_features->end());
  }

  end = clock();
  if (is_training)
    printf("train target&feature loaded in %.2f (s)\n", (double)(end-start) / CLOCKS_PER_SEC);
  else
    printf("test target&feature loaded in %.2f (s)\n", (double)(end-start) / CLOCKS_PER_SEC);

  // _check_pa_test_data();
}

void DataHelper::construct_group(bool test_only) {
  int prev_src, cur_size, i;

  if (!test_only) {
    // train group
    printf("Constructing train group from train pairs...\r");
    fflush(stdout);
    prev_src = train_pairs[0].first;
    cur_size = 1, i = 0;
    for (vector<pair<int, int> >::const_iterator it = train_pairs.begin() + 1; it != train_pairs.end();
        ++it) {
      i++;
      int src = it->first;
      if (src == prev_src) {
        cur_size++;
      } else {
        train_group.push_back(cur_size);
        cur_size = 1;
        prev_src = src;
      }
    }
    train_group.push_back(cur_size);
  }

  // test_group
  printf("Constructing test group from test pairs...\r");
  fflush(stdout);
  prev_src = test_pairs[0].first;
  cur_size = 1, i = 0;
  string prev_type_name;
  bool set_task_start_end = test_pairs_type.size() > 0? true: false;
  if (set_task_start_end) {
    prev_type_name = test_pairs_type[0];
    test_task_group_start_end[prev_type_name] = make_pair(0, -1);
  }
  for (vector<pair<int, int> >::const_iterator it = test_pairs.begin() + 1; it != test_pairs.end();
      ++it) {
    i++;
    int src = it->first;
    if (src == prev_src) {
      cur_size++;
    } else {
      test_group.push_back(cur_size);
      cur_size = 1;
      prev_src = src;
    }
    if (set_task_start_end) {
      string &cur_type_name = test_pairs_type[i];
      if (cur_type_name != prev_type_name) {
        if (cur_size != 1) printf("%d\t%d\n", i, cur_size);
        assert(cur_size == 1);  // task switch can only occur at the same time as group switch
        test_task_group_start_end[prev_type_name].second = test_group.size();
        // make sure the task types' continuity by assuring it never appears before
        assert(test_task_group_start_end.find(cur_type_name) == test_task_group_start_end.end());
        test_task_group_start_end[cur_type_name] = make_pair(test_group.size(), -1);
        prev_type_name = cur_type_name;
      }
    }
  }
  test_group.push_back(cur_size);
  if (set_task_start_end) test_task_group_start_end[prev_type_name].second = test_group.size();

  printf("Number of test groups %ld                  \n", test_group.size());

  /* test and print test_task_group_start_end */
  if (set_task_start_end) {
    printf("[test tasks]\n");
    for (map<string, pair<int, int> >::const_iterator it = test_task_group_start_end.begin();
        it != test_task_group_start_end.end(); it ++) {
      printf("\t%s, start group: %d, end group %d\n", it->first.c_str(),
        it->second.first,  it->second.second);
    }
  }

  printf("Done constructing group                              \n");
}

void DataHelper::load_test(string test_file) {
  char name_v1[MAX_STRING], name_v2[MAX_STRING], line_buffer[MAX_LINE_LEN], type_name[MAX_STRING];
  int src, dst, num_separator = 0, skip = 0;
  int64 num_lines = 0;
  float label = 0;
  bool has_missing_test_etype = false;

  // saving real test into file
  bool save_real_test = false;
  FILE *fo = NULL;
  if (save_real_test) {
    printf("[INFO] saving test real to file..\n");
    string real_test_file("test.txt.regular_real");
    fo = fopen(real_test_file.c_str(), "wb");
    assert(fo != NULL);
  }

  FILE *fin = fopen(test_file.c_str(), "rb");
  if (fin == NULL) {
    printf("ERROR: test file not found!\n");
    exit(-1);
  }
  while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;
  test_pairs.reserve(num_lines);
  test_pairs_label.reserve(num_lines);

  fin = fopen(test_file.c_str(), "rb");
  for (int64 i = 0; i != num_lines; i++) {
    type_name[0] = '\0';
    fgets(line_buffer, sizeof(line_buffer), fin);
    if (num_separator == 0) {
      // read one line to find out the separator, and be consistent
      for (size_t i = 0; i < MAX_LINE_LEN; i++) {
        if (line_buffer[i] == '\0') break;
        else if (line_buffer[i] == ' ' || line_buffer[i] == '\t') num_separator++;
      }
    }
    if (num_separator == 2) {
      sscanf(line_buffer, "%s %s %f", name_v1, name_v2, &label);
    } else if (num_separator == 3) {
      sscanf(line_buffer, "%s %s %f %s", name_v1, name_v2, &label, type_name);
    } else {
      printf("ERROR: separator mis-match, check test file format..\n");
      exit(-1);
    }
    if (i % 10000 == 0) {
      printf("Reading test lines: %.3lf%%%c", i / (double)(num_lines + 1) * 100, 13);
      fflush(stdout);
    }

    src = search_hash_table(name_v1, vertex);
    if (src == -1) {skip++; continue;}  // debug, not big deal, just ignore unseen nodes
    // if (src == -1) printf("%s\n", name_v1);
    // assert(src != -1);
    dst = search_hash_table(name_v2, vertex);
    if (dst == -1) {skip++; continue;}
    // if (dst == -1) printf("%s\n", name_v2);
    // assert(dst != -1);

    test_pairs.push_back(make_pair(src, dst));
    test_pairs_label.push_back(label);
    if (type_name[0] != '\0') {
      test_pairs_type.push_back(string(type_name));
      int e_type = -1;
      if (edge_name2type.find(type_name) != edge_name2type.end()) {
        e_type = edge_name2type[type_name];
      }
      if (e_type == -1) {
        has_missing_test_etype = true;
        test_pairs_etype.push_back(-1);
      } else {
        test_pairs_etype.push_back(e_type);
      }
    }

    if (save_real_test) {
      fprintf(fo, "%s\t%s\t%f\t%s\n", name_v1, name_v2, label, type_name);
    }
  }

  if (has_missing_test_etype)
    printf("[WARNING!!!!!!!!!!] There are unknown edge type in test pairs.\n");

  printf("Number of test pairs: %ld          \n", test_pairs.size());
  if (skip > 0)
    printf("[WARNING!!!!!!!!] %d test points are skipped due to node is not in training.\n", skip);

  // save real test file
  if (save_real_test) {
    fclose(fo);
    exit(-1);
  }

  assert(test_pairs.size() == test_pairs_label.size());
  fclose(fin);
}

void DataHelper::_check_pa_test_data() {
  printf("\nChecking test data..\n");
  int i = -1;
  for (vector<pair<int, int> >::const_iterator it = test_pairs.begin(); it != test_pairs.end();
      ++it) {
    i++;
    int src = it->first;
    int dst = it->second;
    float label = test_pairs_label[i];
    printf("%s %s %.2f\n", non_vertex_id2name.at(src).c_str(), vertex[dst].name, label);
  }

  for (map<int, vector<int> >::const_iterator it = test_src_features.begin();
      it != test_src_features.end(); ++it) {
    const int &paper = it->first;
    const vector<int> &features = it->second;
    printf("%s:", non_vertex_id2name.at(paper).c_str());
    for (size_t i = 0; i < features.size(); i++) {
      printf("\t%s", vertex[features[i]].name);
    }
    printf("\n");
  }
}

