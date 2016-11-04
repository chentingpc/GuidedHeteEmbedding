#include "supf_model.h"

/**********************************************
 * Work scheduler and helpers
 *********************************************/

void SupervisedFeatureModel::fit() {
  time_t start, end;
  time(&start);
  int num_train_thread, num_eval_thread = 1, a;
  num_train_thread = conf_p->num_train_threads;
  if (num_train_thread == 0) printf("[WARNING!] num_train_thread is set to zero!!!!!!!!!!!!!!\n");

  pthread_t *pt = (pthread_t *)malloc((num_train_thread + num_eval_thread) * sizeof(pthread_t));
  Context *context[num_train_thread + num_eval_thread];
  for (a = 0; a < num_train_thread; a++) {
    context[a] = new Context;
    context[a]->model_ptr = this;
    context[a]->id = a;
    pthread_create(&pt[a], NULL, train_helper, (void *)(context[a]));
  }

  for (a = num_train_thread; a < num_train_thread + num_eval_thread; a++) {
    context[a] = new Context;
    context[a]->model_ptr = this;
    context[a]->id = conf_p->map_topk;
    pthread_create(&pt[a], NULL, eval_helper, (void *)(context[a]));
  }

  EmbeddingModel::fit();  // cant put before as it has inside waiting blocking until fitting finished.

  for (a = 0; a < num_train_thread + num_eval_thread; a++) {
    pthread_join(pt[a], NULL);
    free(context[a]);
  }

  time(&end);
  printf("Supf_model training finished in %ld seconds.\n", (end - start));
}

void SupervisedFeatureModel::eval_thread(int map_topk) {
  int eval_cnt = 100;  // total number to evaluate during the training
  int64 eval_per_sample = total_samples / eval_cnt;
  int64 eval_cur_sample = 0;
  while (true) {
    if (current_sample_count >= eval_cur_sample) {
      eval(map_topk);
      while (current_sample_count >= eval_cur_sample)
        eval_cur_sample += eval_per_sample;
    } else if (eval_cur_sample >= total_samples) {
      eval(map_topk);  // last eval before exit
      break;
    } else {
      sleep(1);  // sleep for seconds before checking to eval again
    }
  }
}

void SupervisedFeatureModel::eval(int map_topk) {
  const vector<real> &test_pairs_label = *test_pairs_label_p;
  predict();
  real mapk = eval_mapk(test_pairs_label, pred, *test_group_p, map_topk);
  real prog = (real)current_sample_count / (real)(total_samples + 1);
  printf("# Prog: %.2lf%%, test map@%d: %f \n", prog * 100, map_topk, mapk);
}

void SupervisedFeatureModel::save(string embedding_file, bool is_binary, string pred_file) {
  // save model
  EmbeddingModel::save(embedding_file, is_binary);

  // save prediction
  if (pred_file.size() == 0) {
    printf("[WARNING] pred_file not saved due to no path given.\n");
    return;
  }
  printf("[INFO] saving prediction to file..\n");
  FILE *fo = fopen(pred_file.c_str(), "wb");
  assert(fo != NULL);
  for (size_t i = 0; i < pred.size(); i++) {
    // const pair<int, int> &the_test_pair = (*test_pairs_p)[i];  // debug
    const pair<int, int> &the_test_pair = test_pairs_dup[i];
    fprintf(fo, "%s\t%s\t%f\t%f\n", non_vertex_id2name->at(the_test_pair.first).c_str(),  // paper using input id
      vertex[the_test_pair.second].name, (*test_pairs_label_p)[i], pred[i]);
  }
  fclose(fo);

  // save node weights
  return;  // debug
  printf("[INFO] saving node weights to file..\n");
  string weight_file(pred_file);
  weight_file += ".node_weights";
  fo = fopen(weight_file.c_str(), "wb");
  assert(fo != NULL);
  for (int i = 0; i < num_vertices; i++) {
    fprintf(fo, "%s\t%f\n", vertex[i].name, weights_node[i]);
  }
  fclose(fo);
}


/**********************************************
 * Initializations
 *********************************************/

void SupervisedFeatureModel::init_variables() {
  train_group_p = data_helper->get_train_group();
  train_pairs_p = data_helper->get_train_pairs();
  train_pairs_label_p = data_helper->get_train_pairs_label();
  train_src_features_p = data_helper->get_train_src_features();
  train_dst_features_p = data_helper->get_train_dst_features();
  if (conf_p->train_percent > 0 && conf_p->train_percent < 1)
    _down_sample_train_data(conf_p->train_percent);
  else if (conf_p->train_percent != 0 && conf_p->train_percent != 1) {
    // zero will be taken care of below, 1 can be ignored
    printf("[ERROR!] invalid train_percent %f\n", conf_p->train_percent);
    exit(0);
  }

  test_group_p = data_helper->get_test_group();
  test_pairs_p = data_helper->get_test_pairs();
  test_pairs_label_p = data_helper->get_test_pairs_label();
  test_src_features_p = data_helper->get_test_src_features();
  test_dst_features_p = data_helper->get_test_dst_features();

  pred.resize(test_pairs_p->size());
  test_pairs_dup = *test_pairs_p;

  dropout = conf_p->supf_dropout;
  if (dropout > 0) {
    for (map<int, vector<int> >::const_iterator it = train_src_features_p->begin();
        it != train_src_features_p->end(); it++) {
      int feature_size = it->second.size();
      if (feature_size > max_src_feat_size)
        max_src_feat_size = feature_size;
    }
    printf("[INFO!] dropout %f, max_src_feat_size is %d.\n", dropout, max_src_feat_size);
  } else printf("[INFO!] dropout is off.\n");

  if (conf_p->train_percent == 0) {
    init_lr_supf_emb = lr_supf_emb = 0;
    init_lr_supf_ntype_w = lr_supf_ntype_w = 0;
    init_lr_supf_nbias = lr_supf_nbias = 0;
  }
  else {
    init_lr_supf_emb = lr_supf_emb = conf_p->lr_supf_emb;
    init_lr_supf_ntype_w = lr_supf_ntype_w = conf_p->lr_supf_ntype_w;
    init_lr_supf_nbias = lr_supf_nbias = conf_p->lr_supf_nbias;
  }
}

void SupervisedFeatureModel::init_vector() {
  bias_node = new real[num_vertices];
  for (int i = 0; i < num_vertices; i ++) bias_node[i] = 0.;
  /*
  for (vector<pair<int, int> >::const_iterator it = train_pairs_p->begin();
      it != train_pairs_p->end(); ++it) {
    int src = it->first;
    int dst = it->second;
    bias_node[dst] += 1;
  }
  */

  weights_node = new real[num_vertices];
  for (int i = 0; i < num_vertices; i ++) weights_node[i] = 1.;

  weights_node_type = new real[num_node_type];
  for (int i = 0; i < num_node_type; i++) weights_node_type[i] = 1.;
}

void SupervisedFeatureModel::init_runtime() {
  _build_feature_node_type_cnt(train_src_features_p, train_src_feat_has_node_type,
                               train_src_feat_node_type_cnt);
  _build_feature_node_type_cnt(train_dst_features_p, train_dst_feat_has_node_type,
                               train_dst_feat_node_type_cnt);
  _build_feature_node_type_cnt(test_src_features_p, test_src_feat_has_node_type,
                               test_src_feat_node_type_cnt);
  _build_feature_node_type_cnt(test_dst_features_p, test_dst_feat_has_node_type,
                               test_dst_feat_node_type_cnt);

  printf("train_src_feat_has_node_type:");
  for (int i = 0; i < num_node_type; i++) printf("\t%d", train_src_feat_has_node_type[i]);
  printf("\n");
  printf("train_dst_feat_has_node_type:");
  for (int i = 0; i < num_node_type; i++) printf("\t%d", train_dst_feat_has_node_type[i]);
  printf("\n");
  printf("test_src_feat_has_node_type:");
  for (int i = 0; i < num_node_type; i++) printf("\t%d", test_src_feat_has_node_type[i]);
  printf("\n");
  printf("test_dst_feat_has_node_type:");
  for (int i = 0; i < num_node_type; i++) printf("\t%d", test_dst_feat_has_node_type[i]);
  printf("\n");

  negative_by_sampling = int(conf_p->supf_negative_by_sampling * NEG_BY_SAMPLING_TOP);

  // prepare train_pairs_pos
  int i = -1;
  for (vector<real>::const_iterator it = train_pairs_label_p->begin();
      it != train_pairs_label_p->end(); it++) {
    i++;
    if (*it > 0) train_pairs_pos.push_back(i);
  }

  if (negative_by_sampling > 0) {
    // setup node sampler for authors
    const vector<pair<int, int> > *pairs_p = NULL;
    NodeSampler **_author_sampler = NULL;
    const vector<real> *pairs_label_p = NULL;
    int step_cont = 0;
    _construct_sampler:
    if (step_cont > 0) {
      // compute author degree & construct negative sampler
      int author_vtype = vertex_type[(*pairs_p)[0].second];
      double *author_dist = new double[num_vertices];
      memset(author_dist, 0, sizeof(double) * (num_vertices));
      double base_deg = conf_p->supf_neg_base_deg;
      for (int i = 0; i < num_vertices; i++) {
        if (vertex_type[i] == author_vtype) {
          author_dist[i] += base_deg;
        }
      }
      int i = -1;
      for (vector<pair<int, int> >::const_iterator it = pairs_p->begin();
          it != pairs_p->end(); it++) {
        i++;
        if ((*pairs_label_p)[i] <= 0) continue;  // only keep pos pairs
        author_dist[it->second]++;
      }
      float neg_sampling_power = conf_p->supf_neg_sampling_pow;
      for (int i = 0; i < num_vertices; i++)
        if (author_dist[i] > 0) author_dist[i] = pow(author_dist[i], neg_sampling_power);
      *_author_sampler = new NodeSampler(author_dist, num_vertices);
      delete []author_dist;
    }
    if (step_cont == 0) {
      pairs_p = train_pairs_p;
      pairs_label_p = train_pairs_label_p;
      _author_sampler = &neg_author_sampler;
      step_cont++;
      goto _construct_sampler;
    } else if (step_cont == 1) {
      pairs_p = test_pairs_p;
      pairs_label_p = test_pairs_label_p;
      _author_sampler = &test_neg_author_sampler;
      step_cont++;
      // goto _construct_sampler;  // debug, normally we don't need test_neg_author_sampler
    }
  }
  if (negative_by_sampling < NEG_BY_SAMPLING_TOP) {
    // prepare train_pairs_neg_by_src
    int i = -1;
    int64 num_neg_pairs = 0;
    for (vector<real>::const_iterator it = train_pairs_label_p->begin();
        it != train_pairs_label_p->end(); it++) {
      i++;
      if (*it != 1) {
        // neg pair
        num_neg_pairs++;
        int src = (*train_pairs_p)[i].first;
        if (train_pairs_neg_by_src.find(src) == train_pairs_neg_by_src.end()) {
          vector<int> _train_pairs_neg;
          _train_pairs_neg.push_back(i);
          train_pairs_neg_by_src[src] = _train_pairs_neg;
        } else {
          train_pairs_neg_by_src[src].push_back(i);
        }
      }
    }
    // make sure all pos src has negataive samples
    for (vector<int>::const_iterator it = train_pairs_pos.begin(); it != train_pairs_pos.end();
        it++) {
      int src = (*train_pairs_p)[*it].first;
      assert(train_pairs_neg_by_src.find(src) != train_pairs_neg_by_src.end());
    }
    printf("training pos / neg pairs: %ld, %lld\n", train_pairs_pos.size(), num_neg_pairs);
  }
}

void SupervisedFeatureModel::_build_feature_node_type_cnt(const map<int, vector<int> > *features_p,
    bool *&feat_has_node_type, int *&feat_node_type_cnt) {
  // init
  int max_key_p1 = features_p->rbegin()->first + 1;  // get the largest key (can be larger than num_vertices)
  feat_has_node_type = new bool[num_node_type];
  feat_node_type_cnt = new int[num_node_type * max_key_p1];
  memset(feat_has_node_type, 0, sizeof(bool) * num_node_type);
  memset(feat_node_type_cnt, 0, sizeof(int) * num_node_type * max_key_p1);

  // count it
  for (map<int, vector<int> >::const_iterator it = features_p->begin();
      it != features_p->end(); it++) {
    int n = it->first;
    assert(n < max_key_p1);
    const vector<int> &the_features = it->second;
    for (vector<int>::const_iterator jt = the_features.begin(); jt < the_features.end(); jt++) {
      int n_type = vertex_type[*jt];
      feat_has_node_type[n_type] = true;
      feat_node_type_cnt[n * num_node_type + n_type]++;
    }
  }
}

void SupervisedFeatureModel::_down_sample_train_data(real train_percent) {
  printf("[Warning!!!!!!!!!] Downsampling the training data with keep rate %f.\n", train_percent);
  assert(train_percent > 0 && train_percent <= 1);
  int num_group = train_group_p->size();

  printf("num of train_group old: %d\n", num_group);

  int group_start, group_end = 0;
  for (int i = 0; i < num_group; i++) {
    int glen = (*train_group_p)[i];
    group_start = group_end;
    group_end = group_start + glen;

    double p = rand() / (double) RAND_MAX;
    if (p > train_percent)
      continue;

    train_group_new.push_back(glen);
    for (int j = group_start; j < group_end; j++) {
      train_pairs_new.push_back((*train_pairs_p)[j]);
      train_pairs_label_new.push_back((*train_pairs_label_p)[j]);
    }
  }

  printf("num of train_group new %ld\n", train_group_new.size());

  train_group_p = &train_group_new;
  train_pairs_p = &train_pairs_new;
  train_pairs_label_p = &train_pairs_label_new;
}

/**********************************************
 * Training functions
 *********************************************/

void SupervisedFeatureModel::_fetch_train_triple(int &src, int &dst_pos, int &dst_neg,
                                                 uint &r_seed, uint64 &n_seed, bool sample_pos) {
  if (sample_pos) {
    int idx_pos = train_pairs_pos[rand_r(&r_seed) % train_pairs_pos.size()];
    src = (*train_pairs_p)[idx_pos].first;
    dst_pos = (*train_pairs_p)[idx_pos].second;
  }
  if (rand_r(&r_seed) % NEG_BY_SAMPLING_TOP < negative_by_sampling) {
    dst_neg = neg_author_sampler->sample(n_seed);
  } else {
    vector<int> &_train_pairs_neg = train_pairs_neg_by_src[src];
    int idx_neg = _train_pairs_neg[rand_r(&r_seed) % _train_pairs_neg.size()];
    // assert((*train_pairs_p)[idx_neg].first == src);  // should be removed once proved no bug here
    dst_neg = (*train_pairs_p)[idx_neg].second;
  }
}

void SupervisedFeatureModel::train_with_weight(int id) {
  if (samples_before_switch_other == 0) {
    printf("[WARNING!] turn down supf_model training..\n");
    if (id == 0) fit_not_finished = false;
    return;
  }
  int src, dst_pos, dst_neg;
  int64 count = 0, last_count = 0;
  int64 hinge_all = 0, hinge_violation = 0;
  int64 samples_task_round = 0;
  uint r_seed = clock();
  uint64 n_seed = static_cast<int64>(id);
  bool printing = true;

  // forward variables
  real score_pos, score_neg, ll = 0;
  real *src_vec = new real[dim], *dst_vec_pos = new real[dim], *dst_vec_neg = new real[dim];
  real *src_vec_int = new real[num_node_type * dim],  // integrated feature of node_type for a node
       *dst_vec_int_pos = new real[num_node_type * dim],
       *dst_vec_int_neg = new real[num_node_type * dim];
  bool *ignore_feats = NULL;
  if (dropout > 0)
    ignore_feats = new bool[max_src_feat_size];

  // backward variables
  real score_pos_err, score_neg_err;
  real *src_err = new real[dim], *dst_err = new real[dim];

  // training loop
  while (current_sample_count < total_samples) {
    static const int count_interval = 100000;
    if (count - last_count > count_interval) {
      if (id == 0) {
        if (printing) {
          real sample_ratio = current_sample_count_supf / (real)current_sample_count;
          real prog = (real)current_sample_count / (real)(total_samples + 1);
          printf("$ Prog: %.2lf%%, supf_sample_ratio: %f, supf_LogL: %.4lf, hinge violation rate %.4lf\n",
            prog * 100, sample_ratio, ll / (count - last_count), hinge_violation / (double)hinge_all);
          printf("node type weights:");
          for (int i = 0; i < num_node_type; i++)
            printf("\t%s:%.2f", node_type2name[i].c_str(), weights_node_type[i]);
          printf("\n");
        }
        real prog = current_sample_count / (real)(total_samples + 1);
        lr_supf_emb = init_lr_supf_emb * (1 - prog);
        if (lr_supf_emb < init_lr_supf_emb * 0.001) lr_supf_emb = init_lr_supf_emb * 0.001;
        lr_supf_ntype_w = init_lr_supf_ntype_w * (1 - prog);
        if (lr_supf_ntype_w < init_lr_supf_ntype_w * 0.001) lr_supf_ntype_w = init_lr_supf_ntype_w * 0.001;
        lr_supf_nbias = init_lr_supf_nbias * (1 - prog);
        if (lr_supf_nbias < init_lr_supf_nbias * 0.001) lr_supf_nbias = init_lr_supf_nbias * 0.001;
      }
      int64 incremental = count - last_count;
      current_sample_count += incremental;
      current_sample_count_supf += incremental;
      last_count = count;
      ll = 0.;
    }

    // task schedule helper
    if (samples_task_round == samples_before_switch_other) {
      samples_task_round = 0;
      task_switchs_for_embedding[id] = true;
      while (task_switchs_for_embedding[id] && fit_not_finished) {
        usleep(100);
      }
    }

    // triple-based training starts below
    _fetch_train_triple(src, dst_pos, dst_neg, r_seed, n_seed);

    // forward pass, make prediction
    if (ignore_feats) {
      size_t feature_size = train_src_features_p->at(src).size();
      for (size_t i = 0; i < feature_size; i++)
        ignore_feats[i] = gsl_rand() < dropout? true: false;
    }
    _get_weighted_node_vector(src, IS_SRC_TRAIN, src_vec_int, src_vec, ignore_feats);
    _get_weighted_node_vector(dst_pos, IS_DST_TRAIN, dst_vec_int_pos, dst_vec_pos);
    _get_weighted_node_vector(dst_neg, IS_DST_TRAIN, dst_vec_int_neg, dst_vec_neg);
    score_pos = bias_node[dst_pos];
    score_neg = bias_node[dst_neg];
    for (int k = 0; k < dim; k ++) score_pos += src_vec[k] * dst_vec_pos[k];
    for (int k = 0; k < dim; k ++) score_neg += src_vec[k] * dst_vec_neg[k];

    // backward pass, derivative w.r.t. f(p,a), and update embeddings and weights
    static int objective = conf_p->supf_loss;
    if (objective == 0) {  // max-margin (maximize its negative)
      const real margin = -1.0;
      // static const real margin = -conf_p->lambda; // error
      real margin_temp = score_pos - score_neg + margin;
      // if (margin_temp >= 0) continue;  // ll += 0; score_pos_err = score_neg_err = 0
      hinge_all++;
      if (margin_temp >= 0) goto count_continue;  // nothing to update
      hinge_violation++;
      ll += margin_temp;
      score_pos_err = 1;
      score_neg_err = -score_pos_err;
    } else if (objective == 1) {  // bayesian personalized ranking
      real score = score_pos - score_neg;
      real sigmoid_temp = (*sigmoid)(score);
      ll += fast_log(sigmoid_temp+LOG_MIN);
      score_pos_err = 1 - sigmoid_temp;
      score_neg_err = -score_pos_err;
    } else {  // NCE with neg-1
      score_pos_err = 1 - (*sigmoid)(score_pos);
      score_neg_err = -(*sigmoid)(score_neg);
      ll += fast_log(1. - score_pos_err + LOG_MIN) + fast_log(1. + score_neg_err + LOG_MIN);
    }

    if (gsl_rand() < 0.5) {
      _update_fpa(src, dst_pos, score_pos_err, src_vec, src_vec_int, dst_vec_pos, dst_vec_int_pos,
                  src_err, dst_err, ignore_feats);
      _update_fpa(src, dst_neg, score_neg_err, src_vec, src_vec_int, dst_vec_neg, dst_vec_int_neg,
                  src_err, dst_err, ignore_feats);
    } else {
      _update_fpa(src, dst_neg, score_neg_err, src_vec, src_vec_int, dst_vec_neg, dst_vec_int_neg,
                  src_err, dst_err, ignore_feats);
      _update_fpa(src, dst_pos, score_pos_err, src_vec, src_vec_int, dst_vec_pos, dst_vec_int_pos,
                  src_err, dst_err, ignore_feats);
    }

    count_continue:
    count++;
    samples_task_round++;
  }

  fit_not_finished = false;
}

inline void SupervisedFeatureModel::_update_fpa(const int &src, const int &dst,
    const real &score_err, const real *src_vec, const real *src_vec_int,
    const real *dst_vec, const real *dst_vec_int, real *src_err, real *dst_err, bool *ignore_feats) {
  for (int k = 0; k < dim; k++) {
    src_err[k] = score_err * dst_vec[k];
    dst_err[k] = score_err * src_vec[k];
  }
  const int update_node_weight_or_embedding = 1;  // 1 for embedding, 0 for node weight
  // int update_node_weight_or_embedding = gsl_rand() > 0.5? 1: 0;
  _update_node_weight_or_embedding(src, IS_SRC_TRAIN, src_err, update_node_weight_or_embedding, ignore_feats);
  _update_node_weight_or_embedding(dst, IS_DST_TRAIN, dst_err, update_node_weight_or_embedding, ignore_feats);

  // update bias
  bias_node[dst] += lr_supf_nbias * score_err;

  // update weights of node types
  for (int i = 0; i < num_node_type; i++) {
    real temp = 0;
    for (int k = 0; k < dim; k++) {
      temp += src_err[k] * src_vec_int[i * dim + k] + dst_err[k] * dst_vec_int[i * dim + k];
    }
    weights_node_type[i] += lr_supf_ntype_w * temp;
  }
}

inline void SupervisedFeatureModel::_update_node_weight_or_embedding(const int &node,
    const int &is_src_train, real *pre_vec_err, int choice, bool *ignore_feats) {
  const vector<int> *features;
  int *this_feat_node_type_cnt;
  switch (is_src_train) {
    case IS_SRC_TRAIN:
      features = &train_src_features_p->at(node);
      this_feat_node_type_cnt = &train_src_feat_node_type_cnt[node * num_node_type];
      break;
    case IS_SRC_TEST:
      features = &test_src_features_p->at(node);
      this_feat_node_type_cnt = &test_src_feat_node_type_cnt[node * num_node_type];
      break;
    case IS_DST_TRAIN:
      features = &train_dst_features_p->at(node);
      this_feat_node_type_cnt = &train_dst_feat_node_type_cnt[node * num_node_type];
      break;
    case IS_DST_TEST:
      features = &test_dst_features_p->at(node);
      this_feat_node_type_cnt = &test_dst_feat_node_type_cnt[node * num_node_type];
      break;
    default:
      printf("ERROR\n");
      exit(-1);
  }
  if (choice == 0) {
    // update node weight
    for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
      int n_type = vertex_type[*it];
      int f_cnt = this_feat_node_type_cnt[n_type];
      // assert(f_cnt > 0);  // ensure the f_cnt is non_negative
      real w_nt = weights_node_type[n_type];
      real *vec = &emb_vertex[*it * dim];
      real err = 0;
      for (int k = 0; k < dim; k++) err += pre_vec_err[k] * vec[k] * w_nt / f_cnt;
      weights_node[*it] += 0.01 * lr_supf_emb * (err - reg_supf_emb * weights_node[*it]);  // hazard
    }
  } else if (choice == 1) {
    // update node embedding
    if (ignore_feats) {
      int *f_cnts = new int[num_node_type];
      memset(f_cnts, 0, sizeof(int) * num_node_type);
      int i = -1;
      for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
        i++;
        if (ignore_feats[i]) continue;
        int n_type = vertex_type[*it];
        f_cnts[n_type]++;
      }
      i = -1;
      for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
        i++;
        if (ignore_feats[i]) continue;
        int n_type = vertex_type[*it];
        int f_cnt = f_cnts[n_type];
        real w_nt = weights_node_type[n_type];
        real *vec = &emb_vertex[*it * dim];
        real w_n =  weights_node[*it];
        for (int k = 0; k < dim; k++)
          vec[k] += lr_supf_emb * (pre_vec_err[k] * w_nt * w_n / f_cnt - reg_supf_emb * vec[k]);
      }
      delete [] f_cnts;
    } else {
      for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
        int n_type = vertex_type[*it];
        int f_cnt = this_feat_node_type_cnt[n_type];
        // assert(f_cnt > 0);  // ensure the f_cnt is non_negative
        real w_nt = weights_node_type[n_type];
        real *vec = &emb_vertex[*it * dim];
        real w_n =  weights_node[*it];
        for (int k = 0; k < dim; k++)
          vec[k] += lr_supf_emb * (pre_vec_err[k] * w_nt * w_n / f_cnt - reg_supf_emb * vec[k]);
      }
    }
  } else {
    printf("ERROR\n");
    exit(-1);
  }
}

inline void SupervisedFeatureModel::_get_weighted_node_vector(const int &node,
    const int &is_src_train, real *vec_int, real *vec, bool *ignore_feats) {
  // average feature vectors of the same node type into vec_int
  const vector<int> *features;
  int *this_feat_node_type_cnt;
  bool *feat_has_node_type;
  switch (is_src_train) {
    case IS_SRC_TRAIN:
      features = &train_src_features_p->at(node);
      this_feat_node_type_cnt = &train_src_feat_node_type_cnt[node * num_node_type];
      feat_has_node_type = train_src_feat_has_node_type;
      break;
    case IS_SRC_TEST:
      features = &test_src_features_p->at(node);
      this_feat_node_type_cnt = &test_src_feat_node_type_cnt[node * num_node_type];
      feat_has_node_type = test_src_feat_has_node_type;
      break;
    case IS_DST_TRAIN:
      features = &train_dst_features_p->at(node);
      this_feat_node_type_cnt = &train_dst_feat_node_type_cnt[node * num_node_type];
      feat_has_node_type = train_dst_feat_has_node_type;
      break;
    case IS_DST_TEST:
      features = &test_dst_features_p->at(node);
      this_feat_node_type_cnt = &test_dst_feat_node_type_cnt[node * num_node_type];
      feat_has_node_type = test_dst_feat_has_node_type;
      break;
    default:
      printf("error\n");
      exit(-1);
  }
  memset(vec_int, 0, sizeof(real) * num_node_type * dim);
  if (ignore_feats) {
    int *f_cnts = new int[num_node_type];
    memset(f_cnts, 0, sizeof(int) * num_node_type);
    int i = -1;
    for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
      i++;
      if (ignore_feats[i]) continue;
      int n_type = vertex_type[*it];
      f_cnts[n_type]++;
    }
    i = -1;
    for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
      i++;
      if (ignore_feats[i]) continue;
      int n_type = vertex_type[*it];
      int f_cnt = f_cnts[n_type];
      real w = weights_node[*it];
      real *vec_temp_from = &emb_vertex[*it * dim];
      real *vet_temp_to = &vec_int[n_type * dim];
      for (int k = 0; k < dim; k++) vet_temp_to[k] += vec_temp_from[k] / f_cnt *  w;
    }
    delete [] f_cnts;
  } else {
    for (vector<int>::const_iterator it = features->begin(); it != features->end(); it++) {
      int n_type = vertex_type[*it];
      int f_cnt = this_feat_node_type_cnt[n_type];
      // assert(f_cnt > 0);  // ensure the f_cnt is non_negative
      real w = weights_node[*it];
      real *vec_temp_from = &emb_vertex[*it * dim];
      real *vet_temp_to = &vec_int[n_type * dim];
      for (int k = 0; k < dim; k++) vet_temp_to[k] += vec_temp_from[k] / f_cnt *  w;
    }
  }

  // compute final vec by combining different node type
  memset(vec, 0, sizeof(real) * dim);
  for (int i = 0; i < num_node_type; i++) {
    if (feat_has_node_type[i]) {
      real *vec_temp = &vec_int[i * dim];
      for (int k = 0; k < dim; k++) vec[k] += vec_temp[k] * weights_node_type[i];
    }
  }
}

inline void SupervisedFeatureModel::_get_averaged_node_vector(const vector<int> &features,
    real *vec) {
  int src_fsize = features.size();
  memset(vec, 0, sizeof(real) * dim);
  for (vector<int>::const_iterator it = features.begin(); it != features.end(); ++it) {
    real *vec_temp = &emb_vertex[(*it) * dim];
    for (int k = 0; k < dim; k++) {
      vec[k] += vec_temp[k] / src_fsize;
    }
  }
}

const vector<real> & SupervisedFeatureModel::predict(int choice) {
  int i, src, dst;
  real score;
  real *src_vec = new real[dim], *dst_vec = new real[dim];
  real *src_vec_int = NULL, *dst_vec_int = NULL;
  // uint64 n_seed = static_cast<int64>(time(NULL));

  if (choice == 1) {
    src_vec_int = new real[num_node_type * dim];
    dst_vec_int = new real[num_node_type * dim];
  }

  i = -1;
  for (vector<pair<int, int> >::const_iterator it = test_pairs_p->begin();
      it != test_pairs_p->end(); ++it) {
    i++;
    src = it->first;
    dst = it->second;

    // normally evaluation based on provided test file which includes both pos and neg instances
    // if (test_neg_author_sampler != NULL && (*test_pairs_label_p)[i] == 0) {
    //  dst = test_neg_author_sampler->sample(n_seed);  // random sample negative, debug
    //  test_pairs_dup[i].second = dst;
    // }

    // calc. scr_vec and dst_vec
    if (choice == 0) {
      _get_averaged_node_vector(test_src_features_p->at(src), src_vec);
      _get_averaged_node_vector(test_dst_features_p->at(dst), dst_vec);
    } else {
      _get_weighted_node_vector(src, IS_SRC_TEST, src_vec_int, src_vec);
      _get_weighted_node_vector(dst, IS_DST_TEST, dst_vec_int, dst_vec);
    }

    score = 0;
    for (int k = 0; k < dim; k ++) score += src_vec[k] * dst_vec[k];
    score += bias_node[dst];
    pred[i] = score;
    // pred[i] = rand() / (double) RAND_MAX;  // random prediction
  }

  delete [] src_vec;
  delete [] dst_vec;
  if (src_vec_int != NULL) delete [] src_vec_int;
  if (dst_vec_int != NULL) delete [] dst_vec_int;
  return pred;
}
