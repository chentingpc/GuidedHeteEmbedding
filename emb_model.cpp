#include "emb_model.h"

/**********************************************
 * Work scheduler and helpers
 *********************************************/

void EmbeddingModel::fit() {
  time_t start, end;
  time(&start);

  if (num_threads == 0) printf("[WARNING!] num_threads is set to zero!!!!!!!!!!!!!!!!!!!!!!!!\n");
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("--------------------------------\n");
  Context *context[num_threads];
  for (int a = 0; a < num_threads; a++) {
    context[a] = new Context;
    context[a]->model_ptr = this;
    context[a]->id = a;
    pthread_create(&pt[a], NULL, fit_thread_helper, (void *)(context[a]));
  }
  for (int a = 0; a < num_threads; a++) {
    pthread_join(pt[a], NULL);
    free(context[a]);
  }
  printf("\n");

  time(&end);
  printf("Embedding training finished in %ld seconds.\n", (end - start));
}

void EmbeddingModel::save(string embedding_file, bool is_binary) {
  if (embedding_file.size() == 0) {
    printf("[WARNING] embedding_file not saved due to no path given.\n");
    return;
  }
  vector<int> desired_node_types;
  printf("[INFO] saving embedding to file..\n");
  FILE *fo = fopen(embedding_file.c_str(), "wb");
  assert(fo != NULL);
  fprintf(fo, "%d %d\n", num_vertices, dim);
  for (int a = 0; a < num_vertices; a++) {
    fprintf(fo, "%s ", vertex[a].name);
    if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
    else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

void EmbeddingModel::load(string embedding_file, bool is_binary) {
  printf("[INFO] loading embedding from file..\n");

  char _name[MAX_STRING];
  int _num_vertices, _dim;
  map<string, int> name2vid;
  for (int a = 0; a < num_vertices; a++) {
    string name(vertex[a].name);
    name2vid[name] = a;
  }

  FILE *fi = fopen(embedding_file.c_str(), "rb");
  assert(fi != NULL);
  fscanf(fi, "%d %d\n", &_num_vertices, &_dim);
  assert(_num_vertices == num_vertices);
  assert(_dim == dim);
  for (int a = 0; a < num_vertices; a++) {
    fscanf(fi, "%s", _name);
    int v = name2vid[_name];
    assert(strcmp(vertex[v].name, _name) == 0);
    _name[0] = fgetc(fi);
    if (is_binary) {
      for (int b = 0; b < dim; b++)
        fread(&emb_vertex[v * dim + b], sizeof(real), 1, fi);
    } else {
      for (int b = 0; b < dim; b++)
        fscanf(fi, "%f", &emb_vertex[v * dim + b]);
    }
  }
  fclose(fi);
}

/**********************************************
 * Initializations
 *********************************************/

void EmbeddingModel::init_variables () {
  sigmoid = new Sigmoid();

  graph = data_helper->get_graph();
  node_type_to_edge_type = graph->node_type_to_edge_type;
  vertex = graph->vertex;
  vertex_type = graph->vertex_type;
  vertex_degree_of_etype = graph->vertex_degree_of_etype;
  node_name2type = data_helper->get_node_name2type();
  node_type2name = data_helper->get_node_type2name();
  edge_name2type = data_helper->get_edge_name2type();
  edge_type2name = data_helper->get_edge_type2name();
  non_vertex_id2name = data_helper->get_non_vertex_id2name();
  num_node_type = *graph->num_node_type_p;
  edge_type = graph->edge_type;
  edge_source_id = graph->edge_source_id;
  edge_target_id = graph->edge_target_id;
  num_vertices = *graph->num_vertices_p;
  num_edges = *graph->num_edges_p;
  num_edge_type = *graph->num_edge_type_p;
  use_path_conf = conf_p->use_path_conf;
  if (use_path_conf) {
    path_direction = conf_p->path_direction;
    path_order = conf_p->path_order;
  }

  init_lr_net_emb = lr_net_emb = conf_p->lr_net_emb;
  init_lr_net_w = lr_net_w = conf_p->lr_net_w;
  init_lr_net_etype_bias = lr_net_etype_bias = conf_p->lr_net_etype_bias;

  this->num_threads = conf_p->num_threads;
  this->num_negative = conf_p->num_negative;
  this->total_samples = conf_p->total_samples;

  edge_type_using_context = new bool[num_edge_type];
  if (use_path_conf)
    for (int m = 0; m < num_edge_type; m++) edge_type_using_context[m] = conf_p->path_order[m];
  else
    for (int m = 0; m < num_edge_type; m++) edge_type_using_context[m] = conf_p->path_order_default;
  printf("[using context] edge_type whether-use-context\n");
  for (int m = 0; m < num_edge_type; m++)
    printf("\t%s, %d\n", edge_type2name[m].c_str(), edge_type_using_context[m]);
}

void EmbeddingModel::init_task_schduler() {
  // current task switcher is mini-batch based
  // network embedding and supervised task has same number of threads
  // yield when the designated mini-batch finished
  current_sample_count = 0;
  current_sample_count_emb = 0;
  fit_not_finished = true;

  real emb_task_sampling_rate = conf_p->omega;
  if (emb_task_sampling_rate >= 0) {
    const int rounds_task_switch = 100;  // dark parameter
    int samples_per_round_thread = total_samples / rounds_task_switch / num_threads;
    samples_before_switch_emb = samples_per_round_thread * emb_task_sampling_rate;
    samples_before_switch_other = samples_per_round_thread * (1 - emb_task_sampling_rate);
  } else {
    samples_before_switch_emb = -1;
    samples_before_switch_other = -1;
  }

  printf("[INFO!] embedding task sampling rate %f, samples_before_switch_emb %lld, "
    "samples_before_switch_other %lld.\n", emb_task_sampling_rate, samples_before_switch_emb,
    samples_before_switch_other);

  task_switchs_for_embedding = new bool[num_threads];
  for (int i = 0; i < num_threads; i++) task_switchs_for_embedding[i] = true;
}

void EmbeddingModel::init_vector() {
  printf("Initialize embedding vectors... \r");
  fflush(stdout);

  clock_t start, end;
  start = clock();
  int64 a, b;
  srand(time(NULL));

  // speed tips:
  // 1. posix_memalign can help a little bit, less than 5% probably.
  // 2. 1d indexing compared to 2d indexing help a little bit, less than 5% probably.
  // 3. memory continuity helps more, more than 15% probably.
  a = posix_memalign((void **)&emb_vertex, 128, (int64)num_vertices * dim * sizeof(real));
  if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
  for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
    emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

  bool filter_edge_type_using_context = false;
  for (int m = 0; m < num_edge_type; m++)
    if (edge_type_using_context[m]) { filter_edge_type_using_context = true; break; }
  if (filter_edge_type_using_context) {
    a = posix_memalign((void **)&emb_context, 128, (int64)num_vertices * dim * sizeof(real));
    if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
      emb_context[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
  }

  // todo - following can be optimized by using memory chunk trick
  // initilize w vector for edge type
  w_mn = new real**[num_edge_type];
  for (int m = 0; m < num_edge_type; m++) {
    w_mn[m] = new real*[num_node_type];
    for (int n = 0; n < num_node_type; n++) {
      if (node_type_to_edge_type[n * num_edge_type + m]) {
        w_mn[m][n] = new real[dim];
        // debug
        for (int c = 0; c < dim; c++) {
          w_mn[m][n][c] = 1.;
          // w_mn[m][n][c] = rand() / (real)RAND_MAX + 0.5;
        }
      } else {
        w_mn[m][n] = NULL;
      }
    }
  }

  // initialize bias for edge type
  bias_edge_type = new real[num_edge_type];
  for (int i = 0; i < num_edge_type; i++) bias_edge_type[i] = 0.;

  ll_edge_type = new double[num_edge_type];
  memset(ll_edge_type, 0, sizeof(double) * num_edge_type);

  ll_edge_type_cnt = new int64[num_edge_type];
  memset(ll_edge_type_cnt, 0, sizeof(int64) * num_edge_type);

  end = clock();
  printf("Embedding vectors initialized in %.2f (s).\n", (double)(end-start) / CLOCKS_PER_SEC);
}

/**********************************************
 * Training functions
 *********************************************/

void EmbeddingModel::fit_thread(int id) {
  if (samples_before_switch_emb == 0) {
    printf("[WARNING!] turn down emb_model training..\n");
    if (id == 0) fit_not_finished = false;
    return;
  }
  if (init_lr_net_w > 0) using_transformation_vector = true;
  if (init_lr_net_etype_bias > 0) using_edge_type_bias = true;

  int64 u, v;
  int64 count = 0, last_count = 0, ll_count = 0, curedge;
  int64 samples_task_round = 0;
  int direction = conf_p->path_direction_default;
  double prog = 0., ll = 0.;
  uint64 seed = static_cast<int64>(id);
  real lr_w = init_lr_net_w;
  real lr_bias = init_lr_net_etype_bias;
  real *vec_error = new real[dim];
  real *e_type_bias_err_vec = NULL, ***w_mn_err = NULL;

  // init error holder for relational weights & bias, if not done, will follow homo-net embedding
  if (using_transformation_vector) {
    w_mn_err = new real**[num_edge_type];
    for (int m = 0; m < num_edge_type; m++) {
      w_mn_err[m] = new real*[num_node_type];
      for (int n = 0; n < num_node_type; n++) {
        if (node_type_to_edge_type[n * num_edge_type + m]) {
          w_mn_err[m][n] = new real[dim];
          memset(w_mn_err[m][n], 0, sizeof(real) * dim);
        } else { w_mn_err[m][n] = NULL; }
      }
    }
  }
  if (using_edge_type_bias) {
    e_type_bias_err_vec = new real[num_edge_type];
    memset(e_type_bias_err_vec, 0, sizeof(real) * num_edge_type);
  }

  while (current_sample_count < total_samples) {
    static const int count_interval = 100000;
    if (count - last_count > count_interval) {
      // reset and (slow) updates for weights and biases
      int64 incremental = count - last_count;
      current_sample_count += incremental;
      current_sample_count_emb += incremental;
      last_count = count;

      if (id == 0) {
        // reset logistics, update learning rates
        real sample_ratio = current_sample_count_emb / (double)(current_sample_count + 1);
        prog = (real)current_sample_count / (real)(total_samples + 1);
        printf("Prog: %.2lf%%, emb_sample_ratio: %f, LogL: %.4lf\n", prog * 100, sample_ratio,
          ll / ll_count);
        fflush(stdout);
        lr_net_emb = init_lr_net_emb * (1. - prog);
        lr_net_w = init_lr_net_w * (1. - prog);
        lr_net_etype_bias = init_lr_net_etype_bias * (1. - prog);
        if (lr_net_emb < init_lr_net_emb * 0.001) lr_net_emb = init_lr_net_emb * 0.001;
        if (lr_net_w < init_lr_net_w * 0.001) lr_net_w = init_lr_net_w * 0.001;
        if (lr_net_etype_bias < init_lr_net_etype_bias * 0.001) lr_net_etype_bias = init_lr_net_etype_bias * 0.001;
        lr_w = lr_net_w;
        lr_bias = lr_net_etype_bias;
        ll = ll_count = 0;
        static const int compress_to_cnt = count_interval / 5;  // downweight previous for smoothing
        for (int m = 0; m < num_edge_type; m++) {
          ll_edge_type[m] *= static_cast<double>(compress_to_cnt) / ll_edge_type_cnt[m];
          ll_edge_type_cnt[m] = compress_to_cnt;
        }

        // update weights for transforming embedding
        if (using_transformation_vector) {
          real _w_mn_learn_rate = 1. / count_interval * lr_w;
          for (int m = 0; m < num_edge_type; m++) {
            for (int n = 0; n < num_node_type; n++) {
              if (w_mn_err[m][n] != NULL) {
                real *w_mn_mn = w_mn[m][n];
                real *w_mn_err_mn = w_mn_err[m][n];
                for (int c = 0; c < dim; c++) w_mn_mn[c] += _w_mn_learn_rate * w_mn_err_mn[c];
                memset(w_mn_err_mn, 0, sizeof(real) * dim);
              }
            }
          }
        }

        // update global bias
        if (using_edge_type_bias) {
          real _b_learn_rate = 1. / count_interval * lr_bias;
          for (int m = 0; m < num_edge_type; m++) {
            bias_edge_type[m] += e_type_bias_err_vec[m] * _b_learn_rate;
          }
          memset(e_type_bias_err_vec, 0, sizeof(real) * num_edge_type);
        }
      }
    }

    // task schedule helper
    if (samples_task_round == samples_before_switch_emb) {
      samples_task_round = 0;
      task_switchs_for_embedding[id] = false;
      while (!task_switchs_for_embedding[id] && fit_not_finished) {
        usleep(100);
      }
    }

    // sample an edge for training
    curedge = edge_sampler->sample();
    u = edge_source_id[curedge];
    v = edge_target_id[curedge];

    if (use_path_conf)
      direction = path_direction[edge_type[curedge]];

    if (direction == PATH_DIRECTION_BIDIRECTION) {
      if (gsl_rand() < 0.5)
        train_on_sample(id, u, v, curedge, ll, seed, vec_error,
                        e_type_bias_err_vec, w_mn_err);
      else
        train_on_sample(id, v, u, curedge, ll, seed, vec_error,
                        e_type_bias_err_vec, w_mn_err);
    } else if (direction == PATH_DIRECTION_NORMAL) {
      train_on_sample(id, u, v, curedge, ll, seed, vec_error,
                      e_type_bias_err_vec, w_mn_err);
    } else if (direction == PATH_DIRECTION_REVERSE) {
      train_on_sample(id, v, u, curedge, ll, seed, vec_error,
                      e_type_bias_err_vec, w_mn_err);
    } else {
      printf("[ERROR!] direction %d not recognized\n", direction);
      exit(-1);
    }
    count++;
    samples_task_round++;
    ll_count++;
  }

  if (id == 0) {
    if (using_transformation_vector) {
      // print out the transformation vector
      sleep(2);
      printf("------------------------------- w_mn -------------------------------\n");
      for (int n = 0; n < num_node_type; n++) {
        printf("\n\n[node_type %s]\n\n", node_type2name[n].c_str());
        for (int m = 0; m < num_edge_type; m++) {
          if (w_mn[m][n] != NULL) {
            printf("\t[edge_type %s]", edge_type2name[m].c_str());
            real *w_mn_mn = w_mn[m][n];
            for (int c = 0; c < dim; c++) printf(", %f", w_mn_mn[c]);
            printf("\n");
          }
        }
      }
    }
    if (using_edge_type_bias) {
      sleep(2);
      printf("------------------------------- bias_edge_type -------------------------------\n");
      for (int m = 0; m < num_edge_type; m++) {
        printf("\n[bias_edge_type %s] %f\n", edge_type2name[m].c_str(), bias_edge_type[m]);
      }
    }
    fflush(stdout);
  }

  fit_not_finished = false;
  pthread_exit(NULL);
}

inline void EmbeddingModel::train_on_sample(const int &id, int64 &u, int64 &v,
    const int64 &curedge, double &ll, uint64 &seed, real *vec_error,
    real *e_type_bias_err_vec, real ***w_mn_err) {
  int64 lu, lv, target;
  real *src_vec, *dst_vec, *dst_vec_pos = NULL, *dst_vec_neg = NULL;
  int label, src_type, dst_type, e_type;
  real e_type_bias = 0;
  real *e_type_bias_err = NULL;
  real *w_mn_u = NULL, *w_mn_v = NULL, *w_mn_err_u = NULL, *w_mn_err_v = NULL;

  src_type = vertex_type[u];
  dst_type = vertex_type[v];
  e_type = edge_type[curedge];
  if (e_type_bias_err_vec != NULL) {
    e_type_bias = bias_edge_type[e_type];
    e_type_bias_err = &e_type_bias_err_vec[e_type];
  }
  if (w_mn_err != NULL) {
    w_mn_u = w_mn[e_type][src_type];
    w_mn_v = w_mn[e_type][dst_type];
    w_mn_err_u = w_mn_err[e_type][src_type];
    w_mn_err_v = w_mn_err[e_type][dst_type];
  }

  lu = u * dim;
  src_vec = &emb_vertex[lu];
  memset(vec_error, 0, sizeof(real) * dim);

  // NEGATIVE SAMPLING
  real ll_acc = 0;
  for (int d = 0; d != num_negative + 1; d++) {
    if (d == 0) {
      target = v;
      label = 1;
    } else {
      target = node_sampler->sample(seed, e_type, dst_type);
      label = 0;
      // random flipping sign, might be slightly helpful
      // if (uniform() < 0.01)
      //  label = 1;
    }
    lv = target * dim;
    if (edge_type_using_context[e_type])
      dst_vec = &emb_context[lv];
    else
      dst_vec = &emb_vertex[lv];
    real ll_local;
    static int objective = conf_p->net_loss;
    if (w_mn_err != NULL) {
      ll_local = update_with_weight(src_vec, dst_vec, vec_error, label,
                                    e_type_bias, e_type_bias_err,
                                    w_mn_u, w_mn_v, w_mn_err_u, w_mn_err_v);
    } else if (objective == 0) {
      ll_local = update_skipgram(src_vec, dst_vec, vec_error, label, e_type_bias, e_type_bias_err);
    } else {
      if (d == 0) {
        if (edge_type_using_context[e_type])
          dst_vec_pos = &emb_context[v * dim];
       else
          dst_vec_pos = &emb_vertex[v * dim];
        continue;
      }
      dst_vec_neg = dst_vec;
      ll_local = update_maxmargin(src_vec, dst_vec_pos, dst_vec_neg, vec_error);
    }
    ll += ll_local;
    ll_acc += ll_local;
  }
  for (int c = 0; c < dim; c++) src_vec[c] += vec_error[c];

  if (id == 0) {
    ll_edge_type[e_type] += ll_acc;
    ll_edge_type_cnt[e_type]++;
  }
}

inline real EmbeddingModel::update_skipgram(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err) {
  real x = 0, f, g, g_w_lr;
  real reg_lr = reg_net_emb * lr_net_emb;
  for (int c = 0; c < dim; c++) x += vec_u[c] * vec_v[c];
  f = (*sigmoid)(x + e_type_bias);
  g = (label - f);
  g_w_lr = g * lr_net_emb;
  for (int c = 0; c < dim; c++) vec_error[c] += g_w_lr * vec_v[c] - reg_lr * vec_u[c];
  for (int c = 0; c < dim; c++) vec_v[c] += g_w_lr * vec_u[c] - reg_lr * vec_v[c];
  if (e_type_bias_err != NULL)
    *e_type_bias_err += g;
  return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
}

inline real EmbeddingModel::update_maxmargin(real *src_vec, real *dst_vec_pos,
    real *dst_vec_neg, real *vec_error) {
  real ll_local = 0;
  real score_pos = 0, score_neg = 0;
  real *src_err = vec_error;

  for (int c = 0; c < dim; c++) score_pos += src_vec[c] * dst_vec_pos[c];
  for (int c = 0; c < dim; c++) score_neg += src_vec[c] * dst_vec_neg[c];
  const real margin = -1;
  real margin_temp = score_pos - score_neg + margin;
  ll_local = margin_temp > 0? 0: margin_temp;

  real score_pos_err = margin_temp >= 0? 0: 1;
  real score_neg_err = -score_pos_err;
  for (int k = 0; k < dim; k++) {
    src_err[k] = lr_net_emb * (score_pos_err * (dst_vec_pos[k] - dst_vec_neg[k]) -
                        reg_net_emb * src_err[k]);
    dst_vec_pos[k] += lr_net_emb * (score_pos_err * src_vec[k] - reg_net_emb * dst_vec_pos[k]);
    dst_vec_neg[k] += lr_net_emb * (score_neg_err * src_vec[k] - reg_net_emb * dst_vec_neg[k]);
  }
  return ll_local;
}

// dark parameter, debug
#define REG_ON_WU

inline real EmbeddingModel::update_with_weight(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err,
    real *w_mn_u, real *w_mn_v, real *w_mn_err_u, real *w_mn_err_v) {
  real x = 0, f, g;
  for (int c = 0; c < dim; c++)
    x += (vec_u[c] * w_mn_u[c]) * (vec_v[c] * w_mn_v[c]);
  f = (*sigmoid)(x + e_type_bias);
  g = (label - f);
  for (int c = 0; c < dim; c++) {
    real temp = g * (vec_v[c] * w_mn_v[c]);
#ifdef REG_ON_WU
    // regularization on WU
    real wu = w_mn_u[c] * vec_u[c];
    real reg_w_mn = reg_net_emb * wu * vec_u[c];
    real reg_vec = reg_net_emb * wu * w_mn_u[c];
#else
    // regularization on W and U
    real reg_w_mn = gamma * w_mn_u[c];  // debug, gamma
    real reg_vec = reg_net_emb * vec_u[c];
#endif
    w_mn_err_u[c] += temp * vec_u[c] - reg_w_mn;
    vec_error[c] += lr_net_emb * (temp * w_mn_u[c] - reg_vec);
  }
  for (int c = 0; c < dim; c++) {
    real temp = g * (w_mn_u[c] * vec_u[c]);
    // regularization on WV
#ifdef REG_ON_WU
    real wv = w_mn_v[c] * vec_v[c];
    real reg_w_mn = reg_net_emb * wv * vec_v[c];
    real reg_vec = reg_net_emb * wv * w_mn_v[c];
#else
    // regularization on W and V, debug
    real reg_w_mn = gamma * w_mn_v[c];  // debug, gamma
    real reg_vec = reg_net_emb * vec_v[c];
#endif
    w_mn_err_v[c] += temp * vec_v[c] - reg_w_mn;
    vec_v[c] += lr_net_emb * (temp * w_mn_v[c] - reg_vec);
  }
  if (e_type_bias_err != NULL)
    *e_type_bias_err += g;

  return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
}


