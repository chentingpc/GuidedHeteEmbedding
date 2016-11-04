#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <utility>
#include <cassert>
#include <unistd.h>
#include "./common.h"
#include "./utility.h"
#include "./sampler.h"
#include "./data_helper.h"
#include "./emb_model.h"
#include "./metrics.h"
using namespace std;

class SupervisedModel: public EmbeddingModel {
  const vector<int>             *train_group_p;
  const vector<pair<int, int> > *train_pairs_p;
  const vector<real>            *train_pairs_label_p;

  const vector<int>             *test_group_p;
  const vector<pair<int, int> > *test_pairs_p;
  const vector<real>            *test_pairs_label_p;
  const vector<int>             *test_pairs_etype_p;
  const vector<string>          *test_pairs_type_p;
  const map<string, pair<int, int> >
                                *test_task_group_start_end_p;
  vector<string>                test_task_name_of_group;

  vector<real>                  pred;

  static void *train_helper(void* context) {
      Context *c = (Context *)context;
      SupervisedModel* p = static_cast<SupervisedModel*>(c->model_ptr);
      p->train_thread(c->id);
      return NULL;
  }

  static void *eval_helper(void* context) {
      Context *c = (Context *)context;
      SupervisedModel* p = static_cast<SupervisedModel*>(c->model_ptr);
      p->eval_thread(c->id);
      return NULL;
  }

  void eval_thread(int map_topk) {
    int eval_cnt = 100;  // total number to evaluate during the training
    int64 eval_per_sample = total_samples / eval_cnt;
    int64 eval_cur_sample = 0;
    while (true) {
      if (current_sample_count >= eval_cur_sample) {
        eval(map_topk);
        while (current_sample_count >= eval_cur_sample)
          eval_cur_sample += eval_per_sample;
      } else if (eval_cur_sample >= total_samples) {
        break;
      } else {
        sleep(1);  // sleep for seconds before checking to eval again
      }
    }
  }

  void train_thread(int id);

 public:
  SupervisedModel(DataHelper *data_helper,
                  NodeSampler *node_sampler,
                  EdgeSampler *edge_sampler,
                  int dim, const Config *conf_p) :
                  EmbeddingModel(data_helper, node_sampler, edge_sampler, dim, conf_p) {
    train_group_p = data_helper->get_train_group();
    train_pairs_p = data_helper->get_train_pairs();
    train_pairs_label_p = data_helper->get_train_pairs_label();
    test_group_p = data_helper->get_test_group();
    test_pairs_p = data_helper->get_test_pairs();
    test_pairs_label_p = data_helper->get_test_pairs_label();
    test_pairs_etype_p = data_helper->get_test_pairs_etype();
    test_pairs_type_p = data_helper->get_test_pairs_type();
    test_task_group_start_end_p = data_helper->get_test_task_group_start_end();
    int i = -1;
    for (map<string, pair<int, int> >::const_iterator it = test_task_group_start_end_p->begin();
        it != test_task_group_start_end_p->end(); it ++) {
      i++;
      test_task_name_of_group.push_back(it->first);
    }
    pred.resize(test_pairs_p->size());
  }

  void fit() {
    int num_train_thread = 1, num_eval_thread = 1, a;
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

    EmbeddingModel::fit();

    for (a = 0; a < num_train_thread + num_eval_thread; a++) {
      pthread_join(pt[a], NULL);
      free(context[a]);
    }
  }

  void eval(int map_topk = -1) {
    const vector<real> &test_pairs_label = *test_pairs_label_p;
    predict();
    const map<string, pair<int, int> > &test_task_group_start_end = *test_task_group_start_end_p;
    int num_task = test_task_group_start_end.size();
    if (num_task == 0) {
      real mapk = eval_mapk(test_pairs_label, pred, *test_group_p, map_topk);
      printf(" Test map@%d: %f\n", map_topk, mapk);
    } else {
      real *mapk_container = new real[num_task];
      int i = -1;
      for (map<string, pair<int, int> >::const_iterator it = test_task_group_start_end.begin();
          it != test_task_group_start_end.end(); it ++) {
        i++;
        mapk_container[i] = eval_mapk(test_pairs_label, pred, *test_group_p, map_topk,
          it->second.first, it->second.second);
      }
      printf(" Test map@%d", map_topk);
      for (int i = 0; i < num_task; i++)
        printf(", %s:%.4f", test_task_name_of_group[i].c_str(), mapk_container[i]);
      printf("\n");
      delete [] mapk_container;
    }
  }

  const vector<real> & predict();

  void save(string embedding_file, bool is_binary, string pred_file) {
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
      const pair<int, int> &the_test_pair = (*test_pairs_p)[i];
      fprintf(fo, "%s\t%s\t%f\t%f\t%s\n", vertex[the_test_pair.first].name,
        vertex[the_test_pair.second].name, (*test_pairs_label_p)[i], pred[i],
        (*test_pairs_type_p)[i].c_str());
    }
    fclose(fo);
  }
};
