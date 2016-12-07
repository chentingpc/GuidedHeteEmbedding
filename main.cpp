#include "./common.h"
#include "./config.h"
#include "./data_helper.h"
#include "./sampler.h"
#include "./emb_model.h"
#include "./sup_model.h"
#include "./supf_model.h"
#include <time.h>
//#include <ctime>

int main(int argc, char **argv) {
  time_t start, end;
  char time_buffer[80];
  time(&start);
  strftime(time_buffer, 80, "Start timestamp: %Y-%m-%d %I:%M:%S", localtime(&start));
  printf("%s\n", time_buffer);

  Config conf(argc, argv);

#ifdef PA_TASK
  printf("[INFO!] Current run on paper-author mode.\n");
  EmbeddingModel* model;
  DataHelper data_helper = DataHelper(&conf);
  if (conf.train_file.size() * conf.train_feature_file.size() * \
      conf.test_file.size() * conf.test_feature_file.size() == 0) {
    conf.omega = -1;
    printf("[WARNING] No train/test (feature) files are given, proceed without supervised model.\n");
  }

  NodeSampler node_sampler = NodeSampler(data_helper.get_graph(), &conf);
  EdgeSampler edge_sampler = EdgeSampler(data_helper.get_graph());
  if (conf.omega < 0) {
    model = new EmbeddingModel(&data_helper, &node_sampler, &edge_sampler, conf.dim, &conf);
    if (conf.embedding_infile.size() > 0)
      model->load(conf.embedding_infile, conf.is_binary);
  } else {
    data_helper.load_pa_trainortest(conf.train_file, conf.train_feature_file, true);
    data_helper.load_pa_trainortest(conf.test_file, conf.test_feature_file, false);
    data_helper.construct_group(false);
    model = new SupervisedFeatureModel(&data_helper, &node_sampler, &edge_sampler, conf.dim, &conf);
  }
  model->fit();
  model->save(conf.embedding_outfile, conf.is_binary, conf.pred_file);
  delete model;
#else
  // [[legacy]]
  printf("[INFO!] Current run on multi-task mode.\n");
  DataHelper data_helper = DataHelper(&conf);
  data_helper.load_test(conf.test_file);  data_helper.construct_group();
  NodeSampler node_sampler = NodeSampler(data_helper.get_graph(), &conf);
  EdgeSampler edge_sampler = EdgeSampler(data_helper.get_graph());
  SupervisedModel model = SupervisedModel(&data_helper, &node_sampler, &edge_sampler,
                                          conf.dim, &conf);
  if (conf.embedding_infile.size() > 0)
    model.load(conf.embedding_infile, conf.is_binary);
  model.fit();
  model.save(conf.embedding_outfile, conf.is_binary, conf.pred_file);
#endif

  time(&end);
  strftime(time_buffer, 80,"End timestamp: %Y-%m-%d %I:%M:%S", localtime(&end));
  printf("%s\n", time_buffer);
  printf("The program finishes in %ld seconds.\n", (end - start));
  return 0;
}
