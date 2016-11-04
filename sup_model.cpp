#include "sup_model.h"

// no much to train, adjust path task weight if needed
void SupervisedModel::train_thread(int id) {
  int count = 0, last_count = 0;

  while (current_sample_count < total_samples) {
    static const int count_interval = 10000;
    if (count - last_count > count_interval) {
      sleep(3);
      if (id == 0) {
        bool printing = true;
        if (printing) {
          printf("                                                                           $ll:");
          for (int i = 0; i < num_edge_type; i++) {
            printf(" %.3f", ll_edge_type[i] / ll_edge_type_cnt[i]);  // print path likelihood
          }
          printf(" $b:");
          for (int i = 0; i < num_edge_type; i++)
            printf(" %.3f", bias_edge_type[i]);
        }
      }
      last_count = count;
    }

    count++;
  }
}

const vector<real> & SupervisedModel::predict() {
  int i = -1;

  for (vector<pair<int, int> >::const_iterator it = test_pairs_p->begin();
      it != test_pairs_p->end(); ++it) {
    i++;
    real score = 0;
    int src = it->first;
    int dst = it->second;
    int e_type = (*test_pairs_etype_p)[i];
    real *src_vec = &emb_vertex[src * dim];
    real *dst_vec = &emb_vertex[dst * dim];

    if (e_type != -1 && false) {
      // predict with transformed embedding, debug
      int src_type = vertex_type[src];
      int dst_type = vertex_type[dst];
      if (using_transformation_vector) {
        real *w_mn_u = w_mn[e_type][src_type];
        real *w_mn_v = w_mn[e_type][dst_type];
        for (int c = 0; c != dim; c++)
          score += (src_vec[c] * w_mn_u[c]) * (dst_vec[c] * w_mn_v[c]);
      } else if (using_transformation_matrix) {
        real *W_m_uv = W_m_band[e_type];
        for (int c = 0; c < dim; c++)
          for (int l = 0, k = c + (l - ls); l < band_width; l++, k++) {
            if (k < 0 || k >= dim) continue;
            score += src_vec[c] * W_m_uv[c * band_width + l] * dst_vec[k];
          }
      } else {
        for (int k = 0; k < dim; k++) score += src_vec[k] * dst_vec[k];
      }
    } else {
      // predict with raw embedding
      for (int k = 0; k < dim; k++) score += src_vec[k] * dst_vec[k];  // norm;
    }
    // pred[i] = (*sigmoid)(score * e_weight);
    pred[i] = score;
  }

  return pred;
}
