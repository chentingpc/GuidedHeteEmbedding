#pragma once
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>


// comparator functions for sorting pairs in descending order
inline static bool CmpFirst(const std::pair<float, unsigned> &a,
                            const std::pair<float, unsigned> &b) {
  return a.first > b.first;
}

inline double eval_apk(std::vector<std::pair<float, unsigned> > &rec, int map_topk) {
  std::sort(rec.begin(), rec.end(), CmpFirst);
  int nhits = 0;
  unsigned k = map_topk >= 0? map_topk: rec.size();
  double sumap = 0.0;
  for (size_t i = 0; i < rec.size(); ++i) {
    if (rec[i].second != 0) {
      nhits += 1;
      if (i < k) {
        sumap += static_cast<double>(nhits) / (i+1);
      }
    }
  }
  if (nhits != 0) {
    sumap /= std::min(nhits, static_cast<int>(k));
    return static_cast<double>(sumap);
  } else {
    return 0.0f;
  }
}

// group: each element should be the size the correpsonding group
// group_start: start count in as evaluation from group_start
// group_end: stop count in as evaluation from group_end
inline double eval_mapk(const std::vector<float> truth, const std::vector<float> pred,
    const std::vector<int> &group, int map_topk, int group_start = -1, int group_end = -1) {
  double mapk = 0, ap_topk;
  int num_group = group.size();
  if (group_start < 0) group_start = 0;
  if (group_end < 0) group_end = num_group;

  int num_group_counted = group_end - group_start;
  int cur = 0, nex;
  for (size_t i = 0; i < group.size(); i ++) {
    nex = cur + group[i];
    if (static_cast<int>(i) < group_start) {
      cur = nex;
      continue;
    }
    if (static_cast<int>(i) >= group_end) break;
    std::vector<std::pair<float, unsigned> > rec;
    rec.reserve(group[i]);
    for (int j = cur; j < nex; j ++)
      rec.push_back(std::make_pair(pred[j], static_cast<int>(truth[j])));
    ap_topk = eval_apk(rec, map_topk);
    mapk += ap_topk / num_group_counted;
    cur = nex;
  }
  return mapk;
}

// some test cases

inline void test_apk() {
  std::vector<std::pair<float, unsigned> > rec(7, std::make_pair(0,0));
  for (size_t i = 0; i < 5; i++)
    rec[i].second = 1;
  rec[0].first = -4;
  rec[1].first = -5;
  rec[2].first = -99;
  rec[3].first = -2;
  rec[4].first = -99;
  rec[5].first = -1;
  rec[6].first = -3;
  assert(eval_apk(rec, 2) == 0.25);
}
