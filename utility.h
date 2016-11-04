#pragma once
#include <math.h>
#include <vector>
#include <string>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <cassert>
#include "./common.h"

using namespace std;

template<typename T>
inline T& max(T &x, T &y) {
  return x > y? x : y;
}

template<typename T>
inline T& min(T &x, T &y) {
  return x > y? y : x;
}

template<typename T>
inline T Sqr(T x) {
  return x*x;
}

/* Build a hash table, mapping each vertex name to a unique vertex id */
class VertexHashTable {
 protected:
  int                     *vertex_hash_table;

  inline void init_hash_table() {
    vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
    for (int k = 0; k != hash_table_size; k++)
      vertex_hash_table[k] = -1;
  }

  inline uint hash(char *key) {
    uint seed = 131;
    uint hash = 0;
    while (*key) {
      hash = hash * seed + (*key++);
    }
    return hash % hash_table_size;
  }

 public:

  VertexHashTable() {
    init_hash_table();
  }

  inline int add_vertex(char *name, Vertex *&vertex,
      int & num_vertices, int & max_num_vertices) {
    int length = strlen(name) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
    strcpy(vertex[num_vertices].name, name);
    vertex[num_vertices].degree = 0;
    num_vertices++;
    if (num_vertices + 2 >= max_num_vertices) {
      max_num_vertices += 10000;
      vertex = (struct Vertex *)realloc(vertex, max_num_vertices * sizeof(struct Vertex));
    }
    insert_hash_table(name, num_vertices - 1);
    return num_vertices - 1;
  }

  inline void insert_hash_table(char *key, int value) {
    int addr = hash(key);
    while (vertex_hash_table[addr] != -1)
      addr = (addr + 1) % hash_table_size;
    vertex_hash_table[addr] = value;
  }

  inline int search_hash_table(char *key, Vertex *vertex) {
    int addr = hash(key);
    while (1) {
      if (vertex_hash_table[addr] == -1)
        return -1;
      if (!strcmp(key, vertex[vertex_hash_table[addr]].name))
        return vertex_hash_table[addr];
      addr = (addr + 1) % hash_table_size;
    }
    return -1;
  }
};


/* Fastly compute sigmoid function */
class Sigmoid {
  real *sigmoid_table;

  void init_sigmoid_table() {
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k < sigmoid_table_size + 1; k++) {
      x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
      real val = 1 / (1 + exp(-x));
      val = val >= 1.? 1.: val;
      val = val <= 0.? 0.: val;
      sigmoid_table[k] = val;
    }
  }

 public:
  Sigmoid() {
    init_sigmoid_table();
  }

  real operator()(real x) {
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2.0;
    return sigmoid_table[k];
  }
};

inline float fast_log2 (float val) {
  // assert(val >= 0.);
  int * const    exp_ptr = reinterpret_cast <int *> (&val);
  int            x = *exp_ptr;
  const int      log_2 = ((x >> 23) & 255) - 128;
  x &= ~(255 << 23);
  x += 127 << 23;
  *exp_ptr = x;

  val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

  return (val + log_2);
}

inline float fast_log (const float &val) {
  return log(val);  // correctness is more important for now
  // return (fast_log2 (val) * 0.69314718f);  // can save like 10s+ in 170s.
}

inline vector<string> split(const string &text, char sep) {
  vector<string> tokens;
  size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
}
