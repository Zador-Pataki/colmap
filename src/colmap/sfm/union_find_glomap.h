#pragma once
#include <cstdint>
#include <unordered_map>

// TODO(dedup-glomap-vs-colmap4): the entire body of this file duplicates
// colmap::UnionFind<T> at colmap/math/union_find.h:40 (native is a strict
// superset — adds Reserve/FindIfExists/Compress/Parents). Drop this header
// and route TrackEngine to the native type. See
// .claude/notes/glomap_audit/audit_glomap_files_vs_colmap4.md.

namespace colmap {
namespace glomap_ra {

// UnionFind class to maintain disjoint sets for creating tracks
template <typename DataType>
class UnionFind {
 public:
  // Find the root of the element x
  DataType Find(DataType x) {
    // If x is not in parent map, initialize it with x as its parent
    auto parentIt = parent_.find(x);
    if (parentIt == parent_.end()) {
      parent_.emplace_hint(parentIt, x, x);
      return x;
    }
    // Path compression: set the parent of x to the root of the set containing x
    if (parentIt->second != x) {
      parentIt->second = Find(parentIt->second);
    }
    return parentIt->second;
  }

  // Unite the sets containing x and y
  void Union(DataType x, DataType y) {
    DataType root_x = Find(x);
    DataType root_y = Find(y);
    if (root_x != root_y) parent_[root_x] = root_y;
  }

  void Clear() { parent_.clear(); }

 private:
  // Map to store the parent of each element
  std::unordered_map<DataType, DataType> parent_;
};

}  // namespace glomap_ra
}  // namespace colmap
