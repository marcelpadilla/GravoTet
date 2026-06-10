#ifdef _MSC_VER
#define _USE_MATH_DEFINES  // expose M_PI from <cmath> on MSVC
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Check if parallel execution policies are available
// Apple Clang and older compilers don't support std::execution::par
// Check if parallel execution policies are available
// Apple Clang and older compilers don't support std::execution::par
#if defined(__cpp_lib_parallel_algorithm) &&                                   \
    __cpp_lib_parallel_algorithm >= 201603L
#include <execution>
#define HAS_PARALLEL_EXECUTION 1
#define PARALLEL_SORT(begin, end) std::sort(std::execution::par, begin, end)
#define PARALLEL_SORT_CMP(begin, end, cmp)                                     \
  std::sort(std::execution::par, begin, end, cmp)
#else
#define HAS_PARALLEL_EXECUTION 0
#define PARALLEL_SORT(begin, end) std::sort(begin, end)
#define PARALLEL_SORT_CMP(begin, end, cmp) std::sort(begin, end, cmp)
#endif

// Hash specialization for std::array<int, 3> and std::pair<int, int>
namespace std {
template <> struct hash<std::array<int, 3>> {
  std::size_t operator()(const std::array<int, 3> &arr) const noexcept {
    std::size_t h1 = std::hash<int>{}(arr[0]);
    std::size_t h2 = std::hash<int>{}(arr[1]);
    std::size_t h3 = std::hash<int>{}(arr[2]);
    // Combine hashes
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};
template <> struct hash<std::pair<int, int>> {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    std::size_t h1 = std::hash<int>{}(p.first);
    std::size_t h2 = std::hash<int>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};
} // namespace std
#include "multigrid_solver.h"

namespace GravoMG {

// Constructor
TetMultigridSolver::TetMultigridSolver() {
  // OpenMP uses all available threads by default (OMP_NUM_THREADS)
}

// Destructor - clean up allocated memory
TetMultigridSolver::~TetMultigridSolver() {
  clearHierarchy();
}

// Clear all hierarchy data for reuse
void TetMultigridSolver::clearHierarchy() {
  // Input data
  inputMesh = TetrahedralMesh();

  numSurfaceVerticesLevel0 = 0;

  // Per-level data
  all_vertices.clear();
  all_vertices.shrink_to_fit();
  all_tetrahedra.clear();
  all_tetrahedra.shrink_to_fit();
  boundary_indices.clear();
  boundary_indices.shrink_to_fit();
  nr_points.clear();
  nr_points.shrink_to_fit();
  allP.clear();
  allP.shrink_to_fit();
  allNeigh.clear();
  allNeigh.shrink_to_fit();

  // Clustering data
  allNearestSource.clear();
  allNearestSource.shrink_to_fit();
  allSampleIndices.clear();
  allSampleIndices.shrink_to_fit();

  // Simplicial complex data
  allTris.clear();
  allTris.shrink_to_fit();
  allTets.clear();
  allTets.shrink_to_fit();
  allNeighSets.clear();
  allNeighSets.shrink_to_fit();

  // Surface data
  allExteriorVertices.clear();
  allExteriorVertices.shrink_to_fit();
  allExteriorNeigh.clear();
  allExteriorNeigh.shrink_to_fit();
  allExteriorFeature.clear();
  allExteriorFeature.shrink_to_fit();

  // Reset benchmark
  benchmark = BenchmarkData();

  // V-Cycle data
  vcycle_operators_.clear();
  vcycle_operators_.shrink_to_fit();
  vcycle_prolongations_.clear();
  vcycle_prolongations_.shrink_to_fit();
  vcycle_diag_inv_.clear();
  vcycle_diag_inv_.shrink_to_fit();
  vcycle_chebyshev_omega_.clear();
  vcycle_chebyshev_omega_.shrink_to_fit();
  vcycle_hierarchy_built_ = false;
  vcycle_num_levels_ = 0;

  // Release coarse solver
  coarse_solver_.reset();
  coarse_solver_valid_ = false;
  coarse_solver_name_ = "";

  if (verbose) {
    std::cout << "[CPP] clearHierarchy: V-cycle state reset. built=false"
              << std::endl;
  }
}


// Generate a cube tetrahedral mesh with N subdivisions per axis
TetrahedralMesh TetMultigridSolver::generateCubeMesh(int resolution) {
  int N = resolution;
  if (N < 2)
    N = 2; // Minimum 2 points to make a cell

  int numVertices = N * N * N;
  int numCells = (N - 1) * (N - 1) * (N - 1);
  int numTets = numCells * 6;

  TetrahedralMesh mesh(numVertices, numTets);

  // Grid spacing
  double dx = 1.0 / (N - 1);

  // Generate vertices
  for (int z = 0; z < N; ++z) {
    for (int y = 0; y < N; ++y) {
      for (int x = 0; x < N; ++x) {
        int idx = x + y * N + z * N * N;
        mesh.vertices(idx, 0) = x * dx;
        mesh.vertices(idx, 1) = y * dx;
        mesh.vertices(idx, 2) = z * dx;
      }
    }
  }

  // Generate tetrahedra (Kuhn triangulation)
  int tetIdx = 0;
  for (int z = 0; z < N - 1; ++z) {
    for (int y = 0; y < N - 1; ++y) {
      for (int x = 0; x < N - 1; ++x) {
        // 8 corners of the cell
        int v0 = x + y * N + z * N * N;
        int v1 = (x + 1) + y * N + z * N * N;
        int v2 = x + (y + 1) * N + z * N * N;
        int v3 = (x + 1) + (y + 1) * N + z * N * N;
        int v4 = x + y * N + (z + 1) * N * N;
        int v5 = (x + 1) + y * N + (z + 1) * N * N;
        int v6 = x + (y + 1) * N + (z + 1) * N * N;
        int v7 = (x + 1) + (y + 1) * N + (z + 1) * N * N;

        // 6 tetrahedra
        mesh.tetrahedra.row(tetIdx++) << v0, v1, v3, v7;
        mesh.tetrahedra.row(tetIdx++) << v0, v5, v1, v7;
        mesh.tetrahedra.row(tetIdx++) << v0, v4, v5, v7;
        mesh.tetrahedra.row(tetIdx++) << v0, v6, v4, v7;
        mesh.tetrahedra.row(tetIdx++) << v0, v2, v6, v7;
        mesh.tetrahedra.row(tetIdx++) << v0, v3, v2, v7;
      }
    }
  }

  return mesh;
}


// =============================================================================
// SOLID ANGLE APPROACH FOR EXTERIOR/INTERIOR CLASSIFICATION
// =============================================================================

/**
 * @brief Computes the solid angle at a vertex of a tetrahedron.
 *
 * The solid angle at vertex v0 is the area on a unit sphere subtended by the
 * triangular face opposite to v0. Uses the formula based on the triple product
 * and edge lengths.
 *
 * Formula: Omega = 2 * atan2(|a.(bxc)|, |a||b||c| + |a|(b.c) + |b|(a.c) + |c|(a.b))
 * where a, b, c are vectors from v0 to v1, v2, v3.
 *
 * @param v0 The vertex at which to compute the solid angle
 * @param v1, v2, v3 The other three vertices of the tetrahedron
 * @return Solid angle in steradians (0 to 2pi for a valid tet corner)
 */
double TetMultigridSolver::computeTetSolidAngle(
    const Eigen::RowVector3d &v0, const Eigen::RowVector3d &v1,
    const Eigen::RowVector3d &v2, const Eigen::RowVector3d &v3) {
  // Vectors from v0 to the other vertices
  Eigen::Vector3d a = (v1 - v0).transpose();
  Eigen::Vector3d b = (v2 - v0).transpose();
  Eigen::Vector3d c = (v3 - v0).transpose();

  double a_norm = a.norm();
  double b_norm = b.norm();
  double c_norm = c.norm();

  // Handle degenerate cases
  if (a_norm < 1e-14 || b_norm < 1e-14 || c_norm < 1e-14) {
    return 0.0;
  }

  // Triple scalar product: a . (b x c)
  double tripleProduct = a.dot(b.cross(c));

  // Denominator: |a||b||c| + |a|(b.c) + |b|(a.c) + |c|(a.b)
  double ab = a.dot(b);
  double ac = a.dot(c);
  double bc = b.dot(c);

  double denom =
      a_norm * b_norm * c_norm + a_norm * bc + b_norm * ac + c_norm * ab;

  // Solid angle using atan2 for numerical stability
  double solidAngle = 2.0 * std::atan2(tripleProduct, denom);

  // Ensure non-negative (solid angle should be positive for properly oriented
  // tets)
  return std::abs(solidAngle);
}

/**
 * @brief Computes exterior vertices using solid angle sum approach.
 *
 * For each vertex, sums the solid angles of all incident tetrahedra.
 * A vertex is considered exterior if its solid angle sum is significantly
 * less than 4pi (the full sphere).
 *
 * This method naturally handles:
 * - Isolated vertices (no tets -> sum = 0 -> exterior)
 * - Vertices on floating triangles (no tets -> exterior)
 * - Non-manifold configurations
 *
 * @param mesh          Tetrahedral mesh
 * @param solidAngleSums Output: solid angle sum for each vertex
 * @param threshold     Fraction of 4pi below which a vertex is exterior (default
 * 0.99)
 * @return Indices of exterior vertices, sorted by solid angle sum ascending
 * (sharpest first)
 */
std::vector<int>
TetMultigridSolver::computeExteriorVerticesBySolidAngle(
    const TetrahedralMesh &mesh, std::vector<double> &solidAngleSums,
    double threshold) {
  const int numVerts = mesh.vertices.rows();
  const int numTets = mesh.tetrahedra.rows();
  const Eigen::MatrixXd &verts = mesh.vertices;
  const Eigen::MatrixXi &tets = mesh.tetrahedra;

  // Initialize solid angle sums to zero
  solidAngleSums.assign(numVerts, 0.0);

  // For thread safety, use per-thread accumulators
  int max_threads = omp_get_max_threads();
  std::vector<std::vector<double>> threadSums(max_threads);
  for (int t = 0; t < max_threads; ++t) {
    threadSums[t].assign(numVerts, 0.0);
  }

// Compute solid angles for all vertices in all tetrahedra
#pragma omp parallel for
  for (int tetIdx = 0; tetIdx < numTets; ++tetIdx) {
    int tid = omp_get_thread_num();

    int v0 = tets(tetIdx, 0);
    int v1 = tets(tetIdx, 1);
    int v2 = tets(tetIdx, 2);
    int v3 = tets(tetIdx, 3);

    Eigen::RowVector3d p0 = verts.row(v0);
    Eigen::RowVector3d p1 = verts.row(v1);
    Eigen::RowVector3d p2 = verts.row(v2);
    Eigen::RowVector3d p3 = verts.row(v3);

    // Compute solid angle at each vertex of this tet
    threadSums[tid][v0] += computeTetSolidAngle(p0, p1, p2, p3);
    threadSums[tid][v1] += computeTetSolidAngle(p1, p0, p2, p3);
    threadSums[tid][v2] += computeTetSolidAngle(p2, p0, p1, p3);
    threadSums[tid][v3] += computeTetSolidAngle(p3, p0, p1, p2);
  }

// Merge thread-local sums
#pragma omp parallel for
  for (int i = 0; i < numVerts; ++i) {
    double sum = 0.0;
    for (int t = 0; t < max_threads; ++t) {
      sum += threadSums[t][i];
    }
    solidAngleSums[i] = sum;
  }

  // 4pi is the solid angle of a complete sphere
  const double FOUR_PI = 4.0 * M_PI;
  const double exteriorThreshold = threshold * FOUR_PI;

  // Find exterior vertices (solid angle sum < threshold * 4pi)
  std::vector<int> exteriorIndices;
  exteriorIndices.reserve(numVerts / 10); // Estimate ~10% are exterior

  for (int i = 0; i < numVerts; ++i) {
    if (solidAngleSums[i] < exteriorThreshold) {
      exteriorIndices.push_back(i);
    }
  }

  // NOTE: Do NOT sort here - sorting is done separately via
  // sortPriorityIndicesBySolidAngle for feature_priority mode. For
  // surface_priority, we want arbitrary order.

  if (verbose) {
    std::cout << "  Solid angle classification:\n";
    std::cout << "    Total vertices: " << numVerts << "\n";
    std::cout << "    Exterior vertices: " << exteriorIndices.size() << "\n";
    std::cout << "    Interior vertices: "
              << (numVerts - exteriorIndices.size()) << "\n";
    if (!exteriorIndices.empty()) {
      std::cout << "    Min solid angle sum: "
                << solidAngleSums[exteriorIndices.front()] << " (vertex "
                << exteriorIndices.front() << ")\n";
      std::cout << "    Max exterior solid angle sum: "
                << solidAngleSums[exteriorIndices.back()] << "\n";
    }
    std::cout << "    Threshold: " << exteriorThreshold << " ("
              << (threshold * 100) << "% of 4pi)\n";
  }

  return exteriorIndices;
}

/**
 * @brief Computes the dihedral angle at an edge of a tetrahedron.
 *
 * The dihedral angle is the angle between the two planes meeting at the edge.
 * It is computed as the angle between the projections of the two non-edge
 * vertices onto the plane perpendicular to the edge.
 *
 * @param v0, v1 The vertices defining the edge
 * @param v2, v3 The other two vertices of the tetrahedron
 * @return Dihedral angle in radians (0 to pi)
 */
double TetMultigridSolver::computeTetDihedralAngle(
    const Eigen::RowVector3d &v0, const Eigen::RowVector3d &v1,
    const Eigen::RowVector3d &v2, const Eigen::RowVector3d &v3) {
  // Edge direction
  Eigen::Vector3d edge = (v1 - v0).transpose();
  double edgeLen = edge.norm();

  if (edgeLen < 1e-14)
    return 0.0;
  edge /= edgeLen;

  // Vectors to other vertices
  Eigen::Vector3d r2 = (v2 - v0).transpose();
  Eigen::Vector3d r3 = (v3 - v0).transpose();

  // Project onto plane perpendicular to edge
  // v_proj = v - (v . edge) * edge
  Eigen::Vector3d p2 = r2 - r2.dot(edge) * edge;
  Eigen::Vector3d p3 = r3 - r3.dot(edge) * edge;

  double p2Len = p2.norm();
  double p3Len = p3.norm();

  if (p2Len < 1e-14 || p3Len < 1e-14)
    return 0.0;

  p2 /= p2Len;
  p3 /= p3Len;

  // Angle between projections
  double cosAngle = p2.dot(p3);
  cosAngle = std::max(-1.0, std::min(1.0, cosAngle)); // Clamp for stability

  return std::acos(cosAngle);
}

// =============================================================================
// EXTERIOR GRAPH: Solid-Angle Classification + Connectivity-Based Dihedral
// =============================================================================
//
// Builds the exterior subgraph and per-vertex sharpness feature for every
// hierarchy level that has not yet been processed.
//
//   Phase 1: Solid angle sums per vertex (parallel per-tet, no hash maps).
//            A vertex is exterior iff its solid angle sum is below a fraction
//            of 4*pi.
//   Phase 2: Exterior edges from the neighbour matrix (parallel per-vertex)
//            + dihedral angle sum filter: an edge is exterior iff the dihedral
//            angles around it sum to less than 2*pi.
//   Phase 3: Per-vertex feature: for each exterior edge incident to v, find
//            the two triangular faces via shared exterior neighbours and sum
//            the angles between adjacent face normals (parallel per-vertex).
//
// Populates: allExteriorVertices, allExteriorNeigh, allExteriorIndices,
// allExteriorFeature for every newly-processed level.
//
void TetMultigridSolver::buildExteriorGraph() {
  // Catch-up loop: process any levels that haven't been processed yet
  while (allExteriorVertices.size() < all_vertices.size()) {
    size_t level = allExteriorVertices.size();

    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
      std::cout << "  Building Exterior Graph for Level " << level << " ... ";
      std::cout.flush();
    }

    const Eigen::MatrixXd &vertices = all_vertices[level];
    const Eigen::MatrixXi &tets = all_tetrahedra[level];
    const Eigen::MatrixXi &neigh = allNeigh[level];
    int numVerts = vertices.rows();
    int numTets = tets.rows();

    // =========================================================================
    // PHASE 1: Identify Exterior Vertices via Solid Angle Sum
    // =========================================================================
    // For each vertex, sum the solid angles of all incident tets.
    // Interior: sum approx 4pi.  Exterior: sum < 4pi.
    // Fully parallel with per-thread accumulators (no hashing).

    int maxThreads = omp_get_max_threads();
    std::vector<std::vector<double>> threadSums(maxThreads);
    for (int t = 0; t < maxThreads; ++t)
      threadSums[t].assign(numVerts, 0.0);

    #pragma omp parallel for schedule(static)
    for (int tetIdx = 0; tetIdx < numTets; ++tetIdx) {
      int tid = omp_get_thread_num();
      int v0 = tets(tetIdx, 0), v1 = tets(tetIdx, 1);
      int v2 = tets(tetIdx, 2), v3 = tets(tetIdx, 3);
      Eigen::RowVector3d p0 = vertices.row(v0), p1 = vertices.row(v1);
      Eigen::RowVector3d p2 = vertices.row(v2), p3 = vertices.row(v3);
      threadSums[tid][v0] += computeTetSolidAngle(p0, p1, p2, p3);
      threadSums[tid][v1] += computeTetSolidAngle(p1, p0, p2, p3);
      threadSums[tid][v2] += computeTetSolidAngle(p2, p0, p1, p3);
      threadSums[tid][v3] += computeTetSolidAngle(p3, p0, p1, p2);
    }

    // Merge per-thread sums and classify
    std::vector<double> solidAngleSums(numVerts, 0.0);
    const double FOUR_PI = 4.0 * M_PI;
    const double exteriorThreshold = 0.99 * FOUR_PI;
    std::vector<bool> isExterior(numVerts, false);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numVerts; ++i) {
      double sum = 0.0;
      for (int t = 0; t < maxThreads; ++t)
        sum += threadSums[t][i];
      solidAngleSums[i] = sum;
      isExterior[i] = (sum < exteriorThreshold);
    }
    threadSums.clear(); // free memory

    // Collect exterior indices (serial -- just a compact gather)
    std::vector<int> exteriorIndices;
    exteriorIndices.reserve(numVerts / 8);
    for (int i = 0; i < numVerts; ++i) {
      if (isExterior[i])
        exteriorIndices.push_back(i);
    }

    int numExterior = static_cast<int>(exteriorIndices.size());

    // Global -> local mapping (O(1) via dense vector)
    std::vector<int> globalToLocal(numVerts, -1);
    for (int i = 0; i < numExterior; ++i)
      globalToLocal[exteriorIndices[i]] = i;

    // =========================================================================
    // PHASE 2: Identify Exterior Edges + Build Adjacency
    // =========================================================================
    // Scan the neighbour matrix: for each exterior vertex, check which
    // neighbours are also exterior.  This gives candidate exterior edges.
    //
    // Filter using dihedral angle sum around each edge:
    //   interior edge -> sum approx 2pi,  exterior edge -> sum < 2pi
    //
    // Dihedral angle sum is accumulated from tets.  We build a per-edge
    // accumulator in parallel using the tet array (similar to solid angle
    // phase).
    //
    // Edge key: pack sorted (u,v) into uint64_t for O(1) lookup.

    auto packEdge = [](int a, int b) -> uint64_t {
      if (a > b) std::swap(a, b);
      return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    // Step 2a: Collect all candidate exterior-exterior edges from neigh matrix
    // (parallel per exterior vertex, then merge + deduplicate)
    std::vector<std::vector<uint64_t>> threadEdges(maxThreads);

    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto &localEdges = threadEdges[tid];

      #pragma omp for schedule(static)
      for (int ei = 0; ei < numExterior; ++ei) {
        int gv = exteriorIndices[ei];
        int numCols = neigh.cols();
        for (int c = 1; c < numCols; ++c) { // skip col 0 (self)
          int gn = neigh(gv, c);
          if (gn < 0) break;
          if (isExterior[gn] && gv < gn) { // only store once (u < v)
            localEdges.push_back(packEdge(gv, gn));
          }
        }
      }
    }

    // Merge + deduplicate
    std::vector<uint64_t> candidateEdges;
    {
      size_t total = 0;
      for (auto &v : threadEdges) total += v.size();
      candidateEdges.reserve(total);
      for (auto &v : threadEdges) {
        candidateEdges.insert(candidateEdges.end(), v.begin(), v.end());
        v.clear(); v.shrink_to_fit();
      }
      threadEdges.clear();
      PARALLEL_SORT(candidateEdges.begin(), candidateEdges.end());
      candidateEdges.erase(
          std::unique(candidateEdges.begin(), candidateEdges.end()),
          candidateEdges.end());
    }

    int numCandEdges = static_cast<int>(candidateEdges.size());

    // Step 2b: Accumulate dihedral angle sums for candidate edges
    // Build a lookup: edge key -> index in candidateEdges (for O(1) accumulation)
    std::unordered_map<uint64_t, int> edgeKeyToIdx;
    edgeKeyToIdx.reserve(numCandEdges * 2);
    for (int i = 0; i < numCandEdges; ++i)
      edgeKeyToIdx[candidateEdges[i]] = i;

    // Per-thread dihedral angle accumulators
    std::vector<std::vector<double>> threadDihedral(maxThreads);
    for (int t = 0; t < maxThreads; ++t)
      threadDihedral[t].assign(numCandEdges, 0.0);

    #pragma omp parallel for schedule(static)
    for (int tetIdx = 0; tetIdx < numTets; ++tetIdx) {
      int tid = omp_get_thread_num();
      int v[4] = {tets(tetIdx, 0), tets(tetIdx, 1),
                  tets(tetIdx, 2), tets(tetIdx, 3)};

      // 6 edges per tet
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          if (!isExterior[v[i]] || !isExterior[v[j]])
            continue; // skip non-exterior edges early

          uint64_t key = packEdge(v[i], v[j]);
          auto it = edgeKeyToIdx.find(key);
          if (it == edgeKeyToIdx.end())
            continue;

          // Other two vertices
          int k = 0; while (k == i || k == j) k++;
          int l = k + 1; while (l == i || l == j) l++;

          double angle = computeTetDihedralAngle(
              vertices.row(v[i]), vertices.row(v[j]),
              vertices.row(v[k]), vertices.row(v[l]));

          threadDihedral[tid][it->second] += angle;
        }
      }
    }

    // Merge dihedral sums and filter
    const double TWO_PI = 2.0 * M_PI;
    const double edgeThreshold = 0.99 * TWO_PI;

    // Build per-vertex adjacency lists (local indices) in parallel
    // First pass: determine which edges are exterior
    std::vector<bool> isExteriorEdge(numCandEdges, false);

    #pragma omp parallel for schedule(static)
    for (int e = 0; e < numCandEdges; ++e) {
      double sum = 0.0;
      for (int t = 0; t < maxThreads; ++t)
        sum += threadDihedral[t][e];
      isExteriorEdge[e] = (sum < edgeThreshold);
    }
    threadDihedral.clear();

    // Build adjacency lists from exterior edges
    std::vector<std::vector<int>> adjList(numExterior);
    for (int e = 0; e < numCandEdges; ++e) {
      if (!isExteriorEdge[e])
        continue;
      uint64_t key = candidateEdges[e];
      int gu = static_cast<int>(key >> 32);
      int gv = static_cast<int>(key & 0xFFFFFFFF);
      int lu = globalToLocal[gu];
      int lv = globalToLocal[gv];
      adjList[lu].push_back(lv);
      adjList[lv].push_back(lu);
    }

    // Sort adjacency lists (parallel)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numExterior; ++i) {
      std::sort(adjList[i].begin(), adjList[i].end());
      // No duplicates possible since we process each edge once, but just in case:
      adjList[i].erase(std::unique(adjList[i].begin(), adjList[i].end()),
                       adjList[i].end());
    }

    int numExteriorEdges = 0;
    for (int e = 0; e < numCandEdges; ++e)
      if (isExteriorEdge[e]) numExteriorEdges++;

    // =========================================================================
    // PHASE 3: Compute Normal-Angle-Sum Feature (connectivity-based)
    // =========================================================================
    // For each exterior vertex v and each exterior neighbour u, the edge (v,u)
    // has two adjacent triangular faces on the surface.  We find these faces
    // by looking for shared exterior neighbours: if w is an exterior neighbour
    // of BOTH v and u, then (v, u, w) forms a surface triangle.
    //
    // For a manifold exterior surface, each edge has exactly 2 shared
    // neighbours -> 2 faces -> 1 dihedral angle (angle between normals).
    // The feature for vertex v is the sum of these angles over all its edges.
    //
    // This is fully parallel per vertex, no hash maps needed.

    // Pre-build sets for fast shared-neighbour queries
    // Using sorted vectors + set_intersection (cache-friendly)
    // adjList is already sorted from Phase 2.

    std::vector<double> exteriorFeatures(numExterior, 0.0);

    // Build vertex positions for exterior vertices
    Eigen::MatrixXd extVerts(numExterior, 3);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numExterior; ++i)
      extVerts.row(i) = vertices.row(exteriorIndices[i]);

    std::atomic<int> failureCount(0);
    int maxFailuresToPrint = 5;

    #pragma omp parallel for schedule(dynamic)
    for (int li = 0; li < numExterior; ++li) {
      int gi = exteriorIndices[li];
      double angleSum = 0.0;

      // Lambda to find the 4th vertex of the tetrahedron formed by (u, v, w).
      // k must be a neighbor of u, v, AND w in the volume graph (neigh).
      // Since max degree is small (~15-20), linear scans are fast enough.
      auto findFourthVertex = [&](int u, int v, int w) -> int {
        for (int c = 0; c < neigh.cols(); ++c) {
          int k = neigh(u, c);
          if (k < 0) break;
          if (k == u || k == v || k == w) continue;

          // Check if k is neighbor of v
          bool isNeighV = false;
          for (int cv = 0; cv < neigh.cols(); ++cv) {
             if (neigh(v, cv) == k) { isNeighV = true; break; }
             if (neigh(v, cv) < 0) break;
          }
          if (!isNeighV) continue;

          // Check if k is neighbor of w
          bool isNeighW = false;
          for (int cw = 0; cw < neigh.cols(); ++cw) {
             if (neigh(w, cw) == k) { isNeighW = true; break; }
             if (neigh(w, cw) < 0) break;
          }
          if (isNeighW) return k;
        }
        return -1;
      };

      for (int lj : adjList[li]) {
        if (lj <= li) continue; // process each edge once to avoid double-counting

        int gj = exteriorIndices[lj];

        // Find shared exterior neighbours via sorted intersection
        std::vector<int> shared;
        std::set_intersection(
            adjList[li].begin(), adjList[li].end(),
            adjList[lj].begin(), adjList[lj].end(),
            std::back_inserter(shared));

        if (shared.size() < 2) {
          int f = failureCount.fetch_add(1);
          if (f < maxFailuresToPrint && verbose) {
             #pragma omp critical
             std::cout << "  [Warn] Edge (" << gi << "," << gj << ") has " << shared.size() << " shared neighbors (boundary edge?)\n";
          }
          continue;
        }

        // For a manifold surface edge, there are exactly 2 shared neighbours
        Eigen::Vector3d pi = vertices.row(gi).transpose();
        Eigen::Vector3d pj = vertices.row(gj).transpose();

        int gw0 = exteriorIndices[shared[0]];
        int gw1 = exteriorIndices[shared[1]];
        Eigen::Vector3d pw0 = vertices.row(gw0).transpose();
        Eigen::Vector3d pw1 = vertices.row(gw1).transpose();

        // Compute normals
        Eigen::Vector3d n1 = (pj - pi).cross(pw0 - pi);
        Eigen::Vector3d n2 = (pj - pi).cross(pw1 - pi);
        double len1 = n1.norm(), len2 = n2.norm();

        if (len1 < 1e-14 || len2 < 1e-14) continue;
        n1 /= len1;
        n2 /= len2;

        // Orient Normal 1 using 4th vertex
        int gk0 = findFourthVertex(gi, gj, gw0);
        if (gk0 != -1) {
            Eigen::Vector3d pk0 = vertices.row(gk0).transpose();
            Eigen::Vector3d v_in = pk0 - pi; 
            if (n1.dot(v_in) > 0) n1 = -n1; 
        } else {
            int f = failureCount.fetch_add(1);
            if (f < maxFailuresToPrint && verbose) {
                #pragma omp critical
                std::cout << "  [Warn] findFourthVertex failed for Triangle(" << gi << "," << gj << "," << gw0 << ")\n";
            }
        }

        // Orient Normal 2 using 4th vertex
        int gk1 = findFourthVertex(gi, gj, gw1);
        if (gk1 != -1) {
            Eigen::Vector3d pk1 = vertices.row(gk1).transpose();
            Eigen::Vector3d v_in = pk1 - pi; 
            if (n2.dot(v_in) > 0) n2 = -n2; 
        } else {
            int f = failureCount.fetch_add(1);
            if (f < maxFailuresToPrint && verbose) {
                #pragma omp critical
                std::cout << "  [Warn] findFourthVertex failed for Triangle(" << gi << "," << gj << "," << gw1 << ")\n";
            }
        }

        double dot = std::max(-1.0, std::min(1.0, n1.dot(n2)));
        double angle = std::acos(dot); 

        angleSum += angle;
        #pragma omp atomic
        exteriorFeatures[lj] += angle;
      }

      #pragma omp atomic
      exteriorFeatures[li] += angleSum;
    }
    
    if (verbose && failureCount > 0) {
        std::cout << "  [Warn] findFourthVertex failed " << failureCount << " times.\n";
    }

    // =========================================================================
    // PHASE 4: Convert to Output Format (MatrixXi neighbour matrix)
    // =========================================================================
    int maxDegree = 0;
    for (int i = 0; i < numExterior; ++i)
      maxDegree = std::max(maxDegree, static_cast<int>(adjList[i].size()));

    Eigen::MatrixXi extNeigh(numExterior, maxDegree + 1);
    extNeigh.setConstant(-1);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numExterior; ++i) {
      extNeigh(i, 0) = i; // self
      for (size_t j = 0; j < adjList[i].size(); ++j)
        extNeigh(i, static_cast<int>(j) + 1) = adjList[i][j];
    }

    // Store results
    allExteriorVertices.push_back(extVerts);
    allExteriorNeigh.push_back(extNeigh);
    allExteriorFeature.push_back(exteriorFeatures);
    allExteriorIndices.push_back(exteriorIndices);

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (verbose) {
      std::cout << "Done. " << numExterior << " verts, "
                << numExteriorEdges << " edges. (" << ms << " ms)\n";
    }
  }
}

/**
 * @brief Sorts priority indices by solid angle sum (ascending = sharpest
 * first).
 *
 * Vertices with lower solid angle sums have less "surrounding" material,
 * indicating sharper features (corners have very low sums, flat surface points
 * have sums close to 2pi).
 */
void TetMultigridSolver::sortPriorityIndicesBySolidAngle(
    std::vector<int> &Exterior_indices,
    const std::vector<double> &solidAngleSums) {
  std::sort(Exterior_indices.begin(), Exterior_indices.end(),
            [&solidAngleSums](int a, int b) {
              return solidAngleSums[a] < solidAngleSums[b];
            });
}

/**
 * @brief Sorts exterior indices for a given level by feature value (descending).
 * 
 * Used to prioritize sampling of sharp boundary features.
 * Updates allExteriorIndices[level] in-place.
 */
void TetMultigridSolver::sortExteriorIndicesByFeature(int level) {
    if (verbose) std::cout << "  [Sort] Sorting exterior indices for level " << level << " by feature (descending)...\n";
    
    // Safety checks
    if (level < 0 || level >= allExteriorIndices.size() || level >= allExteriorFeature.size()) {
        if (verbose) std::cout << "  [Sort] Level " << level << " data missing. Skipping sort.\n";
        return;
    }
    
    std::vector<int>& indices = allExteriorIndices[level];
    std::vector<double>& features = allExteriorFeature[level];
    
    if (indices.empty() || features.empty() || indices.size() != features.size()) {
        if (verbose) std::cout << "  [Sort] Mismatched or empty data. indices=" << indices.size() 
                               << ", features=" << features.size() << ". Skipping sort.\n";
        return;
    }

    // Create permutation
    std::vector<size_t> p(indices.size());
    for(size_t i=0; i<p.size(); ++i) p[i] = i;

    // Sort permutation based on feature values (descending)
    std::sort(p.begin(), p.end(), [&](size_t i, size_t j){
        return features[i] > features[j]; 
    });

    // Apply permutation to both vectors to keep them aligned
    std::vector<int> sortedIndices(indices.size());
    std::vector<double> sortedFeatures(features.size());
    for (size_t i = 0; i < p.size(); ++i) {
        sortedIndices[i] = indices[p[i]];
        sortedFeatures[i] = features[p[i]];
    }
    
    indices = sortedIndices;
    features = sortedFeatures;

    // DEBUG: Print top 10 sorted indices and their features/positions
    if (verbose) {
        std::cout << "  [Sort] Top 10 sorted indices (Level " << level << "):\n";
        int count = 0;
        const Eigen::MatrixXd& verts = all_vertices[level];
        for (size_t i = 0; i < indices.size() && count < 10; ++i) {
            int idx = indices[i];
            double feat = features[i];
            std::cout << "    Rank " << count << ": Idx " << idx << ", Feat " << feat 
                      << ", Pos (" << verts(idx, 0) << ", " << verts(idx, 1) << ", " << verts(idx, 2) << ")\n";
            count++;
        }
    }
}

// Build neighborhood matrix from tetrahedra connectivity
// Returns Eigen::MatrixXi where each row contains neighbor indices (-1 for no
// more neighbors)

// Helper for cross-platform atomic fetch-add
#ifdef _MSC_VER
#include <intrin.h>
#endif

Eigen::MatrixXi TetMultigridSolver::buildNeighborMatrixFromTetrahedra(
    const Eigen::MatrixXi &tetrahedra, int numVertices) {

  // ==========================================================
  // Optimzed Parallel Adjacency Construction (Flat CSR)
  // ==========================================================
  // Strategy:
  // 1. Count degrees (upper bound with duplicates) in parallel
  // 2. Compute offsets (prefix sum)
  // 3. Fill flat adjacency array efficiently in parallel
  // 4. Sort and Unique each row in parallel
  // 5. Pack into Eigen Matrix

  int numTets = tetrahedra.rows();

  // 1. Count degrees (upper bound)
  // Using atomic adds on a vector. For typical meshes (avg degree ~14-20),
  // contention is manageable.
  std::vector<int> counts(numVertices, 0);

#pragma omp parallel for
  for (int i = 0; i < numTets; ++i) {
    // Each tet adds 3 neighbors to each of its 4 vertices
    for (int j = 0; j < 4; ++j) {
#pragma omp atomic
      counts[tetrahedra(i, j)] += 3;
    }
  }

  // 2. Compute offsets (Prefix Sum)
  std::vector<int> offsets(numVertices + 1, 0);
  // Sequential prefix sum is extremely fast for N=20k-1M
  for (int i = 0; i < numVertices; ++i) {
    offsets[i + 1] = offsets[i] + counts[i];
  }

  // Total edges (with duplicates)
  int totalEntries = offsets[numVertices];
  std::vector<int> flatAdj(totalEntries);

  // Copy offsets to current_pos to track write position
  // We need atomic increments on positions during fill
  std::vector<int> currentPos = offsets; // Copy

// 3. Fill flat array
#pragma omp parallel for
  for (int i = 0; i < numTets; ++i) {
    int v[4] = {tetrahedra(i, 0), tetrahedra(i, 1), tetrahedra(i, 2),
                tetrahedra(i, 3)};

    // For each vertex, add other 3
    for (int j = 0; j < 4; ++j) {
      int u = v[j];
      // Determine where to write using atomic fetch-add
      int pos;

#ifdef _MSC_VER
      // MSVC OpenMP 2.0 doesn't support capture. Use intrinsic.
      // FIX: We are adding 3 neighbors, so reserve 3 slots!
      pos = _InterlockedExchangeAdd((volatile long *)&currentPos[u], 3);
#else
      // GCC/Clang or newer OpenMP
      pos = __sync_fetch_and_add(&currentPos[u], 3);
#endif

      // Add 3 neighbors
      int idx = 0;
      for (int k = 0; k < 4; ++k) {
        if (j != k) {
          flatAdj[pos + idx] = v[k];
          idx++;
        }
      }
    }
  }

  // 4. Sort, Unique, and Find Max Degree
  // We process each row independently in parallel
  int globalMaxNeigh = 0;

#pragma omp parallel
  {
    int localMax = 0;
#pragma omp for
    for (int i = 0; i < numVertices; ++i) {
      int start = offsets[i];
      int end =
          currentPos[i]; // Use currentPos as strict end (logic matches counts)

      if (start < end) {
        std::sort(flatAdj.begin() + start, flatAdj.begin() + end);
        auto last = std::unique(flatAdj.begin() + start, flatAdj.begin() + end);

        // Compute unique size
        int uniqueDegree =
            static_cast<int>(std::distance(flatAdj.begin() + start, last));

        // We can update counts[i] to store actual unique degree
        counts[i] = uniqueDegree;

        if (uniqueDegree > localMax)
          localMax = uniqueDegree;
      } else {
        counts[i] = 0;
      }
    }
#pragma omp critical
    {
      if (localMax > globalMaxNeigh)
        globalMaxNeigh = localMax;
    }
  }

  // 5. Build Matrix
  Eigen::MatrixXi neigh(numVertices, globalMaxNeigh + 1);
  neigh.setConstant(-1);

#pragma omp parallel for
  for (int i = 0; i < numVertices; ++i) {
    neigh(i, 0) = i;        // Self as first neighbor
    int degree = counts[i]; // unique degree
    int start = offsets[i];

    for (int j = 0; j < degree; ++j) {
      neigh(i, j + 1) = flatAdj[start + j];
    }
  }

  return neigh;
}

// Compute average edge length using neighborhood matrix (matches reference
// implementation)
double TetMultigridSolver::computeAverageEdgeLengthFromNeigh(
    const Eigen::MatrixXd &verts, const Eigen::MatrixXi &neigh) {
  double sumLength = 0;
  int nEdges = 0;
  for (int i = 0; i < verts.rows(); ++i) {
    for (int j = 0; j < neigh.cols(); ++j) {
      int neighIdx = neigh(i, j);
      if (neighIdx < 0)
        break;
      if (i >= neighIdx)
        continue; // Count each edge only once
      double dist = (verts.row(i) - verts.row(neighIdx)).norm();
      if (dist > 0) {
        sumLength += dist;
        ++nEdges;
      }
    }
  }
  return nEdges > 0 ? sumLength / nEdges : 0.0;
}

// =============================================================================
// UNIFIED FAST DISK SAMPLING (with priority support)
// =============================================================================
// Samples vertices using Euclidean distance logic with tetrahedral
// connectivity. If priorityIndices is provided, those are scanned first. Then a
// full linear scan ensures remaining vertices are sampled.
//
// This unified approach avoids code duplication and allows arbitrary
// prioritization.
std::vector<int> TetMultigridSolver::fastDiskSample(
    const Eigen::MatrixXd &pos, const Eigen::MatrixXi &tetNeigh,
    const double &radius, Eigen::VectorXd &shortestDistanceToSample,
    std::vector<size_t> &nearestSourceK,
    std::vector<int> &Exterior_indices) {

  int numVertices = pos.rows();
  std::vector<bool> visited(numVertices, false);
  std::vector<int> samples;
  int sampleIdx = 0;
  std::vector<int> successfulPriorityIndices;

  // Helper lambda to mark neighbors as visited (1-ring and 2-ring)
  // This logic was shared by both previous implementations.
  auto markNeighborsVisited = [&](int i) {
    for (int j = 0; j < tetNeigh.cols(); ++j) {
      int neighIdx = tetNeigh(i, j);
      if (neighIdx < 0)
        break;

      double dist = (pos.row(i) - pos.row(neighIdx)).norm();
      if (dist < radius) {
        visited[neighIdx] = true;
        if (dist < shortestDistanceToSample(neighIdx)) {
          shortestDistanceToSample(neighIdx) = dist;
          nearestSourceK[neighIdx] = sampleIdx;
        }

        // 2-ring check
        for (int j2 = 0; j2 < tetNeigh.cols(); ++j2) {
          int neighIdx2 = tetNeigh(neighIdx, j2);
          if (neighIdx2 < 0)
            break;

          double dist2 = dist + (pos.row(neighIdx) - pos.row(neighIdx2)).norm();
          if (dist2 < radius) {
            visited[neighIdx2] = true;
            if (dist2 < shortestDistanceToSample(neighIdx2)) {
              shortestDistanceToSample(neighIdx2) = dist2;
              nearestSourceK[neighIdx2] = sampleIdx;
            }
          }
        }
      }
    }
  };

  // Phase 1: Sample from priority indices first (if any)
  if (!Exterior_indices.empty()) {
    for (int idx : Exterior_indices) {
      if (idx >= 0 && idx < numVertices && !visited[idx]) {
        samples.push_back(idx);
        nearestSourceK[idx] = sampleIdx;
        shortestDistanceToSample(idx) = 0.0;
        markNeighborsVisited(idx);
        ++sampleIdx;
        successfulPriorityIndices.push_back(idx);
      }
    }
  }

  // Phase 2: Sample from remaining vertices (linear scan)
  for (int i = 0; i < numVertices; ++i) {
    if (!visited[i]) {
      samples.push_back(i);
      nearestSourceK[i] = sampleIdx;
      shortestDistanceToSample(i) = 0.0;
      markNeighborsVisited(i);
      ++sampleIdx;
    }
  }

  // Update Exterior_indices to contain only the indices that were actually sampled
  Exterior_indices = successfulPriorityIndices;

  return samples;
}


// Construct clusters using Dijkstra's algorithm (works for any neighbor matrix)
// Assigns each point to its nearest sample point based on geodesic distance

void TetMultigridSolver::constructDijkstraWithCluster(
    const Eigen::MatrixXd &points, const std::vector<int> &source,
    const Eigen::MatrixXi &neigh, Eigen::VectorXd &shortestDistanceToSample,
    std::vector<size_t> &nearestSource) {
  // Safety check: empty source would cause undefined behavior
  if (source.empty()) {
    std::cerr << "Warning: constructDijkstraWithCluster called with empty "
                 "source vector\n";
    return;
  }

  std::priority_queue<VertexPair, std::vector<VertexPair>,
                      std::greater<VertexPair>>
      DistanceQueue;
  if (nearestSource.empty()) {
    nearestSource.resize(points.rows(), static_cast<size_t>(0));
  }

  // Initialize all source points with distance 0 and add to queue
  for (size_t i = 0; i < source.size(); ++i) {
    shortestDistanceToSample(source[i]) = 0.0;
    VertexPair vp{source[i], shortestDistanceToSample(source[i])};
    DistanceQueue.push(vp);
    nearestSource[source[i]] = i;
  }

  while (!DistanceQueue.empty()) {
    VertexPair vp1 = DistanceQueue.top();
    int curSource = nearestSource[vp1.vId];
    Eigen::RowVector3d vertex1 = points.row(vp1.vId);
    DistanceQueue.pop();

    // Iterate over neighbors
    for (int i = 0; i < neigh.cols(); ++i) {
      int vNeigh = neigh(vp1.vId, i);

      if (vNeigh >= 0) {
        Eigen::RowVector3d vertex2 = points.row(vNeigh);
        double dist = (vertex2 - vertex1).norm();
        double distTemp = vp1.distance + dist;

        if (distTemp < shortestDistanceToSample(vNeigh)) {
          // Assign a new distance
          shortestDistanceToSample(vNeigh) = distTemp;
          VertexPair v2{vNeigh, distTemp};
          DistanceQueue.push(v2);

          // Assign the nearest source to a certain point
          nearestSource[vNeigh] = curSource;
        }
      }
    }
  }
}

// =============================================================================
// TWO-PHASE DIJKSTRA CLUSTERING (EXTERIOR FIRST)
// =============================================================================
// This function implements a two-phase clustering strategy where the exterior
// (surface) is clustered first, in isolation, and then the interior is filled.
//
// Phase 1: Cluster ONLY exterior vertices using paths that stay on the exterior.
//          This ensures surface geodesics are respected and no interior shortcuts are taken.
// Phase 2: Cluster remaining interior vertices.
//          Sources are:
//          1. Interior samples (dist=0)
//          2. Exterior vertices (dist=calculated in Phase 1) acting as boundary sources.
//          Exterior vertices are treated as "locked" and are not updated in Phase 2.
void TetMultigridSolver::constructDijkstraExteriorFirst(
    const Eigen::MatrixXd &points, const std::vector<int> &allSources,
    const Eigen::MatrixXi &volumeNeigh,
    Eigen::VectorXd &shortestDistanceToSample,
    std::vector<size_t> &nearestSource,
    const std::vector<int> &exteriorIndices) {

  // 0. Filter Volume Neighbors to create Exterior Neighbors
  // Create exteriorNeigh by copying volumeNeigh
  // Set entries to -1 if either vertex is NOT exterior
  Eigen::MatrixXi exteriorNeigh = volumeNeigh;

  // 1. Fast lookup for exterior vertices
  std::vector<bool> isExterior(points.rows(), false);
  for (int idx : exteriorIndices) {
    if (idx >= 0 && idx < points.rows()) {
        isExterior[idx] = true;
    }
  }

  // Filter: remove edges connecting to interior
  for (int i = 0; i < exteriorNeigh.rows(); ++i) {
    for (int j = 0; j < exteriorNeigh.cols(); ++j) {
        int neighbor = exteriorNeigh(i, j);
        if (neighbor < 0) continue;
        
        // Keep edge only if BOTH are exterior
        if (!isExterior[i] || !isExterior[neighbor]) {
            exteriorNeigh(i, j) = -1;
        }
    }
  }

  // Fast lookup for priority sources (exterior vertices)

  // Fast lookup for priority sources (exterior vertices)
  // Not strictly needed since we use isExterior, but good for filtering samples
  std::unordered_set<int> exteriorSet(exteriorIndices.begin(),
                                      exteriorIndices.end());

  // Build mapping from source vertex index to its position in allSources
  std::unordered_map<int, size_t> sourceToIndex;
  for (size_t i = 0; i < allSources.size(); ++i) {
    sourceToIndex[allSources[i]] = i;
  }

  std::priority_queue<VertexPair, std::vector<VertexPair>,
                      std::greater<VertexPair>>
      DistanceQueue;

  // =========================================================
  // Phase 1: Exterior Only
  // =========================================================
  // Sources: Exterior samples
  // Propagation: Only through exterior neighbor graph
  
  for (int srcVertex : allSources) {
      if (isExterior[srcVertex]) { // It is an exterior sample
          shortestDistanceToSample(srcVertex) = 0.0;
          nearestSource[srcVertex] = sourceToIndex[srcVertex];
          DistanceQueue.push({srcVertex, 0.0});
      }
  }
  
  while (!DistanceQueue.empty()) {
      VertexPair vp = DistanceQueue.top();
      DistanceQueue.pop();
       
      // Standard Dijkstra check
      if (vp.distance > shortestDistanceToSample(vp.vId)) continue;
      
      size_t curSourceIndex = nearestSource[vp.vId];
      Eigen::RowVector3d p1 = points.row(vp.vId);

      // Expand to neighbors using EXTERIOR graph
      for (int i = 0; i < exteriorNeigh.cols(); ++i) {
          int vNeigh = exteriorNeigh(vp.vId, i);
          if (vNeigh < 0) break;
          
          // Note: In the filtered graph approach, we shouldn't even encounter internal neighbors here,
          // but the check implies validity.
          
          Eigen::RowVector3d p2 = points.row(vNeigh);
          double d = (p2 - p1).norm();
          double newDist = vp.distance + d;
          
          if (newDist < shortestDistanceToSample(vNeigh)) {
              shortestDistanceToSample(vNeigh) = newDist;
              nearestSource[vNeigh] = curSourceIndex;
              DistanceQueue.push({vNeigh, newDist});
          }
      }
  }

  // =========================================================
  // Phase 2: Interior Expansion
  // =========================================================
  // Sources:
  //  A. Interior Samples (dist=0).
  //  B. Exterior Vertices that were assigned in Phase 1 (dist=current).
  //     These act as boundary sources for the interior.
  
  // Clear Q for safety (it should be empty)
  while(!DistanceQueue.empty()) DistanceQueue.pop();

  // Add Interior Samples
  for (int srcVertex : allSources) {
      if (!isExterior[srcVertex]) {
          shortestDistanceToSample(srcVertex) = 0.0;
          nearestSource[srcVertex] = sourceToIndex[srcVertex];
          DistanceQueue.push({srcVertex, 0.0});
      }
  }

  // Add ASSIGNED Exterior Vertices (Frontier)
  // We iterate all exterior indices to check if they have a valid assignment
  // from Phase 1. If they do, they can propagate into the interior.
  for (int vExt : exteriorIndices) {
      if (vExt >= 0 && vExt < points.rows()) {
           double d = shortestDistanceToSample(vExt);
           if (d < std::numeric_limits<double>::max()) {
               // Push with current distance so they can propagate
               DistanceQueue.push({vExt, d});
           }
      }
  }

  while (!DistanceQueue.empty()) {
      VertexPair vp = DistanceQueue.top();
      DistanceQueue.pop();

      if (vp.distance > shortestDistanceToSample(vp.vId)) continue;

      size_t curSourceIndex = nearestSource[vp.vId];
      Eigen::RowVector3d p1 = points.row(vp.vId);

      // Expand to neighbors using VOLUME graph
      for (int i = 0; i < volumeNeigh.cols(); ++i) {
          int vNeigh = volumeNeigh(vp.vId, i);
          if (vNeigh < 0) break;
          
          // CRITICAL: Do NOT update exterior vertices in Phase 2.
          // They are locked to the surface clustering from Phase 1.
          if (isExterior[vNeigh]) continue;

          Eigen::RowVector3d p2 = points.row(vNeigh);
          double d = (p2 - p1).norm();
          double newDist = vp.distance + d;

          if (newDist < shortestDistanceToSample(vNeigh)) {
              shortestDistanceToSample(vNeigh) = newDist;
              nearestSource[vNeigh] = curSourceIndex;
              DistanceQueue.push({vNeigh, newDist});
          }
      }
  }
}

// Create neighborhood list for coarser level (matches reference implementation)
// This is a pure graph operation - works on any graph connectivity, not just
// surface meshes For each fine point, if it has a neighbor that belongs to a
// different coarse cluster, those two coarse clusters become neighbors in the
// coarser graph
// Create neighborhood list for coarser level (matches reference implementation)
// This is a pure graph operation - works on any graph connectivity
// Returns adjacency list where each row contains SORTED neighbor indices
std::vector<std::vector<int>> TetMultigridSolver::buildCoarseGraph(
    const Eigen::MatrixXi &fineNeigh, const std::vector<size_t> &nearestSource,
    int numCoarsePoints) {

  // Optimization: Use Flat Edge List to avoid massive vector<vector> allocation
  // overhead This reduces memory fragmentation and overhead significantly for
  // large meshes

  struct Edge {
    int u, v;
    bool operator<(const Edge &other) const {
      if (u != other.u)
        return u < other.u;
      return v < other.v;
    }
    bool operator==(const Edge &other) const {
      return u == other.u && v == other.v;
    }
  };

  int numThreads = omp_get_max_threads();
  std::vector<std::vector<Edge>> threadEdges(numThreads);

// Phase 1: Collect raw edges in parallel
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    // Estimate reserve size: avg degree ~14, assume fraction cross boundaries
    // 2 edges per vertex is a safe conservative initial estimate for coarse
    // graph
    threadEdges[tid].reserve(fineNeigh.rows() * 2 / numThreads);

#pragma omp for
    for (int fineIdx = 0; fineIdx < fineNeigh.rows(); ++fineIdx) {
      size_t c_i = nearestSource[fineIdx];

      for (int j = 0; j < fineNeigh.cols(); ++j) {
        int neighIdx = fineNeigh(fineIdx, j);
        if (neighIdx < 0)
          break;

        size_t c_j = nearestSource[neighIdx];

        if (c_i != c_j) {
          // Add edge c_i -> c_j
          threadEdges[tid].push_back({(int)c_i, (int)c_j});
        }
      }
    }
  }

  // Phase 2: Merge thread lists
  std::vector<Edge> allEdges;
  size_t totalEdges = 0;
  for (const auto &list : threadEdges)
    totalEdges += list.size();
  allEdges.reserve(totalEdges);

  for (const auto &list : threadEdges) {
    allEdges.insert(allEdges.end(), list.begin(), list.end());
  }

  // Phase 3: Parallel Sort
  PARALLEL_SORT(allEdges.begin(), allEdges.end());

  // Phase 4: Build CSR-like Adjacency List (Unique and Insert)
  std::vector<std::vector<int>> adjList(numCoarsePoints);

  if (allEdges.empty()) {
    return adjList;
  }

  // Iterate through sorted edges and add unique ones
  // Handle first edge
  adjList[allEdges[0].u].push_back(allEdges[0].v);

  for (size_t i = 1; i < allEdges.size(); ++i) {
    // If different from previous, it's a new unique edge
    if (!(allEdges[i] == allEdges[i - 1])) {
      adjList[allEdges[i].u].push_back(allEdges[i].v);
    }
  }

  return adjList;
}

// Build vertex neighbors matrix for coarser level from precomputed adjacency
// list Returns a matrix where each row contains neighbor indices (-1 for no
// more neighbors) Self is always the first neighbor (at index 0)
Eigen::MatrixXi TetMultigridSolver::buildVertexNeighbours(
    const std::vector<std::vector<int>> &adjList, int numCoarsePoints) {

  // Find maximum number of neighbors
  size_t maxNeighNum = 0;
  for (const auto &neighbors : adjList) {
    if (neighbors.size() > maxNeighNum) {
      maxNeighNum = neighbors.size();
    }
  }

  // Store in homogeneous matrix structure
  // Allocate maxNeighNum + 1 columns: column 0 for self, rest for actual
  // neighbors
  Eigen::MatrixXi vertex_neighbours(numCoarsePoints, maxNeighNum + 1);
  vertex_neighbours.setConstant(-1);

#pragma omp parallel for
  for (int i = 0; i < numCoarsePoints; ++i) {
    vertex_neighbours(i, 0) = i; // Self as first neighbor
    int iCounter = 1;
    for (int node : adjList[i]) {
      if (node == i)
        continue; // Should effectively not happen if logic is correct (no
                  // self-loops in adjList usually)
      if (iCounter >= maxNeighNum + 1)
        break;
      vertex_neighbours(i, iCounter) = node;
      iCounter++;
    }
  }
  return vertex_neighbours;
}

// =============================================================================
// FUSED COARSE GRAPH BUILDER
// =============================================================================
// Combines three steps into a single call:
//   1. Dijkstra clustering (constructDijkstraWithCluster)
//   2. Coarse adjacency list (buildCoarseGraph)
//   3. Coarse neighbor matrix (buildVertexNeighbours)
//
// This is the standard fallback variant.
// coarseGraphBuilderExteriorFirst() implements the boundary-aware Ours path.
void TetMultigridSolver::coarseGraphBuilder(
    const Eigen::MatrixXd &vertices,
    const std::vector<int> &sampleIndices,
    const Eigen::MatrixXi &fineNeigh,
    int numCoarsePoints,
    Eigen::VectorXd &shortestDistanceToSample,
    std::vector<size_t> &nearestSource,
    std::vector<std::vector<int>> &coarseAdjList,
    Eigen::MatrixXi &coarseNeigh) {

  // Step 1: Dijkstra clustering
  constructDijkstraWithCluster(
      vertices, sampleIndices, fineNeigh,
      shortestDistanceToSample, nearestSource);

  // Step 2: Build coarse graph adjacency list
  coarseAdjList = buildCoarseGraph(
      fineNeigh, nearestSource, numCoarsePoints);

  // Step 3: Build coarse neighbor matrix
  coarseNeigh = buildVertexNeighbours(coarseAdjList, numCoarsePoints);
}

// =============================================================================
// TWO-PHASE COARSE GRAPH BUILDER (EXTERIOR FIRST)
// =============================================================================
// Builds coarse connectivity in two phases for the Ours hierarchy.
//
// Phase 1 (Exterior):
//   - Dijkstra clustering using only exterior-only connectivity
//     (allExteriorNeigh[0], which was computed by buildExteriorGraph)
//   - Build coarse exterior adjacency list from exterior-only fine edges
//   This treats the exterior as a standalone surface mesh.
//
// Phase 2 (Interior + Merge):
//   - Dijkstra clustering for interior vertices using full volume connectivity
//     (exterior vertices are locked to Phase 1 assignments)
//   - Build full coarse adjacency from volume connectivity
//   - Merge with Phase 1 exterior adjacency (union, no duplicates)
//
// Also stores the exterior coarse adjacency in allExteriorNeigh for
// potential reuse at deeper levels.
void TetMultigridSolver::coarseGraphBuilderExteriorFirst(
    const Eigen::MatrixXd &vertices,
    const std::vector<int> &sampleIndices,
    const Eigen::MatrixXi &fineVolumeNeigh,
    int numCoarsePoints,
    int numCoarseExterior,
    const Eigen::MatrixXi &fineExteriorNeigh,
    const std::vector<int> &fineExteriorIndices,
    Eigen::VectorXd &shortestDistanceToSample,
    std::vector<size_t> &nearestSource,
    std::vector<std::vector<int>> &coarseAdjList,
    Eigen::MatrixXi &coarseNeigh) {

  int numFineVerts = static_cast<int>(vertices.rows());
  int numFineExterior = static_cast<int>(fineExteriorIndices.size());

  if (verbose) {
    std::cout << "  [ExteriorFirst] numFineVerts=" << numFineVerts
              << ", numFineExterior=" << numFineExterior
              << ", numCoarsePoints=" << numCoarsePoints
              << ", numCoarseExterior=" << numCoarseExterior << "\n";
  }

  // =========================================================================
  // Step 1: Build a global-indexed exterior-only fine neighbor matrix
  // =========================================================================
  // allExteriorNeigh[0] uses LOCAL exterior indices (0..numFineExterior-1).
  // We need a neighbor matrix in GLOBAL vertex indices so that Dijkstra can
  // operate on the same index space as sampleIndices and nearestSource.
  //
  // Strategy: create a matrix of size (numFineVerts x cols), all -1.
  // For each exterior vertex, remap its local neighbors to global indices.

  Eigen::MatrixXi exteriorNeighGlobal(numFineVerts, fineExteriorNeigh.cols());
  exteriorNeighGlobal.setConstant(-1);

  for (int li = 0; li < numFineExterior; ++li) {
    int gi = fineExteriorIndices[li]; // global index of this exterior vertex
    for (int c = 0; c < fineExteriorNeigh.cols(); ++c) {
      int localNeigh = fineExteriorNeigh(li, c);
      if (localNeigh < 0) break;
      // Remap local -> global
      exteriorNeighGlobal(gi, c) = fineExteriorIndices[localNeigh];
    }
  }

  // =========================================================================
  // Step 2: Identify exterior vs interior samples
  // =========================================================================
  // After fastDiskSample, exterior samples are the first numCoarseExterior
  // entries in sampleIndices. Their coarse indices are 0..numCoarseExterior-1.

  std::vector<int> exteriorSources(sampleIndices.begin(),
                                    sampleIndices.begin() + numCoarseExterior);
  std::vector<int> interiorSources(sampleIndices.begin() + numCoarseExterior,
                                    sampleIndices.end());

  if (verbose) {
    std::cout << "  [ExteriorFirst] exteriorSources=" << exteriorSources.size()
              << ", interiorSources=" << interiorSources.size() << "\n";
  }

  // Build fast lookup: is this fine vertex exterior?
  std::vector<bool> isExterior(numFineVerts, false);
  for (int idx : fineExteriorIndices) {
    if (idx >= 0 && idx < numFineVerts) {
      isExterior[idx] = true;
    }
  }

  // =========================================================================
  // Step 3: Phase 1 Dijkstra -- Cluster exterior vertices using exterior-only
  //         connectivity (surface geodesics, no interior shortcuts)
  // =========================================================================
  // CRITICAL: Reset shortestDistanceToSample and nearestSource for ALL
  // exterior vertices before Phase 1. fastDiskSample has already populated
  // these using volume 1-ring/2-ring Euclidean distances, which may assign
  // exterior vertices to interior samples (through volume shortcuts).
  // Phase 1 must start from a clean slate so that exterior vertices are
  // clustered ONLY through surface-geodesic paths.
  for (int idx : fineExteriorIndices) {
    if (idx >= 0 && idx < numFineVerts) {
      shortestDistanceToSample(idx) = std::numeric_limits<double>::max();
      nearestSource[idx] = 0; // will be overwritten by Dijkstra
    }
  }

  // Initialize source distances for exterior samples only
  for (size_t i = 0; i < exteriorSources.size(); ++i) {
    int srcVertex = exteriorSources[i];
    shortestDistanceToSample(srcVertex) = 0.0;
    nearestSource[srcVertex] = i; // coarse index = position in sampleIndices
  }

  // Run Dijkstra on exterior-only graph
  constructDijkstraWithCluster(
      vertices, exteriorSources, exteriorNeighGlobal,
      shortestDistanceToSample, nearestSource);

  if (verbose) {
    std::cout << "  [ExteriorFirst] Phase 1 Dijkstra (exterior) done.\n";
  }

  // =========================================================================
  // Step 4: Phase 1 Coarse Graph -- Build exterior-only coarse adjacency
  // =========================================================================
  // Using the exterior-only fine neighbor matrix and the (partially filled)
  // nearestSource, build the coarse adjacency for exterior samples.
  // Interior rows are all -1 so they contribute no edges.

  std::vector<std::vector<int>> exteriorCoarseAdjList =
      buildCoarseGraph(exteriorNeighGlobal, nearestSource, numCoarsePoints);

  // Count exterior edges for debug
  int extEdgeCount = 0;
  for (int i = 0; i < numCoarseExterior; ++i) {
    extEdgeCount += static_cast<int>(exteriorCoarseAdjList[i].size());
  }
  if (verbose) {
    std::cout << "  [ExteriorFirst] Phase 1 coarse graph: "
              << extEdgeCount << " directed exterior edges.\n";
  }

  // =========================================================================
  // Step 5: Phase 2 Dijkstra -- Cluster interior vertices using full volume
  //         connectivity. Exterior vertices are LOCKED.
  // =========================================================================
  // Interior samples get dist=0. Exterior vertices keep their Phase 1
  // distance and act as boundary sources.

  std::priority_queue<VertexPair, std::vector<VertexPair>,
                      std::greater<VertexPair>> DistanceQueue;

  // Add interior samples with dist=0
  for (size_t i = 0; i < interiorSources.size(); ++i) {
    int srcVertex = interiorSources[i];
    size_t coarseIdx = numCoarseExterior + i; // position in sampleIndices
    shortestDistanceToSample(srcVertex) = 0.0;
    nearestSource[srcVertex] = coarseIdx;
    DistanceQueue.push({srcVertex, 0.0});
  }

  // Add assigned exterior vertices as frontier sources for interior expansion
  for (int vExt : fineExteriorIndices) {
    if (vExt >= 0 && vExt < numFineVerts) {
      double d = shortestDistanceToSample(vExt);
      if (d < std::numeric_limits<double>::max()) {
        DistanceQueue.push({vExt, d});
      }
    }
  }

  // Dijkstra expansion: propagate through volume, but do NOT update exterior
  while (!DistanceQueue.empty()) {
    VertexPair vp = DistanceQueue.top();
    DistanceQueue.pop();

    if (vp.distance > shortestDistanceToSample(vp.vId)) continue;

    size_t curSourceIndex = nearestSource[vp.vId];
    Eigen::RowVector3d p1 = vertices.row(vp.vId);

    for (int c = 0; c < fineVolumeNeigh.cols(); ++c) {
      int vNeigh = fineVolumeNeigh(vp.vId, c);
      if (vNeigh < 0) break;

      // Do NOT update exterior vertices -- they are locked from Phase 1
      if (isExterior[vNeigh]) continue;

      Eigen::RowVector3d p2 = vertices.row(vNeigh);
      double d = (p2 - p1).norm();
      double newDist = vp.distance + d;

      if (newDist < shortestDistanceToSample(vNeigh)) {
        shortestDistanceToSample(vNeigh) = newDist;
        nearestSource[vNeigh] = curSourceIndex;
        DistanceQueue.push({vNeigh, newDist});
      }
    }
  }

  if (verbose) {
    std::cout << "  [ExteriorFirst] Phase 2 Dijkstra (interior) done.\n";
  }

  // =========================================================================
  // Step 6: Phase 2 Coarse Graph -- Build volume coarse adjacency
  //         EXCLUDING exterior<->exterior edges, then merge with Phase 1
  // =========================================================================
  // The key insight: if we built the coarse graph from the FULL volume
  // neighbor matrix, it would re-discover all exterior<->exterior edges
  // (since volume connectivity is a superset of exterior connectivity).
  // That would make Phase 1's exterior graph completely redundant.
  //
  // Instead, we build Phase 2's coarse graph from a FILTERED volume graph
  // that removes exterior<->exterior edges. This way:
  //   - Exterior<->Exterior connections come ONLY from Phase 1 (surface topology)
  //   - Interior<->Interior connections come from Phase 2 (volume topology)
  //   - Interior<->Exterior connections come from Phase 2 (volume topology)
  // The merge then produces a graph where exterior connectivity is faithfully
  // determined by surface geodesics, not volume shortcuts.

  // Build filtered volume neighbor matrix (remove exterior<->exterior edges)
  Eigen::MatrixXi volumeNeighNoExtExt = fineVolumeNeigh;
  for (int i = 0; i < numFineVerts; ++i) {
    if (!isExterior[i]) continue; // only filter rows of exterior vertices
    for (int c = 0; c < volumeNeighNoExtExt.cols(); ++c) {
      int n = volumeNeighNoExtExt(i, c);
      if (n < 0) break;
      if (isExterior[n]) {
        volumeNeighNoExtExt(i, c) = -1; // remove exterior<->exterior edge
      }
    }
    // Compact: shift non-(-1) entries to the left to maintain the
    // "contiguous then -1 sentinel" invariant that buildCoarseGraph expects
    int writePos = 0;
    for (int c = 0; c < volumeNeighNoExtExt.cols(); ++c) {
      if (volumeNeighNoExtExt(i, c) >= 0) {
        volumeNeighNoExtExt(i, writePos++) = volumeNeighNoExtExt(i, c);
      }
    }
    for (int c = writePos; c < volumeNeighNoExtExt.cols(); ++c) {
      volumeNeighNoExtExt(i, c) = -1;
    }
  }

  std::vector<std::vector<int>> volumeCoarseAdjList =
      buildCoarseGraph(volumeNeighNoExtExt, nearestSource, numCoarsePoints);

  // Count edges from each source for diagnostics
  int volEdgeCount = 0;
  for (int i = 0; i < numCoarsePoints; ++i) {
    volEdgeCount += static_cast<int>(volumeCoarseAdjList[i].size());
  }
  if (verbose) {
    std::cout << "  [ExteriorFirst] Phase 2 coarse graph (no ext-ext): "
              << volEdgeCount << " directed edges.\n";
  }

  // Count how many Phase 1 exterior edges are NOT in the volume graph
  int uniqueExtEdges = 0;
  for (int i = 0; i < numCoarseExterior; ++i) {
    for (int e : exteriorCoarseAdjList[i]) {
      if (!std::binary_search(volumeCoarseAdjList[i].begin(),
                              volumeCoarseAdjList[i].end(), e)) {
        uniqueExtEdges++;
      }
    }
  }
  if (verbose) {
    std::cout << "  [ExteriorFirst] Exterior edges unique to Phase 1: "
              << uniqueExtEdges
              << " (these would be lost without exterior-first).\n";
  }

  // Merge: union of exterior and volume adjacency lists
  // For each coarse vertex, take the union of both adj lists (sorted, unique)
  coarseAdjList.resize(numCoarsePoints);
  for (int i = 0; i < numCoarsePoints; ++i) {
    // Start with exterior adjacency (will be empty for interior-only vertices)
    std::vector<int> merged;
    merged.reserve(exteriorCoarseAdjList[i].size() +
                   volumeCoarseAdjList[i].size());

    // Both lists are already sorted (buildCoarseGraph produces sorted output)
    std::merge(exteriorCoarseAdjList[i].begin(),
               exteriorCoarseAdjList[i].end(),
               volumeCoarseAdjList[i].begin(),
               volumeCoarseAdjList[i].end(),
               std::back_inserter(merged));

    // Remove duplicates
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

    coarseAdjList[i] = std::move(merged);
  }

  int totalEdges = 0;
  for (int i = 0; i < numCoarsePoints; ++i) {
    totalEdges += static_cast<int>(coarseAdjList[i].size());
  }
  if (verbose) {
    std::cout << "  [ExteriorFirst] Merged coarse graph: "
              << totalEdges << " directed edges total.\n";
  }

  // =========================================================================
  // Step 7: Build coarse neighbor matrix from merged adjacency
  // =========================================================================
  coarseNeigh = buildVertexNeighbours(coarseAdjList, numCoarsePoints);

  // =========================================================================
  // Step 8: Store exterior coarse connectivity for future levels
  // =========================================================================
  // Convert the exterior-only portion of the coarse adj list into an
  // ExteriorNeigh matrix using LOCAL exterior indices (0..numCoarseExterior-1).
  // This mirrors the format of allExteriorNeigh[0].

  int maxExtDegree = 0;
  for (int i = 0; i < numCoarseExterior; ++i) {
    maxExtDegree = std::max(maxExtDegree,
                            static_cast<int>(exteriorCoarseAdjList[i].size()));
  }

  Eigen::MatrixXi coarseExteriorNeigh(numCoarseExterior, maxExtDegree + 1);
  coarseExteriorNeigh.setConstant(-1);
  for (int i = 0; i < numCoarseExterior; ++i) {
    coarseExteriorNeigh(i, 0) = i; // self
    int col = 1;
    for (int neighIdx : exteriorCoarseAdjList[i]) {
      // Only keep edges to other exterior coarse vertices
      if (neighIdx < numCoarseExterior) {
        coarseExteriorNeigh(i, col++) = neighIdx;
      }
    }
  }

  // Push to allExteriorNeigh for this coarser level
  // Note: allExteriorNeigh[0] is level 0. This push adds level 1.
  allExteriorNeigh.push_back(coarseExteriorNeigh);

  // Also push allExteriorVertices so that the visualization has matching
  // vertex positions for the coarse exterior graph.
  // The coarse exterior vertices are sampleIndices[0..numCoarseExterior-1],
  // and their positions are the coarse vertex positions.
  Eigen::MatrixXd coarseExteriorVerts(numCoarseExterior, 3);
  for (int i = 0; i < numCoarseExterior; ++i) {
    coarseExteriorVerts.row(i) = vertices.row(sampleIndices[i]);
  }
  allExteriorVertices.push_back(coarseExteriorVerts);

  if (verbose) {
    std::cout << "  [ExteriorFirst] Stored coarse exterior neigh ("
              << numCoarseExterior << " verts, maxDegree="
              << maxExtDegree << ").\n";
  }
}

// Construct geometric prolongation matrix
// Extracts the parallel matrix construction logic from the main loop
Eigen::SparseMatrix<double>
TetMultigridSolver::constructGeometricProlongationMatrix(
    const Eigen::MatrixXd &fineVertices, const Eigen::MatrixXd &coarseVertices,
    const std::vector<int> &coarseIndices,
    const std::vector<size_t> &nearestCoarseIdx,
    const std::vector<std::vector<int>> &coarseNeighborList,
    const std::vector<std::vector<int>> &tets,
    const std::vector<std::vector<int>> &connectedTets,
    const std::vector<std::vector<int>> &tris,
    const std::vector<std::vector<int>> &connectedTris) {
  // Use vector<char> for fast O(1) lookup
  std::vector<char> isSample(fineVertices.rows(), 0);
  for (int idx : coarseIndices) {
    if (idx >= 0 && idx < isSample.size()) {
      isSample[idx] = 1;
    }
  }

  int numThreads = omp_get_max_threads();
  std::vector<std::vector<Eigen::Triplet<double>>> threadTriplets(numThreads);
  std::vector<std::vector<int>> threadCaseCounts(numThreads,
                                                 std::vector<int>(7, 0));

  // Pre-allocate thread-local triplet vectors
  for (int t = 0; t < numThreads; ++t) {
    threadTriplets[t].reserve((fineVertices.rows() * 4) / numThreads + 100);
  }

// For each fine point, find geometric interpolation
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    std::vector<Eigen::Triplet<double>> &localTriplets = threadTriplets[tid];
    std::vector<int> &localCaseCounts = threadCaseCounts[tid];

#pragma omp for
    for (int fineIdx = 0; fineIdx < fineVertices.rows(); ++fineIdx) {
      Eigen::RowVector3d finePoint = fineVertices.row(fineIdx);
      int coarseIdx = nearestCoarseIdx[fineIdx];
      Eigen::RowVector3d coarsePoint = coarseVertices.row(coarseIdx);

      // Check if this fine point is a sample point (O(1) lookup)
      bool isSamplePoint = isSample[fineIdx];

      int case_used = -1; // Track which case was used

      if (isSamplePoint) {
        // ========== CASE 0: Sample point (direct injection) ==========
        case_used = 0;
        localTriplets.push_back(
            Eigen::Triplet<double>(fineIdx, coarseIdx, 1.0));
      } else if (coarseNeighborList[coarseIdx].empty()) {
        // ========== CASE 1: No neighbors ==========
        case_used = 1;
        localTriplets.push_back(
            Eigen::Triplet<double>(fineIdx, coarseIdx, 1.0));
      } else if (coarseNeighborList[coarseIdx].size() == 1) {
        // ========== CASE 2: One neighbor (edge interpolation) ==========
        case_used = 2;
        int neighIdx = *coarseNeighborList[coarseIdx].begin();
        Eigen::RowVector3d neighPoint = coarseVertices.row(neighIdx);
        Eigen::RowVector3d edgeVec = neighPoint - coarsePoint;
        double edgeLengthSq = edgeVec.squaredNorm();
        double w1 =
            (edgeLengthSq > 1e-20)
                ? ((finePoint - coarsePoint).dot(edgeVec) / edgeLengthSq)
                : 0.0;
        w1 = std::min(std::max(w1, 0.0), 1.0);
        double w0 = 1.0 - w1;

        localTriplets.push_back(Eigen::Triplet<double>(fineIdx, coarseIdx, w0));
        localTriplets.push_back(Eigen::Triplet<double>(fineIdx, neighIdx, w1));
      } else {
        // Multiple neighbors: try geometric containment
        Eigen::RowVector4d bary4;
        int foundTetIdx = -1;
        double foundTetMetric = std::numeric_limits<double>::max() * 0.5;
        Eigen::RowVector4d foundTetBary;

        // ========== CASE 3: Inside tetrahedron ==========
        for (int tetIdx : connectedTets[coarseIdx]) {
          std::vector<int> tet = tets[tetIdx];
          // Rotate so coarseIdx is first
          while (tet[0] != coarseIdx)
            std::rotate(tet.begin(), tet.begin() + 1, tet.end());

          double metric = inTet(finePoint, tet, coarseVertices, bary4);
          if (metric > -1e-5 && metric < foundTetMetric) {
            foundTetIdx = tetIdx;
            foundTetMetric = metric;
            foundTetBary = bary4;
          }
        }

        if (foundTetIdx > -1) {
          // Found tet containing point - use barycentric coordinates
          case_used = 3;
          std::vector<int> foundTet = tets[foundTetIdx];
          while (foundTet[0] != coarseIdx)
            std::rotate(foundTet.begin(), foundTet.begin() + 1, foundTet.end());

          for (int i = 0; i < 4; ++i) {
            if (foundTetBary(i) > 1e-12) // Skip near-zero weights
              localTriplets.push_back(Eigen::Triplet<double>(
                  fineIdx, foundTet[i], foundTetBary(i)));
          }
        } else {
          // ========== CASE 4: Project onto triangle ==========
          case_used = 4;
          Eigen::RowVector3d bary3;
          std::vector<int> foundTri = {-1, -1, -1};
          double foundTriMetric = std::numeric_limits<double>::max() * 0.5;
          Eigen::RowVector3d foundTriBary;
          std::map<int, double> insideEdge;

          for (int triIdx : connectedTris[coarseIdx]) {
            std::vector<int> tri = tris[triIdx];
            while (tri[0] != coarseIdx)
              std::rotate(tri.begin(), tri.begin() + 1, tri.end());

            double metric =
                inTri(finePoint, tri, coarseVertices, bary3, insideEdge);
            if (metric > -1e-5 && metric < foundTriMetric) {
              foundTri = tri;
              foundTriMetric = metric;
              foundTriBary = bary3;
            }
          }

          if (foundTri[0] > -1) {
            // Found triangle containing projection
            case_used = 4;
            for (int i = 0; i < 3; ++i) {
              if (foundTriBary(i) > 1e-12)
                localTriplets.push_back(Eigen::Triplet<double>(
                    fineIdx, foundTri[i], foundTriBary(i)));
            }
          } else {
            // ========== CASE 5: Project onto edge ==========
            int foundNeighIdx = -1;
            double foundNeighMetric = std::numeric_limits<double>::max() * 0.5;

            for (const auto &kv : insideEdge) {
              if (kv.second >= 0.0 && kv.second < foundNeighMetric) {
                foundNeighMetric = kv.second;
                foundNeighIdx = kv.first;
              }
            }

            if (foundNeighIdx > -1) {
              // Found edge containing projection
              case_used = 5;
              Eigen::RowVector3d neighPoint = coarseVertices.row(foundNeighIdx);
              Eigen::RowVector3d edgeVec = neighPoint - coarsePoint;
              double edgeLengthSq = edgeVec.squaredNorm();
              double w1 =
                  (edgeLengthSq > 1e-20)
                      ? ((finePoint - coarsePoint).dot(edgeVec) / edgeLengthSq)
                      : 0.0;
              w1 = std::min(std::max(w1, 0.0), 1.0);
              double w0 = 1.0 - w1;

              localTriplets.push_back(
                  Eigen::Triplet<double>(fineIdx, coarseIdx, w0));
              localTriplets.push_back(
                  Eigen::Triplet<double>(fineIdx, foundNeighIdx, w1));
            } else {
              // ========== CASE 6: Fallback to inverse distance (closest 3-4
              // points) ==========
              case_used = 6;
              std::vector<int> prolongFrom = {coarseIdx};
              std::vector<VertexPair> neighborDistances;

              for (int neighIdx : coarseNeighborList[coarseIdx]) {
                if (neighIdx == coarseIdx)
                  continue;
                double dist = (finePoint - coarseVertices.row(neighIdx)).norm();
                neighborDistances.push_back({neighIdx, dist});
              }

              std::sort(neighborDistances.begin(), neighborDistances.end(),
                        [](const VertexPair &a, const VertexPair &b) {
                          return a.distance < b.distance;
                        });

              int maxNeighbors = std::min(3, (int)neighborDistances.size());
              for (int i = 0; i < maxNeighbors; ++i)
                prolongFrom.push_back(neighborDistances[i].vId);

              std::vector<double> weights = inverseDistanceWeights(
                  coarseVertices, finePoint, prolongFrom);

              for (size_t i = 0; i < prolongFrom.size(); ++i) {
                if (weights[i] > 1e-12)
                  localTriplets.push_back(Eigen::Triplet<double>(
                      fineIdx, prolongFrom[i], weights[i]));
              }
            }
          }
        }
      }

      // Record which case was used (thread-local increment)
      if (case_used >= 0 && case_used < 7) {
        localCaseCounts[case_used]++;
      }
    }
  } // End of #pragma omp parallel

  // Merge thread-local triplets into a single vector
  size_t totalTriplets = 0;
  for (int t = 0; t < numThreads; ++t) {
    totalTriplets += threadTriplets[t].size();
  }
  std::vector<Eigen::Triplet<double>> PTriplets;
  PTriplets.reserve(totalTriplets);
  for (int t = 0; t < numThreads; ++t) {
    PTriplets.insert(PTriplets.end(), threadTriplets[t].begin(),
                     threadTriplets[t].end());
  }

  // Merge thread-local case counts
  std::vector<int> case_counts(7, 0);
  for (int t = 0; t < numThreads; ++t) {
    for (int c = 0; c < 7; ++c) {
      case_counts[c] += threadCaseCounts[t][c];
    }
  }

  // Build sparse matrix
  Eigen::SparseMatrix<double> P(fineVertices.rows(), coarseVertices.rows());
  P.setFromTriplets(PTriplets.begin(), PTriplets.end());

    if (verbose) {
    std::cout << "  Prolongation matrix: " << P.rows() << " x " << P.cols()
          << ", nnz = " << P.nonZeros() << "\n";

    // Report interpolation case statistics
    int total_points = fineVertices.rows();
    std::cout << "  Interpolation case breakdown:\n";
    std::cout << "    Case 0 (Sample/Injection):  " << case_counts[0] << " ("
          << (100.0 * case_counts[0] / total_points) << "%)\n";
    std::cout << "    Case 1 (No neighbors):      " << case_counts[1] << " ("
          << (100.0 * case_counts[1] / total_points) << "%)\n";
    std::cout << "    Case 2 (Edge - 1 neighbor): " << case_counts[2] << " ("
          << (100.0 * case_counts[2] / total_points) << "%)\n";
    std::cout << "    Case 3 (Tet - barycentric): " << case_counts[3] << " ("
          << (100.0 * case_counts[3] / total_points) << "%)\n";
    std::cout << "    Case 4 (Triangle):          " << case_counts[4] << " ("
          << (100.0 * case_counts[4] / total_points) << "%)\n";
    std::cout << "    Case 5 (Edge projection):   " << case_counts[5] << " ("
          << (100.0 * case_counts[5] / total_points) << "%)\n";
    std::cout << "    Case 6 (Inverse distance):  " << case_counts[6] << " ("
          << (100.0 * case_counts[6] / total_points) << "%)\n";
    std::cout << "  Linear-exact cases (0-5): "
          << (case_counts[0] + case_counts[1] + case_counts[2] +
            case_counts[3] + case_counts[4] + case_counts[5])
          << " ("
          << (100.0 *
            (case_counts[0] + case_counts[1] + case_counts[2] +
             case_counts[3] + case_counts[4] + case_counts[5]) /
            total_points)
          << "%)\n";
    }

  return P;
}

// Build simplicial complex from coarse point cloud using neighbor connectivity
// Creates triangles and tetrahedra from neighbor graph cliques (Voronoi-like
// condition) Note: Result is a simplicial complex, not a proper tetrahedral
// mesh - can have isolated vertices, dangling edges, floating triangles, and
// proper tetrahedra Returns: [triangles, connectedTris (per vertex), tets,
// connectedTets (per vertex)] Uses efficient sorted intersection for clique
// finding
std::array<std::vector<std::vector<int>>, 4>
TetMultigridSolver::buildSimplicialComplex(
    const Eigen::MatrixXd &verts,
    const std::vector<std::vector<int>> &adjList) {
  int n = verts.rows();
  std::vector<std::vector<int>> tris, tets;
  tris.reserve(n * 10); // Estimate
  tets.reserve(n * 10);
  std::vector<std::vector<int>> connectedTris(n), connectedTets(n);

  // Use thread-local storage to avoid critical sections
  int max_threads = omp_get_max_threads();
  std::vector<std::vector<std::vector<int>>> thread_tris(max_threads);
  std::vector<std::vector<std::vector<int>>> thread_tets(max_threads);

  // Pre-allocate
  for (int t = 0; t < max_threads; ++t) {
    thread_tris[t].reserve(n / max_threads * 10);
    thread_tets[t].reserve(n / max_threads * 10);
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    int thread_id = omp_get_thread_num();
    const auto &adj_i = adjList[i];

    // For each neighbor j of i
    for (int j : adj_i) {
      if (j <= i)
        continue; // Symmetry breaking: i < j

      const auto &adj_j = adjList[j];

      // Find common neighbors K = intersect(adj_i, adj_j) where k > j
      // Since adj lists are sorted, we can do this efficiently
      std::vector<int> common_K;
      // Reserve simplified estimate
      common_K.reserve(std::min(adj_i.size(), adj_j.size()));

      // Sorted intersection (std::set_intersection logic but custom to filter >
      // j)
      auto it1 = adj_i.begin();
      auto it2 = adj_j.begin();
      while (it1 != adj_i.end() && it2 != adj_j.end()) {
        if (*it1 < *it2) {
          ++it1;
        } else if (*it2 < *it1) {
          ++it2;
        } else {
          // Found common neighbor k
          int k = *it1;
          if (k > j) {
            common_K.push_back(k);
          }
          ++it1;
          ++it2;
        }
      }

      // Check for triangles (i, j, k)
      for (int k : common_K) {
        // Found valid triangle - store in thread-local vector
        thread_tris[thread_id].push_back({i, j, k});

        const auto &adj_k = adjList[k];

        // Look for tetrahedra: i, j, k, l where all four are mutual neighbors
        // We already know k is neighbor of i and j.
        // We need l such that:
        // 1. l is neighbor of i (in adj_i)
        // 2. l is neighbor of j (in adj_j)
        // 3. l is neighbor of k (in adj_k)
        // AND l > k (symmetry)

        // This is equivalent to l being in common_K AND l being in adj_k
        // So we compute L = intersect(common_K, adj_k) where l > k

        auto it3 = common_K.begin(); // already sorted
        auto it4 = adj_k.begin();    // already sorted

        while (it3 != common_K.end() && it4 != adj_k.end()) {
          if (*it3 < *it4) {
            ++it3;
          } else if (*it4 < *it3) {
            ++it4;
          } else {
            int l = *it3;
            if (l > k) {
              // Found valid tetrahedron - store in thread-local vector
              thread_tets[thread_id].push_back({i, j, k, l});
            }
            ++it3;
            ++it4;
          }
        }
      }
    }
  }

  // Merge thread-local results
  for (int t = 0; t < max_threads; ++t) {
    tris.insert(tris.end(), thread_tris[t].begin(), thread_tris[t].end());
    tets.insert(tets.end(), thread_tets[t].begin(), thread_tets[t].end());
  }

  // Build connectivity lists
  for (int triIdx = 0; triIdx < tris.size(); ++triIdx) {
    const auto &tri = tris[triIdx];
    connectedTris[tri[0]].push_back(triIdx);
    connectedTris[tri[1]].push_back(triIdx);
    connectedTris[tri[2]].push_back(triIdx);
  }

  for (int tetIdx = 0; tetIdx < tets.size(); ++tetIdx) {
    const auto &tet = tets[tetIdx];
    connectedTets[tet[0]].push_back(tetIdx);
    connectedTets[tet[1]].push_back(tetIdx);
    connectedTets[tet[2]].push_back(tetIdx);
    connectedTets[tet[3]].push_back(tetIdx);
  }

  return {tris, connectedTris, tets, connectedTets};
}

// Check if point is inside tetrahedron using barycentric coordinates
// Returns minimum barycentric coordinate if inside (>= 0), -1.0 if outside
double TetMultigridSolver::inTet(const Eigen::RowVector3d &p,
                                          const std::vector<int> &tet,
                                          const Eigen::MatrixXd &verts,
                                          Eigen::RowVector4d &bary) {
  std::vector<Eigen::RowVector3d> vs = {verts.row(tet[0]), verts.row(tet[1]),
                                        verts.row(tet[2]), verts.row(tet[3])};

  // Compute first three barycentric coordinates
  for (int i = 0; i < 3; ++i) {
    Eigen::RowVector3d v0 = vs[0], v1 = vs[1], v2 = vs[2], v3 = vs[3];
    Eigen::RowVector3d e12 = v2 - v1, e13 = v3 - v1;
    Eigen::RowVector3d triNormal = e12.cross(e13);

    if (triNormal.norm() < 1e-20)
      return -1.0;
    triNormal.normalize();

    double distP = (p - v1).dot(triNormal);
    double distV0 = (v0 - v1).dot(triNormal);

    bary(i) = (std::abs(distV0) > 1e-20) ? (distP / distV0) : -1.0;
    std::rotate(vs.begin(), vs.begin() + 1, vs.end());
  }

  // Fourth coordinate from constraint
  bary(3) = 1.0 - bary(0) - bary(1) - bary(2);

  // Check if inside
  if (bary(0) < 0.0 || bary(1) < 0.0 || bary(2) < 0.0 || bary(3) < 0.0)
    return -1.0;

  return std::min({bary(0), bary(1), bary(2), bary(3)});
}

// Project point onto triangle and compute barycentric coordinates
// Also checks edge projections and stores in insideEdge map
// Returns distance to triangle if projection is inside, -1.0 otherwise
double TetMultigridSolver::inTri(const Eigen::RowVector3d &p,
                                          const std::vector<int> &triIndices,
                                          const Eigen::MatrixXd &vertices,
                                          Eigen::RowVector3d &barycoords,
                                          std::map<int, double> &insideEdge) {
  Eigen::RowVector3d a = vertices.row(triIndices[0]);
  Eigen::RowVector3d b = vertices.row(triIndices[1]);
  Eigen::RowVector3d c = vertices.row(triIndices[2]);

  Eigen::RowVector3d v0 = b - a;
  Eigen::RowVector3d v1 = c - a;
  Eigen::RowVector3d v2 = p - a;

  double d00 = v0.dot(v0);
  double d01 = v0.dot(v1);
  double d11 = v1.dot(v1);
  double d20 = v2.dot(v0);
  double d21 = v2.dot(v1);

  double denom = d00 * d11 - d01 * d01;

  // If degenerate triangle, handle gracefully
  if (std::abs(denom) < 1e-12)
    return std::numeric_limits<double>::max();

  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;

  barycoords << u, v, w;

  // Check if point is inside triangle (with small epsilon for tolerance)
  const double eps = -1e-5;
  if (u >= eps && v >= eps && w >= eps) {
    // Inside triangle - compute normal distance
    Eigen::RowVector3d normal = v0.cross(v1).normalized();
    return std::abs(normal.dot(v2));
  }

  // If outside, compute distance to edges for fallback strategies

  // Edge 1: a-b (indices 0-1)
  {
    Eigen::RowVector3d e01 = b - a;
    Eigen::RowVector3d e0p = p - a;
    double sqNorm = e01.squaredNorm();
    double alpha = (sqNorm > 1e-20) ? e0p.dot(e01) / sqNorm : 0.0;
    alpha = std::clamp(alpha, 0.0, 1.0);
    double distToEdge = (e0p - alpha * e01).norm();
    insideEdge[triIndices[2]] = distToEdge; // Using opposite vertex as key
  }

  // Edge 2: b-c (indices 1-2)
  {
    Eigen::RowVector3d e12 = c - b;
    Eigen::RowVector3d e1p = p - b;
    double sqNorm = e12.squaredNorm();
    double alpha = (sqNorm > 1e-20) ? e1p.dot(e12) / sqNorm : 0.0;
    alpha = std::clamp(alpha, 0.0, 1.0);
    double distToEdge = (e1p - alpha * e12).norm();
    insideEdge[triIndices[0]] = distToEdge;
  }

  // Edge 3: c-a (indices 2-0)
  {
    Eigen::RowVector3d e20 = a - c;
    Eigen::RowVector3d e2p = p - c;
    double sqNorm = e20.squaredNorm();
    double alpha = (sqNorm > 1e-20) ? e2p.dot(e20) / sqNorm : 0.0;
    alpha = std::clamp(alpha, 0.0, 1.0);
    double distToEdge = (e2p - alpha * e20).norm();
    insideEdge[triIndices[1]] = distToEdge;
  }

  return std::numeric_limits<double>::max();
}

// Compute inverse distance weights (normalized to sum to 1)
std::vector<double> TetMultigridSolver::inverseDistanceWeights(
    const Eigen::MatrixXd &verts, const Eigen::RowVector3d &p,
    const std::vector<int> &indices) {
  std::vector<double> weights(indices.size());
  double sumWeight = 0.0;

  for (int i = 0; i < indices.size(); ++i) {
    double dist = (p - verts.row(indices[i])).norm();

    // Handle exact coincidence
    if (dist < 1e-12) {
      std::fill(weights.begin(), weights.end(), 0.0);
      weights[i] = 1.0;
      return weights;
    }

    weights[i] = 1.0 / dist;
    sumWeight += weights[i];
  }

  // Normalize
  for (auto &w : weights)
    w /= sumWeight;
  return weights;
}

// Test prolongation matrix properties
// Performs essential validation tests on a prolongation matrix
// Tests: 1) Dimensions, 2) Partition of unity, 3) Non-negativity, 4) Constant
// preservation Returns true if all tests pass, false otherwise
bool TetMultigridSolver::testProlongationMatrix(
    const Eigen::SparseMatrix<double> &P, const Eigen::MatrixXd &vertsFine,
    const Eigen::MatrixXd &vertsCoarse, int levelIndex) {
  bool allTestsPassed = true;
  const double TOL_STRICT = 1e-10;    // Tolerance for partition of unity
  const double TOL_NEGATIVE = -1e-10; // Tolerance for non-negativity (allow
                                      // tiny negatives due to floating point)

  std::cout << "\n  ========== PROLONGATION MATRIX TESTS (Level " << levelIndex
            << ") ==========\n";

  // ===== TEST 1: Dimension Check =====
  std::cout << "  [TEST 1] Dimensions: P = " << P.rows() << " x " << P.cols()
            << "\n";
  if (P.rows() != vertsFine.rows() || P.cols() != vertsCoarse.rows()) {
    std::cout << "  *** ERROR *** Dimension mismatch!\n";
    std::cout << "    Expected: " << vertsFine.rows() << " x "
              << vertsCoarse.rows() << "\n";
    std::cout << "    Got:      " << P.rows() << " x " << P.cols() << "\n";
    allTestsPassed = false;
  } else {
    std::cout << "  OK Dimensions correct\n";
  }

  // ===== TEST 2: Partition of Unity (Row Sums) =====
  std::cout
      << "  [TEST 2] Partition of Unity (each row should sum to 1.0)...\n";
  double maxRowSumError = 0.0;
  int worstRow = -1;
  int violationCount = 0;

  // Efficient: accumulate row sums in one pass through the sparse matrix
  std::vector<double> rowSums(P.rows(), 0.0);
  for (int k = 0; k < P.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(P, k); it; ++it) {
      rowSums[it.row()] += it.value();
    }
  }

  // Check errors
  for (int i = 0; i < P.rows(); ++i) {
    double error = std::abs(rowSums[i] - 1.0);
    if (error > maxRowSumError) {
      maxRowSumError = error;
      worstRow = i;
    }
    if (error > TOL_STRICT) {
      violationCount++;
    }
  }

  std::cout << "    Max row sum error: " << maxRowSumError << " (at row "
            << worstRow << ")\n";
  std::cout << "    Rows violating tolerance: " << violationCount << " / "
            << P.rows() << "\n";
  if (maxRowSumError > TOL_STRICT) {
    std::cout << "  *** WARNING *** Partition of unity violated! Max error: "
              << maxRowSumError << "\n";
    allTestsPassed = false;
  } else {
    std::cout << "  OK Partition of unity satisfied\n";
  }

  // ===== TEST 3: Non-Negativity Check =====
  std::cout << "  [TEST 3] Non-Negativity (all entries should be >= 0)...\n";
  int negativeCount = 0;
  double minValue = 0.0;
  int minRow = -1, minCol = -1;

  for (int k = 0; k < P.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(P, k); it; ++it) {
      if (it.value() < TOL_NEGATIVE) {
        negativeCount++;
        if (it.value() < minValue) {
          minValue = it.value();
          minRow = it.row();
          minCol = it.col();
        }
      }
    }
  }

  if (negativeCount > 0) {
    std::cout << "  *** WARNING *** Found " << negativeCount
              << " negative entries!\n";
    std::cout << "    Most negative: " << minValue << " at (" << minRow << ", "
              << minCol << ")\n";
    allTestsPassed = false;
  } else {
    std::cout << "  OK All entries non-negative\n";
  }

  // ===== TEST 4: Constant Preservation =====
  std::cout << "  [TEST 4] Constant Preservation (P * ones = ones)...\n";
  Eigen::VectorXd onesCoarse = Eigen::VectorXd::Ones(P.cols());
  Eigen::VectorXd onesFine = P * onesCoarse;
  double maxConstantError = (onesFine.array() - 1.0).abs().maxCoeff();

  std::cout << "    Max deviation from 1.0: " << maxConstantError << "\n";

  if (maxConstantError > TOL_STRICT) {
    std::cout << "  *** WARNING *** Constant preservation failed! Error: "
              << maxConstantError << "\n";
    allTestsPassed = false;
  } else {
    std::cout << "  OK Constant preservation satisfied\n";
  }

  // ===== SUMMARY =====
  std::cout << "  ========== ";
  if (allTestsPassed) {
    std::cout << "ALL TESTS PASSED";
  } else {
    std::cout << "*** TESTS FAILED ***";
  }
  std::cout << " ==========\n\n";

  return allTestsPassed;
}

void TetMultigridSolver::exportHierarchyTimingJSON(
    const std::string &filename) {
  if (filename.empty()) {
    return;
  }

  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Error: Could not open " << filename << " for writing JSON."
              << std::endl;
    return;
  }

  auto write_double_array = [&out](const char *name,
                                   const std::vector<double> &values,
                                   bool trailing_comma) {
    out << "  \"" << name << "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << values[i];
    }
    out << "]";
    if (trailing_comma) {
      out << ",";
    }
    out << "\n";
  };

  out << "{\n";
  out << "  \"num_vertices_start\": " << benchmark.num_vertices_start << ",\n";
  out << "  \"init_ms\": " << benchmark.init_ms << ",\n";
  out << "  \"init_exterior_extraction_ms\": "
      << benchmark.init_exterior_extraction_ms << ",\n";
  write_double_array("level_sampling_ms", benchmark.level_sampling_ms, true);
  write_double_array("level_sampling_priority_ms",
                     benchmark.level_sampling_priority_ms, true);
  write_double_array("level_exterior_detect_ms",
                     benchmark.level_exterior_detect_ms, true);
  write_double_array("level_sort_feature_ms",
                     benchmark.level_sort_feature_ms, true);
  write_double_array("level_coarse_graph_ms",
                     benchmark.level_coarse_graph_ms, true);
  write_double_array("level_smooth_interior_ms",
                     benchmark.level_smooth_interior_ms, true);
  write_double_array("level_simplicial_complex_ms",
                     benchmark.level_simplicial_complex_ms, true);
  write_double_array("level_interpolation_ms",
                     benchmark.level_interpolation_ms, true);
  write_double_array("level_total_ms", benchmark.level_total_ms, true);
  out << "  \"final_exterior_graph_ms\": " << benchmark.final_exterior_graph_ms
      << ",\n";
  out << "  \"total_hierarchy_ms\": " << benchmark.total_hierarchy_ms << "\n";
  out << "}\n";
}

// =================================================================================================
// MARK: Core Hierarchy Construction
// =================================================================================================

/*
 * Simple timer class for benchmarking
 */
class Timer {
public:
  Timer() : start(std::chrono::high_resolution_clock::now()) {}

  void reset() { start = std::chrono::high_resolution_clock::now(); }

  double elapsed() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

/**
 * @brief Builds the multigrid hierarchy using surface-first sampling.
 *
 * This is the main entry point for constructing the multigrid levels. It
 * performs the following steps:
 * 1. Sorts the mesh to prioritize surface vertices (and sharp features).
 * 2. Extracts the surface triangulation.
 * 3. Computes vertex depth (distance from surface).
 * 4. Iteratively coarsens the mesh by sampling vertices and building
 * prolongation matrices.
 *
 * @param input_tet_mesh The input tetrahedral mesh.
 */
void TetMultigridSolver::constructProlongationOurs(
    const TetrahedralMesh &input_tet_mesh, const std::string &output_dir_arg) {
  if (verbose) {
    std::cout << "constructProlongationOurs( ): input tetrahedral mesh ("
              << input_tet_mesh.numTetrahedra() << " tetrahedra, "
              << input_tet_mesh.numVertices() << " vertices )\n";
  }

  // Decide timing output path early so we can safely export even if we exit
  // early (e.g., due to unexpected input or future guard-returns).
  std::string timingPath;
  {
    std::string effectiveOutputDir =
        output_dir_arg.empty() ? outputDir : output_dir_arg;
    if (!effectiveOutputDir.empty()) {
      std::string sep = "/";
      if (effectiveOutputDir.back() == '/' ||
          effectiveOutputDir.back() == '\\') {
        sep = "";
      }
      timingPath = effectiveOutputDir + sep + "hierarchy_timing.json";
    }
  }
  try {
    if (!timingPath.empty()) {
      std::filesystem::path timingPathFs(timingPath);
      if (timingPathFs.has_parent_path()) {
        std::filesystem::create_directories(timingPathFs.parent_path());
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Warning: failed to create timing output directory for '"
              << timingPath << "': " << e.what() << "\n";
  }

  // =========================================================================
  // TIMING: Single timer with reset pattern
  // =========================================================================
  // One timer for everything. After each timed operation:
  //   1. Record elapsed time to appropriate variable
  //   2. Reset timer for next operation
  // This ensures no timing gaps and keeps code clean.
  
  Timer timer;
  Timer total_timer; // Dedicated timer for total duration (unaffected by resets)
  benchmark = BenchmarkData();
  
  // Guard to ensure timing is exported even on early exit
  bool timing_exported = false;
  struct TimingExportGuard {
    TetMultigridSolver *self;
    Timer *t;
    std::string path;
    bool *exported;
    ~TimingExportGuard() {
      if (self && t && exported && !(*exported)) {
        self->benchmark.total_hierarchy_ms = t->elapsed();
        self->exportHierarchyTimingJSON(path);
      }
    }
  } timing_guard{this, &total_timer, timingPath, &timing_exported};

  // =========================================================================
  // MARK: Init
  // =========================================================================

  // Clear any previous hierarchy data
  clearHierarchy();
  allExteriorIndices.clear();

  // Store input mesh for reference
  inputMesh = input_tet_mesh;
  benchmark.num_vertices_start = input_tet_mesh.numVertices();

  // Get references to the working data
  Eigen::MatrixXd &vertices_start = inputMesh.vertices;
  Eigen::MatrixXi &tetrahedra_start = inputMesh.tetrahedra;

  // Build neighbor matrix from tetrahedra
  Eigen::MatrixXi neighbor_matrix_tet_start = buildNeighborMatrixFromTetrahedra(
      inputMesh.tetrahedra, vertices_start.rows());

  // Push level 0 data
  all_vertices.push_back(vertices_start);
  all_tetrahedra.push_back(tetrahedra_start);
  allNeigh.push_back(neighbor_matrix_tet_start);
  nr_points.push_back(vertices_start.rows());

  // loop initializations
  int level_index = 0;
  Eigen::MatrixXd vertices_prev = vertices_start;
  Eigen::MatrixXi neighbor_matrix_tet_prev = neighbor_matrix_tet_start;

  // Record entire init phase as one bucket
  benchmark.init_ms = timer.elapsed();
  timer.reset();

  // =========================================================================
  // MARK: Extract Boundary (once at start)
  // =========================================================================
  std::vector<int> initial_exterior_priority;
  std::cout << "Building Ours hierarchy...\n";
  std::cout << "  Extracting boundary graph...\n";
  buildExteriorGraph();
  benchmark.init_exterior_extraction_ms = timer.elapsed();

  // Record number of exterior vertices at level 0
  if (!allExteriorIndices.empty()) {
    numSurfaceVerticesLevel0 = static_cast<int>(allExteriorIndices[0].size());
    initial_exterior_priority = allExteriorIndices.front();
  }
  timer.reset();

  // =========================================================================
  // MARK: Feature Priority
  // =========================================================================
  
  double initial_sort_feature_ms = 0.0;

  // Save the ORIGINAL exterior indices before any sorting.
  // allExteriorNeigh[0] uses local indices that correspond to the ORIGINAL
  // ordering of allExteriorIndices[0]. If we sort the indices by feature,
  // the local->global mapping in the neigh matrix becomes invalid.
  // coarseGraphBuilderExteriorFirst needs the original ordering for remapping.
  std::vector<int> original_exterior_indices_level0;
  if (!allExteriorIndices.empty()) {
    original_exterior_indices_level0 = allExteriorIndices[0];
  }

  // Persistent state for the multi-level exterior-first hierarchy:
  // After each exterior-first iteration, these are populated for the NEXT iteration.
  // - original indices: needed for local->global remapping in coarseGraphBuilderExteriorFirst
  // - priority: feeds exterior indices into fastDiskSample for priority sampling
  std::vector<int> original_exterior_indices_for_next_level = original_exterior_indices_level0;
  std::vector<int> next_level_exterior_priority;

  if (!allExteriorIndices.empty()) {
    std::cout << "  Sorting boundary feature priority...\n";
    sortExteriorIndicesByFeature(0);
    initial_sort_feature_ms = timer.elapsed();
    initial_exterior_priority = allExteriorIndices.front();
    timer.reset();
  }

  // =========================================================================
  // MARK: Loop 
  // =========================================================================
  while (nr_points.back() > static_cast<size_t>(minVerts) &&
         level_index < maxLevels) {

    level_index++;
    std::cout << "Level " << level_index << ": sampling coarse points...\n";

    // Level timing variables (all zero until measured)
    double t_sampling = 0.0;
    double t_sort_feature = 0.0;
    double t_coarse_graph = 0.0;
    double t_simplicial = 0.0;
    double t_interpolation = 0.0;

    timer.reset();  // Start fresh for this level

    std::vector<int> exterior_priority_indices;
    if (!initial_exterior_priority.empty()) {
      exterior_priority_indices = std::move(initial_exterior_priority);
      t_sort_feature = initial_sort_feature_ms;
      initial_exterior_priority.clear();
    } else if (!next_level_exterior_priority.empty()) {
      // Level 2+: use the coarse exterior indices propagated
      // from the previous iteration's coarseGraphBuilderExteriorFirst.
      exterior_priority_indices = std::move(next_level_exterior_priority);
      next_level_exterior_priority.clear();
    }

    timer.reset();  // Restart timer for sampling stage

    // =========================================================================
    // MARK: SAMPLING
    // =========================================================================

    // Calculate sampling radius
    double avg_edge = computeAverageEdgeLengthFromNeigh(vertices_prev, neighbor_matrix_tet_prev);
    double radius = std::cbrt(searchRadiusFactor) * avg_edge;

    // Initialize distance tracking
    Eigen::VectorXd shortestDistanceToSample_prev_to_next(vertices_prev.rows());
    shortestDistanceToSample_prev_to_next.setConstant(std::numeric_limits<double>::max());
    std::vector<size_t> nearestSource_prev_to_next(vertices_prev.rows());

    // Fast disk sampling
    std::vector<int> sample_indices_next = fastDiskSample(
      vertices_prev, neighbor_matrix_tet_prev, radius,
      shortestDistanceToSample_prev_to_next, nearestSource_prev_to_next,
      exterior_priority_indices);
    const int sample_count_next =
        static_cast<int>(sample_indices_next.size());

        
    allExteriorIndices.push_back(std::move(exterior_priority_indices));

    t_sampling = timer.elapsed();
    timer.reset();


    // =========================================================================
    // MARK: COARSE GRAPH BUILD (Dijkstra + Graph + Vertex Neighbours)
    // =========================================================================

    std::vector<std::vector<int>> coarse_adj_list_next;
    Eigen::MatrixXi coarseNeigh_next;
    int numCoarseExterior = 0;
    std::cout << "Level " << level_index << ": building exterior-first coarse graph...\n";

    if (allExteriorNeigh.size() >= static_cast<size_t>(level_index) &&
        !allExteriorIndices.empty()) {
      // The sampled exterior vertices are stored first and define the coarse exterior set.
      numCoarseExterior = static_cast<int>(allExteriorIndices.back().size());

      const Eigen::MatrixXi &fineExtNeigh = allExteriorNeigh[level_index - 1];
      const std::vector<int> &fineOrigExtIndices =
          (level_index == 1) ? original_exterior_indices_level0
                             : original_exterior_indices_for_next_level;

      coarseGraphBuilderExteriorFirst(
          vertices_prev, sample_indices_next, neighbor_matrix_tet_prev,
          sample_count_next,
          numCoarseExterior,
          fineExtNeigh,
          fineOrigExtIndices,
          shortestDistanceToSample_prev_to_next, nearestSource_prev_to_next,
          coarse_adj_list_next, coarseNeigh_next);

      original_exterior_indices_for_next_level.resize(numCoarseExterior);
      std::iota(original_exterior_indices_for_next_level.begin(),
                original_exterior_indices_for_next_level.end(), 0);
      next_level_exterior_priority.resize(numCoarseExterior);
      std::iota(next_level_exterior_priority.begin(),
                next_level_exterior_priority.end(), 0);
    } else {
      coarseGraphBuilder(
          vertices_prev, sample_indices_next, neighbor_matrix_tet_prev,
          sample_count_next,
          shortestDistanceToSample_prev_to_next, nearestSource_prev_to_next,
          coarse_adj_list_next, coarseNeigh_next);
    }

    t_coarse_graph = timer.elapsed();
    timer.reset();

    // =========================================================================
    // MARK: SIMPLICIAL COMPLEX BUILD & PROLONGATION MATRIX
    // =========================================================================

    std::vector<std::vector<int>> tris;
    std::vector<std::vector<int>> connectedTris;
    std::vector<std::vector<int>> tets;
    std::vector<std::vector<int>> connectedTets;
    Eigen::SparseMatrix<double> P;

    // Extract sample vertex positions
    Eigen::MatrixXd sample_vertices_next(sample_count_next, 3);
    for (size_t i = 0; i < sample_indices_next.size(); ++i) {
      sample_vertices_next.row(i) = vertices_prev.row(sample_indices_next[i]);
    }

    double t_smooth_interior = 0.0;
    std::cout << "Level " << level_index << ": building simplicial complex...\n";

    auto simplicialComplex =
      buildSimplicialComplex(sample_vertices_next, coarse_adj_list_next);
    tris = simplicialComplex[0];
    connectedTris = simplicialComplex[1];
    tets = simplicialComplex[2];
    connectedTets = simplicialComplex[3];

    t_simplicial = timer.elapsed();
    timer.reset();

    std::cout << "Level " << level_index << ": assembling prolongation...\n";

    P = constructGeometricProlongationMatrix(
      vertices_prev, sample_vertices_next, sample_indices_next,
      nearestSource_prev_to_next, coarse_adj_list_next, tets, connectedTets,
      tris, connectedTris);

    t_interpolation = timer.elapsed();
    timer.reset();

    allP.push_back(P);

    // =========================================================================
    // Record level timings
    // =========================================================================
    benchmark.level_sampling_ms.push_back(t_sampling);
    benchmark.level_sampling_priority_ms.push_back(0.0);
    benchmark.level_exterior_detect_ms.push_back(0.0);
    benchmark.level_sort_feature_ms.push_back(t_sort_feature);
    benchmark.level_coarse_graph_ms.push_back(t_coarse_graph);
    benchmark.level_smooth_interior_ms.push_back(t_smooth_interior);
    benchmark.level_simplicial_complex_ms.push_back(t_simplicial);
    benchmark.level_interpolation_ms.push_back(t_interpolation);

    double t_total_level = t_sampling + t_sort_feature +
                 t_coarse_graph + t_smooth_interior + t_simplicial + t_interpolation;
    benchmark.level_total_ms.push_back(t_total_level);

    // =========================================================================
    // MARK: Store hierarchy data for this level (class members for Python bindings)
    // =========================================================================

    // Core geometry data
    all_vertices.push_back(sample_vertices_next);

    // Convert tets to MatrixXi
    Eigen::MatrixXi tetsMatrix(tets.size(), 4);
    for (size_t i = 0; i < tets.size(); ++i) {
      for (int j = 0; j < 4; ++j)
        tetsMatrix(i, j) = tets[i][j];
    }
    all_tetrahedra.push_back(tetsMatrix);

    allNeigh.push_back(coarseNeigh_next);

    // Store exterior vertex indices for this level (from solid angle
    // computation) These are the vertices identified as being on the exterior
    // surface
    boundary_indices.push_back(allExteriorIndices.back());

    nr_points.push_back(sample_vertices_next.rows());

    // Store clustering data (fine-to-coarse mapping - crucial for
    // restrict/prolong)
    allNearestSource.push_back(nearestSource_prev_to_next);
    allSampleIndices.push_back(sample_indices_next);

    // Store simplicial complex data.
    allTris.push_back(tris);
    allTets.push_back(tets);
    allNeighSets.push_back(coarse_adj_list_next);

    // Advance to the next level.
    vertices_prev = sample_vertices_next;
    neighbor_matrix_tet_prev = coarseNeigh_next;
  }

  // =========================================================================
  // MARK: Export timing data
  // =========================================================================

  benchmark.total_hierarchy_ms = total_timer.elapsed();

  {
    double sum_components = benchmark.init_ms
        + benchmark.init_exterior_extraction_ms
        + std::accumulate(benchmark.level_total_ms.begin(), benchmark.level_total_ms.end(), 0.0)
        + benchmark.final_exterior_graph_ms;
    double drift = std::abs(benchmark.total_hierarchy_ms - sum_components);
    if (drift > 5.0 && verbose) {
      std::cerr << "WARNING: Timing partition drift = " << drift
                << " ms (total=" << benchmark.total_hierarchy_ms
                << ", sum=" << sum_components << ")\n";
    }
  }

  exportHierarchyTimingJSON(timingPath);
  timing_exported = true;

  std::cout << "Ours hierarchy ready: ";
  for (size_t i = 0; i < nr_points.size(); ++i) {
    std::cout << nr_points[i];
    if (i < nr_points.size() - 1) {
      std::cout << " -> ";
    }
  }
  std::cout << " vertices (" << benchmark.total_hierarchy_ms << " ms)\n";
}

} // namespace GravoMG
