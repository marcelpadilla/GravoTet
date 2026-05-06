#ifndef TET_MULTIGRID_SOLVER_H
#define TET_MULTIGRID_SOLVER_H

#include <vector>
#include <set>
#include <map>
#include <limits>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif
#include <iostream>
#include <fstream>
#include <random>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace GravoMG {

	// Structure for Dijkstra's algorithm priority queue
	struct VertexPair {
		int vId;
		double distance;
		bool operator>(const VertexPair& ref) const { return distance > ref.distance; }
		bool operator<(const VertexPair& ref) const { return distance < ref.distance; }
	};

	// Simple tetrahedral mesh structure using Eigen matrices
	struct TetrahedralMesh {
		Eigen::MatrixXd vertices;    // Vertex positions: Nx3 matrix where each row is [x, y, z]
		Eigen::MatrixXi tetrahedra;  // Tetrahedra connectivity: Mx4 matrix where each row is [v0, v1, v2, v3]
		
		// Default constructor
		TetrahedralMesh() {}
		
		// Constructor with sizes
		TetrahedralMesh(int numVertices, int numTetrahedra) 
			: vertices(numVertices, 3), tetrahedra(numTetrahedra, 4) {}
		
		// Get number of vertices and tetrahedra
		int numVertices() const { return vertices.rows(); }
		int numTetrahedra() const { return tetrahedra.rows(); }
	};

	// Simple triangle mesh structure using Eigen matrices
	struct TriangleMesh {
		Eigen::MatrixXd vertices;    // Vertex positions: Nx3 matrix where each row is [x, y, z]
		Eigen::MatrixXi triangles;   // Triangle connectivity: Mx3 matrix where each row is [v0, v1, v2]
		
		// Default constructor
		TriangleMesh() {}
		
		// Constructor with sizes
		TriangleMesh(int numVertices, int numTriangles) 
			: vertices(numVertices, 3), triangles(numTriangles, 3) {}
		
		// Constructor from matrices
		TriangleMesh(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& tris)
			: vertices(verts), triangles(tris) {}
		
		// Get number of vertices and triangles
		int numVertices() const { return vertices.rows(); }
		int numTriangles() const { return triangles.rows(); }
	};

	// Tetrahedral multigrid solver class
	// Samples surface points first using FASTDISK, then samples interior points
	class TetMultigridSolver
	{
	public:
        // Constructor
		TetMultigridSolver();
        
        // Type alias for Row-Major Sparse Matrix (optimizes SpMV and Smoothing)
        using SparseMatrixCSR = Eigen::SparseMatrix<double, Eigen::RowMajor>;

		~TetMultigridSolver();


		


		// Load tetrahedral mesh from PLY format
		static TetrahedralMesh loadFromPLY(const std::string& filename);

		// Generate a cube tetrahedral mesh with N subdivisions per axis
		static TetrahedralMesh generateCubeMesh(int resolution);



		
		// Build the supplementary hierarchy using the Ours construction.
		// - input_mesh:	TetrahedralMesh containing both vertices and connectivity
		// - output_dir:    Directory for output files (optional override)
		void constructProlongationOurs(const TetrahedralMesh& input_mesh, const std::string& output_dir_arg = "");
		
		// Clear all hierarchy data (reset the solver)
		void clearHierarchy();
		
		// Compute exterior visualization data on-demand (lazy computation)
		// This is called when visualization requests exterior data for featurePreserve=false mode.
		// For featurePreserve=true, data is already populated during hierarchy construction.
		// Returns true if data was computed/available, false otherwise.
		bool computeExteriorVisualizationData();
		
		// Get the number of levels in the hierarchy (1 = only finest level, no coarsening)
		int numLevels() const { return static_cast<int>(all_vertices.size()); }
		
		// Check if hierarchy has been built
		bool isHierarchyBuilt() const { return !all_vertices.empty(); }

		// Compute Galerkin coarse operators (R * A * P) for all levels.
		// Takes the fine level matrix A (level 0) and returns the list of coarse matrices.
		// Returns a vector where result[i] is the operator for level i+1.
        std::vector<Eigen::SparseMatrix<double>> computeCoarseOperators(
			const Eigen::SparseMatrix<double>& A_fine
        );

		// Compute RAP product (R * A * P) using optimized parallel implementation
		static Eigen::SparseMatrix<double> computeProductRAP(
			const Eigen::SparseMatrix<double>& R,
			const Eigen::SparseMatrix<double>& A,
			const Eigen::SparseMatrix<double>& P
		);

        // Perform Gauss-Seidel smoothing iterations in C++ (much faster setup than Python SPLU)
        // Computes x_new = GS(A, b, x_old)
        // Iterates: x_i = (b_i - sum_{j!=i} A_ij * x_j) / A_ii
        // Assumes A is CSR. Returns the updated x.
        Eigen::VectorXd computeGaussSeidel(
            const Eigen::SparseMatrix<double>& A,
            const Eigen::VectorXd& b,
            const Eigen::VectorXd& x,
            const int iterations,
            const bool sweep_backward = false,
            const bool symmetric = false
        );
        
        // Overload for Row-Major matrices (used in internal V-cycle)
        Eigen::VectorXd computeGaussSeidel(
            const SparseMatrixCSR& A,
            const Eigen::VectorXd& b,
            const Eigen::VectorXd& x,
            const int iterations,
            const bool sweep_backward = false,
            const bool symmetric = false
        );

        // ========================================================================
        // V-CYCLE SOLVER - Full multigrid solve in C++
        // ========================================================================
        
        // Smoother type enumeration (extensible for future smoothers)
        enum class SmootherType {
            JACOBI,
            GAUSS_SEIDEL
        };
        
        // V-Cycle solve result structure
        struct VCycleSolveResult {
            Eigen::VectorXd x;              // Solution vector
            int num_cycles;                  // Number of cycles performed
            std::vector<double> residual_history;  // Residual norm after each cycle
			std::vector<double> cycle_time_ms_history;  // Wall-clock time per V-cycle
            bool converged;                  // True if tolerance reached
            bool timed_out;                  // True if timeout reached
            double total_time_ms;            // Total solve time
            
            // Detailed timing breakdown (in milliseconds, cumulative over all cycles)
            double sweep_time_ms = 0.0;
            double restriction_time_ms = 0.0;
            double prolongation_time_ms = 0.0;
            double coarse_solve_time_ms = 0.0;
            double residual_time_ms = 0.0;
            std::string coarse_solver_name;
        };
        
        // Perform damped Jacobi smoothing iterations
        // x_new = x + omega * D^{-1} * (b - A*x)
        // Extensible design: matches signature of computeGaussSeidel
        Eigen::VectorXd computeJacobi(
            const Eigen::SparseMatrix<double>& A,
            const Eigen::VectorXd& b,
            const Eigen::VectorXd& x,
            int iterations,
            double omega = 0.6667
        );
        
        // Build V-cycle hierarchy (operators and factorize coarse solver)
        // Must be called before solveVCycle
        // Returns true if hierarchy was successfully built
		bool buildVCycleHierarchy(const Eigen::SparseMatrix<double>& A_fine);
        
		// Set external prolongation operators for the V-cycle solver.
		// When set, buildVCycleHierarchy will use these instead of the internal hierarchy.
        void setExternalProlongations(
            const std::vector<Eigen::SparseMatrix<double>>& prolongations
        );
        
        // Clear external prolongations (revert to using internal hierarchy)
        void clearExternalProlongations();
        
        // Check if external prolongations are set
        bool hasExternalProlongations() const { return use_external_prolongations_; }
        
        // Solve Ax = b using V-cycle multigrid
        // Requires buildVCycleHierarchy to have been called first
        VCycleSolveResult solveVCycle(
            const Eigen::VectorXd& b,
            const Eigen::VectorXd& x0,
            int max_cycles = 100,
            int pre_sweeps = 2,
            int post_sweeps = 2,
            double tol = 1e-6,
            double timeout_ms = 0.0,
            SmootherType smoother = SmootherType::JACOBI,
            double jacobi_omega = 0.6667,
            bool collect_timing = false,
            const Eigen::VectorXd& mass_diag_inv = Eigen::VectorXd()
        );
        
        // Check if V-cycle hierarchy is ready
        bool isVCycleHierarchyBuilt() const { return vcycle_hierarchy_built_; }
        
        // Get the coarse solver name (for reporting)
        std::string getCoarseSolverName() const { return coarse_solver_name_; }

		// Hierarchy parameters
		int minVerts = 500;						// Minimum vertices at coarsest level (stops coarsening when reached)
		int maxLevels = 100;					// Maximum number of hierarchy levels (safety limit)
		int directSolveThreshold = 12000;		// Vertex count below which direct solving is efficient (from scaling tests)
		double searchRadiusFactor = 2.0;		// Search radius factor for sampling (sampling radius = cbrt(factor) * avg_edge_length)
		bool verbose = true;					// Print progress information
		bool featurePreserve = true;			// Preserve sharp boundary features during coarsening.
		std::string outputDir = "";				// Output directory for timing JSON files (empty = current dir)
		bool use_dense_coarse_solver_ = false;  // Use dense LDLT for coarsest level (faster for small systems)

		// ========================================================================
		// HIERARCHY DATA - Persistent data structures for Python bindings & solver
		// ========================================================================
		// These members are populated by constructProlongationOurs() and remain accessible
		// for later use by the solver or Python bindings.
		
		// --- Input Data (preserved for reference) ---
		TetrahedralMesh inputMesh;							// Original input mesh
		int numSurfaceVerticesLevel0 = 0;					// Number of surface vertices at level 0
		
		// --- Per-Level Hierarchy Data ---
		std::vector<size_t> nr_points;										// Degrees of freedom per level
		std::vector<Eigen::MatrixXd> all_vertices;							// Vertex positions per level
		std::vector<Eigen::MatrixXi> all_tetrahedra;							// Tetrahedra per level (from input or simplicial complex build)
		std::vector<std::vector<int>> boundary_indices;						// Boundary vertex indices per level
		std::vector<Eigen::SparseMatrix<double>> allP;						// Prolongation operators (P[i] maps level i+1 to level i)
		std::vector<Eigen::MatrixXi> allNeigh;								// Neighborhood matrices per level
		
		// --- Clustering Data (needed for restriction/prolongation) ---
		std::vector<std::vector<size_t>> allNearestSource;					// Fine-to-coarse vertex mapping per level
		std::vector<std::vector<int>> allSampleIndices;					// Sample vertex indices (exterior + interior) per level
		
		// --- Simplicial Complex Data (from coarse point clouds) ---
		std::vector<std::vector<std::vector<int>>> allTris;					// Triangles per level (from simplicial complex build)
		std::vector<std::vector<std::vector<int>>> allTets;					// Tetrahedra per level (from simplicial complex build)
		std::vector<std::vector<std::vector<int>>> allNeighSets;				// Neighbor sets per level (for simplicial complex build)
		
		// --- Exterior Mesh Data ---
		std::vector<Eigen::MatrixXd> allExteriorVertices;					// Exterior vertices per level
		std::vector<Eigen::MatrixXi> allExteriorNeigh;						// Exterior neighborhood matrices per level
		std::vector<std::vector<int>> allExteriorIndices;					// Exterior vertex indices per level (before sampling)
		std::vector<std::vector<double>> allExteriorFeature;				// Solid angle sums for exterior vertices per level
		
		// Mass matrix and its inverse (for solver)
		Eigen::SparseMatrix<double> M;
		Eigen::SparseMatrix<double> MInv;
		
		// Random number generator
		std::mt19937 generator;
		
		// Benchmarking data
		// PARTITION RULE: total_hierarchy_ms == init_ms + init_exterior_extraction_ms
		//                 + sum(level_total_ms) + final_exterior_graph_ms
		struct BenchmarkData {
			// --- Init phase (everything before the coarsening loop) ---
			double init_ms = 0.0;                        // clearHierarchy + adjacency build + push level 0 + loop var init
			double init_exterior_extraction_ms = 0.0;     // buildExteriorGraphNEWALT (ours_surf/feat/pro only)

			// --- Per-level coarsening loop ---
			std::vector<double> level_sampling_ms;
			std::vector<double> level_sampling_priority_ms;
			std::vector<double> level_exterior_detect_ms;
			std::vector<double> level_sort_feature_ms;
			std::vector<double> level_coarse_graph_ms;       // Includes Dijkstra clustering + coarse graph build + vertex neighbours
			std::vector<double> level_smooth_interior_ms;     // Reserved slot for supplementary timing compatibility
			std::vector<double> level_simplicial_complex_ms;
			std::vector<double> level_interpolation_ms;
			std::vector<double> level_total_ms;

			// --- Finalize phase (after the coarsening loop) ---
			double final_exterior_graph_ms = 0.0;         // Reserved for potential post-loop processing

			// --- Totals ---
			double total_hierarchy_ms = 0.0;              // Wall-clock total (must equal sum of all buckets)
			int num_vertices_start = 0;
		} benchmark;
		


		// Export benchmark data to JSON (simple format for easy Python parsing)
		void exportHierarchyTimingJSON(const std::string& filename);


		// Compute average edge length using neighborhood matrix (matches reference implementation)
		double computeAverageEdgeLengthFromNeigh(
			const Eigen::MatrixXd& verts,
			const Eigen::MatrixXi& neigh
		);

		// Fast disk sampling (matches reference implementation)
		// Returns indices of sampled vertices
		// - pos:				Vertex positions (Nx3 matrix)
		// - edges:				Neighborhood matrix (NxX matrix, -1 indicates no more neighbors)
		// - radius:			Sampling radius
		// - shortestDistanceToSample:	Output: shortest distance to nearest sample for each vertex
		// - nearestSourceK:	Output: index of nearest sample for each vertex
	       std::vector<int> fastDiskSample_exterior(
		       const Eigen::MatrixXd& pos,
		       const Eigen::MatrixXi& edges,
		       const double& radius,
		       Eigen::VectorXd& shortestDistanceToSample,
		       std::vector<size_t>& nearestSourceK,
		       Eigen::MatrixXd& sampledVertices
	       );


		// Construct clusters using Dijkstra's algorithm (works for any neighbor matrix)
		// Assigns each point to its nearest sample point based on geodesic distance
		// Works on any connectivity (surface triangles, tetrahedral mesh, or any graph)
		void constructDijkstraWithCluster(
			const Eigen::MatrixXd& points,
			const std::vector<int>& source,
			const Eigen::MatrixXi& neigh,
			Eigen::VectorXd& shortestDistanceToSample,
			std::vector<size_t>& nearestSource
		);
		
		// Two-phase Dijkstra clustering for pro_priority mode
		// Phase 1: Only exterior vertices are clustered using exterior-only paths
		// Phase 2: Interior vertices are clustered, preserving Phase 1 assignments for exterior vertices
		// This explicitly isolates surface clustering from interior influence
		void constructDijkstraExteriorFirst(
			const Eigen::MatrixXd& points,
			const std::vector<int>& allSources,
			const Eigen::MatrixXi& volumeNeigh,
			Eigen::VectorXd& shortestDistanceToSample,
			std::vector<size_t>& nearestSource,
			const std::vector<int>& exteriorIndices
		);

		// Create neighborhood list for coarser level (matches reference implementation)
		// Returns a vector of vectors, where each vector contains the sorted indices of neighboring coarse points
		std::vector<std::vector<int>> buildCoarseGraph(
			const Eigen::MatrixXi& fineNeigh,
			const std::vector<size_t>& nearestSource,
			int numCoarsePoints
		);

		// Build vertex neighbors matrix for coarser level from precomputed adjacency list
		// Returns a matrix where each row contains neighbor indices (-1 for no more neighbors)
		// Self is always the first neighbor (at index 0)
		Eigen::MatrixXi buildVertexNeighbours(
			const std::vector<std::vector<int>>& adjList,
			int numCoarsePoints
		);

		// Fused coarse graph construction: Dijkstra clustering + coarse graph build + vertex neighbours
		// Combines constructDijkstraWithCluster, buildCoarseGraph, and buildVertexNeighbours into one call.
		// Outputs are written to the provided references.
		void coarseGraphBuilder(
			const Eigen::MatrixXd& vertices,
			const std::vector<int>& sampleIndices,
			const Eigen::MatrixXi& fineNeigh,
			int numCoarsePoints,
			Eigen::VectorXd& shortestDistanceToSample,
			std::vector<size_t>& nearestSource,
			std::vector<std::vector<int>>& coarseAdjList,
			Eigen::MatrixXi& coarseNeigh
		);

		// Two-phase coarse graph builder for the supplementary Ours method.
		// Phase 1: Cluster exterior vertices using exterior-only connectivity
		//          (from allExteriorNeigh[0]) and build exterior-only coarse graph.
		// Phase 2: Cluster interior vertices via full volume graph, then merge
		//          interior edges with the exterior coarse graph (no duplicates).
		// Only used for level_index == 1 (first coarsening from the input mesh).
		// Stores the exterior coarse adjacency in allExteriorNeigh for future levels.
		void coarseGraphBuilderExteriorFirst(
			const Eigen::MatrixXd& vertices,
			const std::vector<int>& sampleIndices,
			const Eigen::MatrixXi& fineVolumeNeigh,
			int numCoarsePoints,
			int numCoarseExterior,
			const Eigen::MatrixXi& fineExteriorNeigh,
			const std::vector<int>& fineExteriorIndices,
			Eigen::VectorXd& shortestDistanceToSample,
			std::vector<size_t>& nearestSource,
			std::vector<std::vector<int>>& coarseAdjList,
			Eigen::MatrixXi& coarseNeigh
		);

		// Build neighborhood matrix from tetrahedra connectivity
		static Eigen::MatrixXi buildNeighborMatrixFromTetrahedra(
			const Eigen::MatrixXi& tetrahedra,
			int numVertices
		);


		// Compute boundary vertices AND surface triangles from tetrahedral mesh
		std::vector<int> computeBoundaryVertices(const TetrahedralMesh& mesh, Eigen::MatrixXi& surfaceTriangles);
		
		// =======================================================================
		// SOLID ANGLE APPROACH FOR EXTERIOR/INTERIOR CLASSIFICATION
		// =======================================================================
		// Computes exterior (boundary) vertices using solid angle sum at each vertex.
		// For each vertex, sums the solid angles of all incident tetrahedra.
		// - If sum ≈ 4π: interior vertex (fully enclosed)
		// - If sum < 4π: exterior vertex (on boundary)  
		// - If no tets: exterior vertex (isolated)
		// Also returns the solid angle sum for each vertex (useful for feature sorting).
		//
		// @param mesh          Tetrahedral mesh
		// @param solidAngleSums Output: solid angle sum for each vertex (size = numVertices)
		// @param threshold     Fraction of 4π below which a vertex is considered exterior (default 0.99)
		// @return              Indices of exterior vertices, sorted by solid angle sum (ascending = sharpest first)
		std::vector<int> computeExteriorVerticesBySolidAngle(
			const TetrahedralMesh& mesh,
			std::vector<double>& solidAngleSums,
			double threshold = 0.99
		);
		
		// Compute solid angle at a vertex of a tetrahedron
		// The solid angle is the area on a unit sphere subtended by the tetrahedron at that vertex
		// @param v0 The vertex position at which to compute the solid angle
		// @param v1, v2, v3 The other three vertices of the tetrahedron
		// @return Solid angle in steradians (0 to 2π for a valid tet corner)
		static double computeTetSolidAngle(
			const Eigen::RowVector3d& v0,
			const Eigen::RowVector3d& v1,
			const Eigen::RowVector3d& v2,
			const Eigen::RowVector3d& v3
		);
		
		// Optimized exterior graph builder using solid-angle vertex classification
		// + connectivity-based dihedral angle feature computation (no face hashing).
		// Algorithm:
		//   1. Classify exterior vertices via solid angle sum (parallel, per-tet)
		//   2. Identify exterior edges from the neighbour matrix (both endpoints exterior)
		//      and filter using dihedral angle sum around each edge (< 2π = exterior)
		//   3. Compute per-vertex feature: for each exterior edge, find the two
		//      triangular faces via shared exterior neighbours, compute normal angle.
		// All phases are fully parallelizable with no hash maps.
		// Populates: allExteriorVertices, allExteriorNeigh, allExteriorFeature
		void buildExteriorGraphNEWALT();
		
		// Compute dihedral angle at an edge in a tetrahedron
		// The dihedral angle is the angle between the projections of the two
		// non-edge vertices onto the plane perpendicular to the edge.
		// @param v0, v1 The edge vertices
		// @param v2, v3 The other two tetrahedron vertices
		// @return Dihedral angle in radians (0 to pi)
		static double computeTetDihedralAngle(
			const Eigen::RowVector3d& v0,
			const Eigen::RowVector3d& v1,
			const Eigen::RowVector3d& v2,
			const Eigen::RowVector3d& v3
		);
		
		// Sort priority indices by solid angle sum (ascending = sharpest/lowest first)
		// Vertices with lower solid angle sums are sharper features (corners, edges)
		void sortPriorityIndicesBySolidAngle(
			std::vector<int>& Exterior_indices,
			const std::vector<double>& solidAngleSums
		);

		// Sort exterior indices by feature value (descending) for a specific level
		void sortExteriorIndicesByFeature(int level);
		
		// Build neighborhood matrix from tetrahedra connectOptional: indices to sample first first (if provided) and then the rest.
        // If Exterior_indices is empty, it behaves like a standard linear scan.
        // - pos:                       All vertex positions
		// - tetNeigh:                  Neighborhood matrix
		// - radius:                    Sampling radius
		// - shortestDistanceToSample:  Output: shortest distance to nearest sample
		// - nearestSourceK:            Output: index of nearest sample
		// - Exterior_indices:           Input/Output: indices to sample first.
        //                              Updated to contain only the indices that were actually sampled.
		std::vector<int> fastDiskSample(
			const Eigen::MatrixXd& pos,
			const Eigen::MatrixXi& tetNeigh,
			const double& radius,
			Eigen::VectorXd& shortestDistanceToSample,
			std::vector<size_t>& nearestSourceK,
            std::vector<int>& Exterior_indices
		);



        // Construct geometric prolongation matrix
        Eigen::SparseMatrix<double> constructGeometricProlongationMatrix(
            const Eigen::MatrixXd& fineVertices,
            const Eigen::MatrixXd& coarseVertices,
            const std::vector<int>& coarseIndices,
            const std::vector<size_t>& nearestCoarseIdx,
            const std::vector<std::vector<int>>& coarseNeighborList,
            const std::vector<std::vector<int>>& tets,
            const std::vector<std::vector<int>>& connectedTets,
            const std::vector<std::vector<int>>& tris,
            const std::vector<std::vector<int>>& connectedTris
        );

		// Test prolongation matrix properties
		// Performs essential validation tests on a prolongation matrix
		// Tests: 1) Dimensions, 2) Partition of unity, 3) Non-negativity, 4) Constant preservation
		// Returns true if all tests pass, false otherwise
		static bool testProlongationMatrix(
			const Eigen::SparseMatrix<double>& P,
			const Eigen::MatrixXd& vertsFine,
			const Eigen::MatrixXd& vertsCoarse,
			int levelIndex
		);

		// Build simplicial complex from coarse point cloud using neighbor connectivity
		// Returns: [triangles, connectedTris, tets, connectedTets]
		// Uses efficient sorted intersection for clique finding
		static std::array<std::vector<std::vector<int>>, 4> buildSimplicialComplex(
			const Eigen::MatrixXd& verts,
			const std::vector<std::vector<int>>& adjList
		);

		// Check if point is inside tetrahedron and compute barycentric coordinates
		// Returns metric (min bary coord) if inside, -1.0 if outside
		static double inTet(
			const Eigen::RowVector3d& p,
			const std::vector<int>& tet,
			const Eigen::MatrixXd& verts,
			Eigen::RowVector4d& bary
		);

		// Project point onto triangle and compute barycentric coordinates
		// Also checks projection onto edges and stores results in insideEdge map
		// Returns distance to triangle if projection is inside, -1.0 otherwise
		static double inTri(
			const Eigen::RowVector3d& p,
			const std::vector<int>& tri,
			const Eigen::MatrixXd& verts,
			Eigen::RowVector3d& bary,
			std::map<int, double>& insideEdge
		);

		// Compute inverse distance weights for interpolation
		// Returns normalized weights that sum to 1.0
		static std::vector<double> inverseDistanceWeights(
			const Eigen::MatrixXd& verts,
			const Eigen::RowVector3d& p,
			const std::vector<int>& indices
		);

		// Export sparse matrix in Matrix Market format (.mtx)
		// This format is widely supported and can be read by scipy.io.mmread() in Python
		static void exportSparseMatrix(
			const Eigen::SparseMatrix<double>& matrix,
			const std::string& filename
		);

	private:
        // ========================================================================
        // V-CYCLE SOLVER PRIVATE MEMBERS
        // ========================================================================
        
        // Type alias for Row-Major Sparse Matrix (optimizes SpMV and Smoothing)
        // using SparseMatrixCSR = Eigen::SparseMatrix<double, Eigen::RowMajor>; // Moved to public

        // Operators for each level (A_0, A_1, ..., A_L)
        // Stored in RowMajor format to enable efficient OpenMP parallelization of smoothing
        std::vector<SparseMatrixCSR> vcycle_operators_;
        
        // Prolongation operators for the active V-cycle hierarchy
        // These remain default (ColMajor) as that's efficient for matrix products
        std::vector<Eigen::SparseMatrix<double>> vcycle_prolongations_; 
        
        // Pre-computed diagonal inverses for Jacobi smoother (one per level except coarsest)
        std::vector<Eigen::VectorXd> vcycle_diag_inv_;
        
        // Coarse level direct solver (Eigen SimplicialLDLT - sparse, or Dense LDLT)
        // Sparse coarse solver (default)
        std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> coarse_solver_;
        // Dense coarse solver (optional, for small coarsest levels)
        std::unique_ptr<Eigen::LDLT<Eigen::MatrixXd>> dense_coarse_solver_;
        Eigen::MatrixXd dense_coarse_matrix_;  // stored for debugging/info only
        bool coarse_solver_valid_ = false;
        std::string coarse_solver_name_ = "SimplicialLDLT";
        
        // V-cycle hierarchy state
        bool vcycle_hierarchy_built_ = false;
        int vcycle_num_levels_ = 0;
        
		// External prolongation support
        bool use_external_prolongations_ = false;
        std::vector<Eigen::SparseMatrix<double>> external_prolongations_;
        
        // Jacobi damping parameter
        double jacobi_omega_ = 0.6667;
        
        // Chebyshev polynomial acceleration parameters (one per level except coarsest)
        // Pre-computed spectral radius estimates (λ_max of D^{-1}A)
        std::vector<double> vcycle_spectral_radius_;
        // Pre-computed Chebyshev omega coefficients per level (omega[s] for sweep s)
        std::vector<std::vector<double>> vcycle_chebyshev_omega_;
        // Number of sweeps used for Chebyshev coefficient computation
        int chebyshev_degree_ = 2;

        
        // Internal V-cycle recursive function
        Eigen::VectorXd vcycleRecursive(
            int level,
            const Eigen::VectorXd& x,
            const Eigen::VectorXd& b,
            int pre_sweeps,
            int post_sweeps,
            SmootherType smoother,
            bool collect_timing,
            VCycleSolveResult& timing
        );
	};

	// ============================================================================
	// Standalone Direct Solve (free function, no multigrid hierarchy needed)
	// ============================================================================

	/**
	 * @brief Result of a standalone direct solve via Eigen.
	 */
	struct DirectSolveResult {
		Eigen::VectorXd x;           ///< Solution vector
		double analyze_time_ms;       ///< Symbolic analysis / fill-reducing ordering
		double factorize_time_ms;     ///< Numerical factorization
		double solve_time_ms;         ///< Forward/back substitution
		double total_time_ms;         ///< Wall-clock total
		std::string solver_name;      ///< Human-readable solver name
		bool success;                 ///< True if solve completed without error
		std::string error;            ///< Error message if !success
	};

	/**
	 * @brief Solve Ax = b using a standalone Eigen direct solver.
	 *
	 * Supports solver_type:
	 *   - "eigen-llt"  : Eigen SimplicialLLT (Cholesky LL^T, AMD ordering)
	 *   - "eigen-ldlt" : Eigen SimplicialLDLT (Cholesky LDL^T, AMD ordering)
	 *
	 * @param A  Sparse SPD matrix (ColMajor)
	 * @param b  Right-hand side vector
	 * @param solver_type  One of the supported solver type strings
	 * @return DirectSolveResult with solution, timing breakdown, and status
	 */
	DirectSolveResult solveDirectEigen(
		const Eigen::SparseMatrix<double>& A,
		const Eigen::VectorXd& b,
		const std::string& solver_type
	);

}

#endif // TET_MULTIGRID_SOLVER_H
