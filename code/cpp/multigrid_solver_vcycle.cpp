/**
 * @file multigrid_solver_vcycle.cpp
 * @brief V-Cycle multigrid solver implementation.
 *
 * Contains all solve-phase logic for the TetMultigridSolver class:
 *   - Galerkin coarse operator construction (R*A*P)
 *   - Chebyshev-accelerated damped Jacobi smoothing
 *   - V-cycle hierarchy build and recursive solve
 *
 * The hierarchy *construction* (sampling, clustering, prolongation assembly)
 * lives in multigrid_solver.cpp.  This file only consumes the prolongation
 * operators produced there.
 */

#ifdef _MSC_VER
#define _USE_MATH_DEFINES  // expose M_PI from <cmath> on MSVC
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "multigrid_solver.h"

// =================================================================================================
// Coarse Operator Construction
// =================================================================================================

// Static helper for RAP product
Eigen::SparseMatrix<double>
GravoMG::TetMultigridSolver::computeProductRAP(
    const Eigen::SparseMatrix<double> &R, const Eigen::SparseMatrix<double> &A,
    const Eigen::SparseMatrix<double> &P) {
  // Optimized Sparse Product: R * A * P
  // Eigen's sparse product is already decent, but for truly optimal performance
  // one might want to use MKL or similar.
  // For now, we rely on Eigen's implementation which is generally faster than
  // Scipy. Also, Eigen handles the intermediate memory better than Python
  // overhead.

  // Note: Eigen (A*B)*C vs A*(B*C)
  // A is NxN. P is NxM. R is MxN. (M < N)
  // A*P performs M sparse-sparse vector products (columns of P).
  // R*A performs N sparse-sparse vector products (columns of A).
  // Since M << N, (A*P) is much cheaper iteration-wise for ColMajor storage.

  Eigen::SparseMatrix<double> AP = A * P;
  Eigen::SparseMatrix<double> RAP = R * AP;

  // Ensure compressed format
  RAP.makeCompressed();
  return RAP;
}

// =================================================================================================
// MARK: V-Cycle Solver Implementation
// =================================================================================================

bool GravoMG::TetMultigridSolver::buildVCycleHierarchy(
    const Eigen::SparseMatrix<double> &A_fine) {
  if (verbose) {
    std::cout << "[CPP] buildVCycleHierarchy called" << std::endl;
  }

  const auto &prols = allP;
  if (prols.empty()) {
    if (verbose) {
      std::cerr
          << "[CPP] Error: No prolongation operators available. "
             "Call constructProlongationOurs first."
          << std::endl;
    }
    return false;
  }

  // Store prolonged operators for recursive solver
  vcycle_prolongations_ = prols;

  vcycle_num_levels_ = static_cast<int>(prols.size()) + 1;

  if (verbose) {
    std::cout << "[CPP] Building V-cycle hierarchy: " << vcycle_num_levels_
              << " levels" << std::endl;
  }

  // Build operators for each level using Galerkin projection
  vcycle_operators_.clear();
  vcycle_operators_.reserve(vcycle_num_levels_);
  vcycle_operators_.push_back(SparseMatrixCSR(A_fine)); // Level 0 = finest

  Eigen::SparseMatrix<double> A_current = A_fine;
  for (size_t i = 0; i < prols.size(); ++i) {
    const auto &P = prols[i];
    Eigen::SparseMatrix<double> A_next =
        computeProductRAP(P.transpose(), A_current, P);
    vcycle_operators_.push_back(SparseMatrixCSR(A_next));
    A_current = A_next;
  }

  // Pre-compute diagonal inverses for Jacobi smoother (all levels except
  // coarsest) Pre-compute diagonal inverses for Jacobi smoother (all levels
  // except coarsest)
  vcycle_diag_inv_.clear();
  vcycle_diag_inv_.reserve(vcycle_num_levels_ - 1);
  for (int level = 0; level < vcycle_num_levels_ - 1; ++level) {
    const auto &A_level = vcycle_operators_[level];
    Eigen::VectorXd d_inv(A_level.rows());

    // Optimize diagonal extraction for RowMajor
    // coeff(i,i) is        // Optimized diagonal extraction for RowMajor
    d_inv.setZero();
    std::atomic<int> zero_diags(0);
#pragma omp parallel for
    for (int i = 0; i < A_level.rows(); ++i) {
      bool found = false;
      for (SparseMatrixCSR::InnerIterator it(A_level, i); it; ++it) {
        if (it.index() == i) {
          double diag = it.value();
          if (std::abs(diag) > 1e-14) {
            d_inv(i) = (1.0 / diag);
          } else {
            zero_diags++;
          }
          found = true;
          break;
        }
      }
      if (!found)
        zero_diags++;
    }

    if (zero_diags > 0 && verbose) {
      std::cout << "[CPP] Warning: Level " << level << " has " << zero_diags
                << " zero/missing diagonals!" << std::endl;
    }

    vcycle_diag_inv_.push_back(d_inv);
  }

  // Pre-compute Chebyshev omega coefficients per level for accelerated Jacobi
  // smoothing.  For each level we estimate lambda_max of D^{-1}A via power
  // iteration, then use the standard Chebyshev formula
  //   omega[k] = 2 / (lambda_max + lambda_min - (lambda_max - lambda_min) cos(theta_k)),
  //   theta_k  = pi (2k + 1) / (2 * chebyshev_degree_)
  // with a conservative lambda_min = lambda_max / 10.
  //
  // Stability note: if lambda_max is *under-estimated*, eigenmodes between
  // the estimate and the true spectral radius are amplified every V-cycle and
  // the iteration diverges.  Very uniform tet meshes (TetGen `q1.1` outputs)
  // have closely clustered eigenvalues, so plain power iteration converges
  // slowly there.  We therefore use 50 power iterations and a 20% safety
  // margin, which is robust on all meshes we have tested without changing
  // the asymptotic V-cycle rate measurably.
  constexpr int POWER_ITERATIONS = 50;
  constexpr double SPECTRAL_SAFETY = 1.20;
  vcycle_chebyshev_omega_.clear();
  vcycle_chebyshev_omega_.reserve(vcycle_num_levels_ - 1);

  for (int level = 0; level < vcycle_num_levels_ - 1; ++level) {
    const auto &A_level = vcycle_operators_[level];
    const auto &d_inv = vcycle_diag_inv_[level];
    int n = static_cast<int>(A_level.rows());

    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    double lambda_max = 1.0;
    for (int iter = 0; iter < POWER_ITERATIONS; ++iter) {
      Eigen::VectorXd w = d_inv.cwiseProduct(A_level * v);
      lambda_max = w.norm();
      if (lambda_max > 1e-10) {
        v = w / lambda_max;
      }
    }
    lambda_max *= SPECTRAL_SAFETY;
    double lambda_min = lambda_max / 10.0;

    std::vector<double> omegas(chebyshev_degree_);
    for (int k = 0; k < chebyshev_degree_; ++k) {
      double theta = M_PI * (2.0 * k + 1.0) / (2.0 * chebyshev_degree_);
      omegas[k] = 2.0 / (lambda_max + lambda_min -
                         (lambda_max - lambda_min) * cos(theta));
    }
    vcycle_chebyshev_omega_.push_back(omegas);

    if (verbose) {
      std::cout << "[CPP] Level " << level
                << ": spectral radius approx " << lambda_max
                << ", Chebyshev omega = [" << omegas[0];
      for (size_t i = 1; i < omegas.size(); ++i)
        std::cout << ", " << omegas[i];
      std::cout << "]" << std::endl;
    }
  }

  // Factorize coarsest level matrix
  const auto &A_coarse_csr = vcycle_operators_.back();

  // Explicit conversion to ColMajor for factorization compatibility/robustness
  Eigen::SparseMatrix<double> A_coarse_col = A_coarse_csr;

  if (verbose) {
    std::cout << "[CPP] Factorizing coarsest level (" << A_coarse_col.rows()
              << " x " << A_coarse_col.cols()
              << ", nnz=" << A_coarse_col.nonZeros() << ")..." << std::endl;
  }

  if (use_dense_coarse_solver_) {
    // Dense LDLT: convert sparse to dense, factorize with LAPACK-backed Eigen dense LDLT.
    // Eliminates sparse format overhead for small coarsest-level matrices.
    if (verbose) std::cout << "[CPP] Using DENSE LDLT for coarse solver." << std::endl;

    dense_coarse_matrix_ = Eigen::MatrixXd(A_coarse_col);
    dense_coarse_solver_ = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>(dense_coarse_matrix_);

    if (dense_coarse_solver_->info() == Eigen::Success) {
      coarse_solver_valid_ = true;
      coarse_solver_name_ = "DenseLDLT";
      coarse_solver_.reset();  // release sparse solver memory
      if (verbose) {
        std::cout << "[CPP] Coarse solver factorization successful "
                     "(DenseLDLT, " << A_coarse_col.rows() << "x" << A_coarse_col.cols() << ")"
                  << std::endl;
      }
    } else {
      throw std::runtime_error("[CPP] FATAL: Dense LDLT factorization "
                               "failed. Matrix may not be SPD.");
    }
  } else {
    // Sparse SimplicialLDLT (default)
    if (verbose) std::cout << "[CPP] Using sparse SimplicialLDLT for coarse solver." << std::endl;

    coarse_solver_ = std::make_unique<
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
    coarse_solver_->analyzePattern(A_coarse_col);
    if (verbose) std::cout << "[CPP] analyzePattern done." << std::endl;
    coarse_solver_->factorize(A_coarse_col);
    if (verbose) std::cout << "[CPP] factorize done." << std::endl;

    if (coarse_solver_->info() == Eigen::Success) {
      coarse_solver_valid_ = true;
      coarse_solver_name_ = "SimplicialLDLT";
      dense_coarse_solver_.reset();  // release dense solver memory
      if (verbose) {
        std::cout << "[CPP] Coarse solver factorization successful "
                     "(SimplicialLDLT)"
                  << std::endl;
      }
    } else {
      throw std::runtime_error("[CPP] FATAL: SimplicialLDLT factorization "
                               "failed. Matrix may not be SPD.");
    }
  }

  vcycle_hierarchy_built_ = true;
  return true;
}

GravoMG::TetMultigridSolver::VCycleSolveResult
GravoMG::TetMultigridSolver::solveVCycle(
    const Eigen::VectorXd &b, const Eigen::VectorXd &x0, int max_cycles,
    int pre_sweeps, int post_sweeps, double tol, double timeout_ms,
    double jacobi_omega, bool collect_timing,
    const Eigen::VectorXd &mass_diag_inv) {
  VCycleSolveResult result;
  this->jacobi_omega_ = jacobi_omega;
  result.x = x0;
  result.num_cycles = 0;
  result.converged = false;
  result.timed_out = false;
  result.total_time_ms = 0.0;
  result.coarse_solver_name = coarse_solver_name_;

  if (!vcycle_hierarchy_built_) {
    std::cerr << "[CPP] Error: V-cycle hierarchy not built. Call "
                 "buildVCycleHierarchy first."
              << std::endl;
    return result;
  }

  const auto &A0 = vcycle_operators_[0];
  if (b.size() != A0.rows() || x0.size() != A0.rows()) {
    std::cerr << "[CPP] Error: Dimension mismatch in solveVCycle. "
              << "A0: " << A0.rows() << "x" << A0.cols() << ", b: " << b.size()
              << ", x0: " << x0.size() << std::endl;
    return result;
  }

  // Determine whether to use weighted (inv-mass) norm for residuals.
  // When mass_diag_inv is provided (non-empty, same size as b), compute
  //   ||r||_{M^{-1}} = sqrt(r^T diag(mass_diag_inv) r)
  // Otherwise fall back to the standard L2 norm.
  const bool use_weighted_norm = (mass_diag_inv.size() == b.size());

  // Helper lambda: compute the chosen norm of a residual vector
  auto residual_norm = [&](const Eigen::VectorXd &r) -> double {
    if (use_weighted_norm) {
      // ||r||_{M^{-1}} = sqrt( sum_i  r_i^2 * mass_diag_inv_i )
      return std::sqrt(std::max(0.0, r.cwiseAbs2().dot(mass_diag_inv)));
    }
    return r.norm(); // L2
  };

  // Initial residual norm
  double r0_norm = residual_norm(b - A0 * result.x);
  result.residual_history.push_back(r0_norm);

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    auto cycle_start_time = std::chrono::high_resolution_clock::now();

    // Perform one V-cycle
    result.x = vcycleRecursive(0, result.x, b, pre_sweeps, post_sweeps,
                               collect_timing, result);
    result.num_cycles++;

    // Compute residual
    double r_norm = residual_norm(b - A0 * result.x);
    result.residual_history.push_back(r_norm);

    auto cycle_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cycle_elapsed =
      cycle_end_time - cycle_start_time;
    result.cycle_time_ms_history.push_back(cycle_elapsed.count());

    // Check convergence
    if (tol > 0 && r0_norm > 1e-15) {
      double rel_residual = r_norm / r0_norm;
      if (rel_residual < tol) {
        result.converged = true;
        break;
      }
    }

    // Check timeout
    if (timeout_ms > 0) {
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = now - start_time;
      if (elapsed.count() >= timeout_ms) {
        result.timed_out = true;
        break;
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> total_elapsed =
      end_time - start_time;
  result.total_time_ms = total_elapsed.count();

  return result;
}

// Apply `n_sweeps` of Chebyshev-accelerated, OpenMP-parallel damped Jacobi.
// Sweeps 0..min(n_sweeps, chebyshev_degree_)-1 use the precomputed omega[s];
// any remaining sweeps fall back to the fixed jacobi_omega_.
//
//   x_new[i] = x[i] + omega * D^{-1}[i] * (b[i] - (A x)[i])
static inline void smoothChebyshevJacobi(
    const GravoMG::TetMultigridSolver::SparseMatrixCSR &A,
    const Eigen::VectorXd &b, Eigen::VectorXd &x_level,
    const Eigen::VectorXd &d_inv, const std::vector<double> &cheb_omega,
    double jacobi_omega, int n_sweeps) {
  int rows = static_cast<int>(A.rows());
  Eigen::VectorXd x_buf1 = x_level;
  Eigen::VectorXd x_buf2(rows);
  Eigen::VectorXd *x_read = &x_buf1;
  Eigen::VectorXd *x_write = &x_buf2;

  for (int s = 0; s < n_sweeps; ++s) {
    double omega = (s < static_cast<int>(cheb_omega.size()))
                       ? cheb_omega[s]
                       : jacobi_omega;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
      double sigma = 0.0;
      for (GravoMG::TetMultigridSolver::SparseMatrixCSR::InnerIterator it(A, i);
           it; ++it) {
        sigma += it.value() * (*x_read)[it.index()];
      }
      double r = b[i] - sigma;
      (*x_write)[i] = (*x_read)[i] + omega * d_inv[i] * r;
    }
    std::swap(x_read, x_write);
  }
  x_level = *x_read;
}

Eigen::VectorXd GravoMG::TetMultigridSolver::vcycleRecursive(
    int level, const Eigen::VectorXd &x, const Eigen::VectorXd &b,
    int pre_sweeps, int post_sweeps, bool collect_timing,
    VCycleSolveResult &timing) {
  const auto &A_level = vcycle_operators_[level];

  // Base case: coarsest level - use direct solver
  if (level == vcycle_num_levels_ - 1) {
    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd x_coarse;
    if (use_dense_coarse_solver_ && dense_coarse_solver_) {
      x_coarse = dense_coarse_solver_->solve(b);
    } else {
      x_coarse = coarse_solver_->solve(b);
    }
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.coarse_solve_time_ms += elapsed.count();
    }
    return x_coarse;
  }

  Eigen::VectorXd x_level = x;
  const auto &P = vcycle_prolongations_[level];
  const auto &d_inv = vcycle_diag_inv_[level];
  const auto &cheb_omega = vcycle_chebyshev_omega_[level];

  // Pre-smoothing
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    smoothChebyshevJacobi(A_level, b, x_level, d_inv, cheb_omega,
                          jacobi_omega_, pre_sweeps);
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.sweep_time_ms += elapsed.count();
    }
  }

  // Compute residual
  Eigen::VectorXd r_level;
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    r_level = b - A_level * x_level;
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.residual_time_ms += elapsed.count();
    }
  }

  // Restrict residual to coarser level
  Eigen::VectorXd b_coarse;
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    b_coarse = P.transpose() * r_level;
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.restriction_time_ms += elapsed.count();
    }
  }

  // Recurse to coarser level
  Eigen::VectorXd x_coarse_init =
      Eigen::VectorXd::Zero(vcycle_operators_[level + 1].rows());
  Eigen::VectorXd e_coarse =
      vcycleRecursive(level + 1, x_coarse_init, b_coarse, pre_sweeps,
                      post_sweeps, collect_timing, timing);

  // Prolongate correction and add to solution
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    x_level = x_level + P * e_coarse;
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.prolongation_time_ms += elapsed.count();
    }
  }

  // Post-smoothing
  {
    auto t_start = std::chrono::high_resolution_clock::now();
    smoothChebyshevJacobi(A_level, b, x_level, d_inv, cheb_omega,
                          jacobi_omega_, post_sweeps);
    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.sweep_time_ms += elapsed.count();
    }
  }

  return x_level;
}
