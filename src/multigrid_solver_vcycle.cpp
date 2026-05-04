/**
 * @file multigrid_solver_vcycle.cpp
 * @brief V-Cycle multigrid solver implementation.
 *
 * Contains all solve-phase logic for the TetMultigridSolver class:
 *   - Galerkin coarse operator construction (R*A*P)
 *   - Gauss-Seidel and Jacobi smoothers
 *   - V-cycle hierarchy build and recursive solve
 *   - External prolongation support
 *
 * The hierarchy *construction* (sampling, clustering, prolongation assembly)
 * lives in multigrid_solver.cpp.  This file only consumes the prolongation
 * operators produced there.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#define _USE_MATH_DEFINES
#include "multigrid_solver.h"

// =================================================================================================
// Coarse Operator Construction
// =================================================================================================

std::vector<Eigen::SparseMatrix<double>>
GravoMG::TetMultigridSolver::computeCoarseOperators(
    const Eigen::SparseMatrix<double> &A_fine, const std::string &method) {
  std::vector<Eigen::SparseMatrix<double>> coarse_ops;
  const auto &prols = (method == "shi06") ? this->allP_shi06 : this->allP;

  if (prols.empty()) {
    return coarse_ops;
  }

  Eigen::SparseMatrix<double> A_current = A_fine;

  if (this->verbose) {
    std::cout << "[CPP] Computing coarse operators (" << method << ") for "
              << prols.size() << " levels..." << std::endl;
  }

  for (size_t i = 0; i < prols.size(); ++i) {
    const auto &P = prols[i];

    if (this->verbose) {
      std::cout << "  [CPP] Level " << i << " Shapes: A(" << A_current.rows()
                << "x" << A_current.cols() << ", nnz=" << A_current.nonZeros()
                << "), P(" << P.rows() << "x" << P.cols()
                << ", nnz=" << P.nonZeros() << ")" << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Use static helper
    Eigen::SparseMatrix<double> A_next =
        computeProductRAP(P.transpose(), A_current, P);

    auto end = std::chrono::high_resolution_clock::now();

    if (this->verbose) {
      std::chrono::duration<double, std::milli> t = end - start;
      std::cout << "  [CPP] Timing: RAP = " << t.count()
                << " ms. Next A nnz: " << A_next.nonZeros() << std::endl;
    }

    coarse_ops.push_back(A_next);
    A_current = A_next;
  }

  return coarse_ops;
}

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
// Gauss-Seidel Smoothers
// =================================================================================================

Eigen::VectorXd GravoMG::TetMultigridSolver::computeGaussSeidel(
    const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
    const Eigen::VectorXd &x, const int iterations, const bool sweep_backward,
    const bool symmetric) {
  // x is the initial guess
  Eigen::VectorXd x_curr = x;
  int n = A.rows();

  // Check dimensions
  if (A.cols() != n || b.size() != n || x.size() != n) {
    std::cerr << "[CPP] Error: Dimension mismatch in computeGaussSeidel"
              << std::endl;
    return x;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    // Forward Sweep
    if (!sweep_backward || symmetric) {
      for (int i = 0; i < n; ++i) {
        double sigma = 0.0;
        double diag = 0.0;

        // Iterate over non-zero elements of row i
        // Eigen sparse matrix (CSR) outer iterator gives rows, inner gives cols
        // But standard inner iterator usage:
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
          if (it.index() == i) {
            diag = it.value();
          } else {
            sigma += it.value() * x_curr[it.index()];
          }
        }

        if (std::abs(diag) > 1e-14) {
          x_curr[i] = (b[i] - sigma) / diag;
        }
      }
    }

    // Backward Sweep (for Symmetric GS)
    if (sweep_backward || symmetric) {
      for (int i = n - 1; i >= 0; --i) {
        double sigma = 0.0;
        double diag = 0.0;

        for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
          if (it.index() == i) {
            diag = it.value();
          } else {
            sigma += it.value() * x_curr[it.index()];
          }
        }

        if (std::abs(diag) > 1e-14) {
          x_curr[i] = (b[i] - sigma) / diag;
        }
      }
    }
  }

  return x_curr;
}

Eigen::VectorXd GravoMG::TetMultigridSolver::computeGaussSeidel(
    const SparseMatrixCSR &A, const Eigen::VectorXd &b,
    const Eigen::VectorXd &x, const int iterations, const bool sweep_backward,
    const bool symmetric) {
  // x is the initial guess
  Eigen::VectorXd x_curr = x;
  int n = static_cast<int>(A.rows());

  // Check dimensions
  if (A.cols() != n || b.size() != n || x.size() != n) {
    std::cerr << "[CPP] Error: Dimension mismatch in computeGaussSeidel (CSR)"
              << std::endl;
    return x;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    // Forward Sweep
    if (!sweep_backward || symmetric) {
      for (int i = 0; i < n; ++i) {
        double sigma = 0.0;
        double diag = 0.0;

        // Iterate over non-zero elements of row i
        // For RowMajor, InnerIterator iterates over the row naturally
        for (SparseMatrixCSR::InnerIterator it(A, i); it; ++it) {
          if (it.index() == i) {
            diag = it.value();
          } else {
            sigma += it.value() * x_curr[it.index()];
          }
        }

        if (std::abs(diag) > 1e-14) {
          x_curr[i] = (b[i] - sigma) / diag;
        }
      }
    }

    // Backward Sweep (for Symmetric GS)
    if (sweep_backward || symmetric) {
      for (int i = n - 1; i >= 0; --i) {
        double sigma = 0.0;
        double diag = 0.0;

        for (SparseMatrixCSR::InnerIterator it(A, i); it; ++it) {
          if (it.index() == i) {
            diag = it.value();
          } else {
            sigma += it.value() * x_curr[it.index()];
          }
        }

        if (std::abs(diag) > 1e-14) {
          x_curr[i] = (b[i] - sigma) / diag;
        }
      }
    }
  }

  return x_curr;
}

// =================================================================================================
// MARK: V-Cycle Solver Implementation
// =================================================================================================

Eigen::VectorXd GravoMG::TetMultigridSolver::computeJacobi(
    const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
    const Eigen::VectorXd &x, int iterations, double omega) {
  // Damped Jacobi: x_new = x + omega * D^{-1} * (b - A*x)
  // Pre-compute D^{-1} for efficiency
  Eigen::VectorXd d_inv(A.rows());
  for (int i = 0; i < A.rows(); ++i) {
    double diag = A.coeff(i, i);
    d_inv(i) = (std::abs(diag) > 1e-14) ? (1.0 / diag) : 0.0;
  }

  Eigen::VectorXd x_curr = x;
  for (int iter = 0; iter < iterations; ++iter) {
    // r = b - A*x
    Eigen::VectorXd r = b - A * x_curr;
    // x = x + omega * D^{-1} * r
    x_curr = x_curr + omega * (d_inv.cwiseProduct(r));
  }

  return x_curr;
}

bool GravoMG::TetMultigridSolver::buildVCycleHierarchy(
    const Eigen::SparseMatrix<double> &A_fine, const std::string &method) {
  if (verbose) {
    std::cout << "[CPP] buildVCycleHierarchy called for method: " << method
              << std::endl;
  }

  // Select prolongation operators: external (if set), or internal based on
  // method
  const std::vector<Eigen::SparseMatrix<double>> *prols_ptr = nullptr;

  if (use_external_prolongations_ && !external_prolongations_.empty()) {
    prols_ptr = &external_prolongations_;
    if (verbose) {
      std::cout << "[CPP] Using EXTERNAL prolongations ("
                << external_prolongations_.size() << " levels)" << std::endl;
    }
  } else if (method == "shi06") {
    prols_ptr = &allP_shi06;
  } else {
    prols_ptr = &allP;
  }

  const auto &prols = *prols_ptr;

  if (prols.empty()) {
    if (verbose) {
      std::cerr
          << "[CPP] Error: No prolongation operators available for method: "
          << method << std::endl;
    }
    return false;
  }

  // Store prolonged operators for recursive solver
  vcycle_prolongations_ = prols;

  vcycle_num_levels_ = static_cast<int>(prols.size()) + 1;

  if (verbose) {
    std::cout << "[CPP] Building V-cycle hierarchy (" << method
              << "): " << vcycle_num_levels_ << " levels" << std::endl;
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

  // Pre-compute spectral radius estimates and Chebyshev coefficients for each
  // level This enables Chebyshev-accelerated Jacobi smoothing
  vcycle_spectral_radius_.clear();
  vcycle_spectral_radius_.reserve(vcycle_num_levels_ - 1);
  vcycle_chebyshev_omega_.clear();
  vcycle_chebyshev_omega_.reserve(vcycle_num_levels_ - 1);

  for (int level = 0; level < vcycle_num_levels_ - 1; ++level) {
    const auto &A_level = vcycle_operators_[level];
    const auto &d_inv = vcycle_diag_inv_[level];
    int n = static_cast<int>(A_level.rows());

    // Power iteration to estimate spectral radius of D^{-1}A
    // Use 10 iterations for a quick estimate (accurate enough for Chebyshev)
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    double lambda_max = 1.0;

    for (int iter = 0; iter < 10; ++iter) {
      // w = D^{-1} * A * v
      Eigen::VectorXd Av = A_level * v;
      Eigen::VectorXd w = d_inv.cwiseProduct(Av);
      lambda_max = w.norm();
      if (lambda_max > 1e-10) {
        v = w / lambda_max;
      }
    }

    // Safety margin: slightly overestimate to ensure stability
    lambda_max *= 1.05;

    vcycle_spectral_radius_.push_back(lambda_max);

    // Compute Chebyshev omega coefficients for this level
    // For SPD matrices, eigenvalues are in [lambda_min, lambda_max]
    // Conservative: lambda_min = lambda_max / 10 (standard multigrid
    // assumption)
    double lambda_min = lambda_max / 10.0;

    // Pre-compute for sweeps 0..chebyshev_degree_-1
    std::vector<double> omegas(chebyshev_degree_);
    for (int k = 0; k < chebyshev_degree_; ++k) {
      double theta = M_PI * (2.0 * k + 1.0) / (2.0 * chebyshev_degree_);
      omegas[k] = 2.0 / (lambda_max + lambda_min -
                         (lambda_max - lambda_min) * cos(theta));
    }
    vcycle_chebyshev_omega_.push_back(omegas);

    if (verbose) {
      std::cout << "[CPP] Level " << level << ": spectral radius ≈ "
                << lambda_max << ", Chebyshev ω = [" << omegas[0];
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
    SmootherType smoother, double jacobi_omega, bool collect_timing,
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
                               smoother, collect_timing, result);
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

Eigen::VectorXd GravoMG::TetMultigridSolver::vcycleRecursive(
    int level, const Eigen::VectorXd &x, const Eigen::VectorXd &b,
    int pre_sweeps, int post_sweeps, SmootherType smoother, bool collect_timing,
    VCycleSolveResult &timing) {
  const auto &A_level = vcycle_operators_[level];
  const auto &prols =
      vcycle_prolongations_; // Use prolongation operators from hierarchy build

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

  // Debug print for omega (only once per solve/level to avoid spam, or relying
  // on verbose flag?) if (verbose && level == 0) { ... } // verbose not
  // accessible easily here without passing it or member

  Eigen::VectorXd x_level = x;
  const auto &P = prols[level];
  const auto &d_inv = vcycle_diag_inv_[level];

  // Pre-smoothing
  {
    auto t_start = std::chrono::high_resolution_clock::now();

    if (smoother == SmootherType::JACOBI) {
      // Optimized OpenMP Parallel Chebyshev-accelerated Jacobi with pointer
      // swap
      int rows = static_cast<int>(A_level.rows());
      Eigen::VectorXd x_buf1 = x_level;
      Eigen::VectorXd x_buf2(rows);
      Eigen::VectorXd *x_read = &x_buf1;
      Eigen::VectorXd *x_write = &x_buf2;

      // Get pre-computed Chebyshev omega values for this level
      const auto &cheb_omega = vcycle_chebyshev_omega_[level];

      for (int s = 0; s < pre_sweeps; ++s) {
        // Use Chebyshev omega if available, else fall back to fixed omega
        double omega = (s < static_cast<int>(cheb_omega.size()))
                           ? cheb_omega[s]
                           : jacobi_omega_;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
          double sigma = 0.0;
          // For RowMajor, InnerIterator iterates over the row 'i'
          for (SparseMatrixCSR::InnerIterator it(A_level, i); it; ++it) {
            sigma += it.value() * (*x_read)[it.index()];
          }
          double r = b[i] - sigma;
          (*x_write)[i] = (*x_read)[i] + omega * d_inv[i] * r;
        }
        std::swap(x_read, x_write); // Pointer swap instead of vector copy
      }
      x_level = *x_read; // Copy final result (only once)
    } else {
      // Gauss-Seidel
      x_level =
          computeGaussSeidel(A_level, b, x_level, pre_sweeps, false, false);
    }

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
                      post_sweeps, smoother, collect_timing, timing);

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

    if (smoother == SmootherType::JACOBI) {
      // Optimized OpenMP Parallel Chebyshev-accelerated Jacobi with pointer
      // swap
      int rows = static_cast<int>(A_level.rows());
      Eigen::VectorXd x_buf1 = x_level;
      Eigen::VectorXd x_buf2(rows);
      Eigen::VectorXd *x_read = &x_buf1;
      Eigen::VectorXd *x_write = &x_buf2;

      // Get pre-computed Chebyshev omega values for this level
      const auto &cheb_omega = vcycle_chebyshev_omega_[level];

      for (int s = 0; s < post_sweeps; ++s) {
        // Use Chebyshev omega if available, else fall back to fixed omega
        double omega = (s < static_cast<int>(cheb_omega.size()))
                           ? cheb_omega[s]
                           : jacobi_omega_;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
          double sigma = 0.0;
          for (SparseMatrixCSR::InnerIterator it(A_level, i); it; ++it) {
            sigma += it.value() * (*x_read)[it.index()];
          }
          double r = b[i] - sigma;
          (*x_write)[i] = (*x_read)[i] + omega * d_inv[i] * r;
        }
        std::swap(x_read, x_write); // Pointer swap instead of vector copy
      }
      x_level = *x_read; // Copy final result (only once)
    } else {
      x_level =
          computeGaussSeidel(A_level, b, x_level, post_sweeps, false, false);
    }

    if (collect_timing) {
      auto t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
      timing.sweep_time_ms += elapsed.count();
    }
  }

  return x_level;
}

// Set external prolongation operators (for unified solver support)
void GravoMG::TetMultigridSolver::setExternalProlongations(
    const std::vector<Eigen::SparseMatrix<double>> &prolongations) {
  external_prolongations_ = prolongations;
  use_external_prolongations_ = true;

  // Clear any previously built V-cycle hierarchy since P changed
  vcycle_hierarchy_built_ = false;

  if (verbose) {
    std::cout << "[CPP] setExternalProlongations: Received "
              << prolongations.size() << " prolongation matrices" << std::endl;
    if (!prolongations.empty()) {
      std::cout << "[CPP]   P[0] shape: " << prolongations[0].rows() << " x "
                << prolongations[0].cols() << std::endl;
    }
  }
}

// Clear external prolongations
void GravoMG::TetMultigridSolver::clearExternalProlongations() {
  external_prolongations_.clear();
  external_prolongations_.shrink_to_fit();
  use_external_prolongations_ = false;

  if (verbose) {
    std::cout
        << "[CPP] clearExternalProlongations: Reverted to internal hierarchy"
        << std::endl;
  }
}

// ============================================================================
// Standalone Direct Solve (free function)
// ============================================================================

namespace {

/// Helper template: run analyzePattern + factorize + solve with 3-phase timing.
template <typename SolverT>
GravoMG::DirectSolveResult solveWithEigenSolver(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const std::string& name) {

  GravoMG::DirectSolveResult result;
  result.solver_name = name;
  result.success = false;

  auto wall_start = std::chrono::high_resolution_clock::now();

  SolverT solver;

  // Phase 1: Symbolic analysis (fill-reducing ordering + symbolic structure)
  auto t0 = std::chrono::high_resolution_clock::now();
  solver.analyzePattern(A);
  auto t1 = std::chrono::high_resolution_clock::now();
  result.analyze_time_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (solver.info() != Eigen::Success) {
    result.error = "analyzePattern failed (info=" +
                   std::to_string(static_cast<int>(solver.info())) + ")";
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(t1 - wall_start).count();
    return result;
  }

  // Phase 2: Numerical factorization
  auto t2 = std::chrono::high_resolution_clock::now();
  solver.factorize(A);
  auto t3 = std::chrono::high_resolution_clock::now();
  result.factorize_time_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count();

  if (solver.info() != Eigen::Success) {
    result.error = "factorize failed (info=" +
                   std::to_string(static_cast<int>(solver.info())) +
                   "). Matrix may not be SPD.";
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(t3 - wall_start).count();
    return result;
  }

  // Phase 3: Forward/back substitution
  auto t4 = std::chrono::high_resolution_clock::now();
  result.x = solver.solve(b);
  auto t5 = std::chrono::high_resolution_clock::now();
  result.solve_time_ms =
      std::chrono::duration<double, std::milli>(t5 - t4).count();

  if (solver.info() != Eigen::Success) {
    result.error = "solve failed (info=" +
                   std::to_string(static_cast<int>(solver.info())) + ")";
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(t5 - wall_start).count();
    return result;
  }

  result.total_time_ms =
      std::chrono::duration<double, std::milli>(t5 - wall_start).count();
  result.success = true;
  return result;
}

} // anonymous namespace

GravoMG::DirectSolveResult GravoMG::solveDirectEigen(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const std::string& solver_type) {

  if (solver_type == "eigen-llt") {
    return solveWithEigenSolver<
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>>(
        A, b, "Eigen SimplicialLLT");
  } else if (solver_type == "eigen-ldlt") {
    return solveWithEigenSolver<
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>(
        A, b, "Eigen SimplicialLDLT");
  } else {
    DirectSolveResult result;
    result.success = false;
    result.error = "Unknown solver_type: '" + solver_type +
                   "'. Supported: 'eigen-llt', 'eigen-ldlt'";
    return result;
  }
}
