/**
 * @file gravotet_binding.cpp
 * @brief Minimal pybind11 bindings for the supplementary GravoTet MG package.
 *
 * The supplement only exposes the functionality required to build the
 * `ours_pro` hierarchy and solve Poisson and biharmonic test problems.
 */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "multigrid_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(gravotet, m) {
    m.doc() = "Minimal GravoTet MG bindings for the SIGGRAPH supplementary package";

    py::class_<GravoMG::TetrahedralMesh>(m, "TetrahedralMesh")
        .def(py::init<>())
        .def_readwrite("vertices", &GravoMG::TetrahedralMesh::vertices)
        .def_readwrite("tetrahedra", &GravoMG::TetrahedralMesh::tetrahedra)
        .def("num_vertices", &GravoMG::TetrahedralMesh::numVertices)
        .def("num_tetrahedra", &GravoMG::TetrahedralMesh::numTetrahedra);

    py::class_<GravoMG::TetMultigridSolver>(m, "MultigridSolver")
        .def(py::init<>())
        .def_static(
            "generate_cube_mesh",
            &GravoMG::TetMultigridSolver::generateCubeMesh,
            py::arg("resolution"),
            "Generate a tetrahedral cube mesh with the requested resolution")
        .def(
            "construct_prolongation_ours",
            &GravoMG::TetMultigridSolver::constructProlongationOurs,
            py::arg("mesh"),
            py::arg("output_dir") = "",
            "Build the boundary-aware `ours_pro` hierarchy")
        .def(
            "compute_coarse_operators",
            &GravoMG::TetMultigridSolver::computeCoarseOperators,
            py::arg("A_fine"),
            py::arg("method") = "ours")
        .def_static(
            "compute_product_rap",
            &GravoMG::TetMultigridSolver::computeProductRAP,
            py::arg("R"), py::arg("A"), py::arg("P"))
        .def(
            "build_vcycle_hierarchy",
            &GravoMG::TetMultigridSolver::buildVCycleHierarchy,
            py::arg("A_fine"),
            py::arg("method") = "ours")
        .def(
            "solve_vcycle",
            [](GravoMG::TetMultigridSolver& solver,
               const Eigen::VectorXd& b,
               const Eigen::VectorXd& x0,
               int max_cycles,
               int pre_sweeps,
               int post_sweeps,
               double tol,
               double timeout_ms,
               const std::string& smoother_str,
               double jacobi_omega,
               bool collect_timing,
               const Eigen::VectorXd& mass_diag_inv) {
                auto smoother = GravoMG::TetMultigridSolver::SmootherType::JACOBI;
                if (smoother_str == "gauss_seidel" || smoother_str == "gs") {
                    smoother = GravoMG::TetMultigridSolver::SmootherType::GAUSS_SEIDEL;
                }

                auto result = solver.solveVCycle(
                    b,
                    x0,
                    max_cycles,
                    pre_sweeps,
                    post_sweeps,
                    tol,
                    timeout_ms,
                    smoother,
                    jacobi_omega,
                    collect_timing,
                    mass_diag_inv);

                py::dict py_result;
                py_result["x"] = result.x;
                py_result["num_cycles"] = result.num_cycles;
                py_result["residual_history"] = result.residual_history;
                py_result["cycle_time_ms_history"] = result.cycle_time_ms_history;
                py_result["converged"] = result.converged;
                py_result["timed_out"] = result.timed_out;
                py_result["total_time_ms"] = result.total_time_ms;
                py_result["coarse_solver_name"] = result.coarse_solver_name;

                py::dict vcycle_timing;
                vcycle_timing["sweep_time_ms"] = result.sweep_time_ms;
                vcycle_timing["restriction_time_ms"] = result.restriction_time_ms;
                vcycle_timing["prolongation_time_ms"] = result.prolongation_time_ms;
                vcycle_timing["coarse_solve_time_ms"] = result.coarse_solve_time_ms;
                vcycle_timing["residual_time_ms"] = result.residual_time_ms;
                vcycle_timing["coarse_solver_name"] = result.coarse_solver_name;
                vcycle_timing["num_cycles"] = result.num_cycles;
                py_result["vcycle_timing"] = vcycle_timing;
                return py_result;
            },
            py::arg("b"),
            py::arg("x0"),
            py::arg("max_cycles") = 100,
            py::arg("pre_sweeps") = 2,
            py::arg("post_sweeps") = 2,
            py::arg("tol") = 1e-8,
            py::arg("timeout_ms") = 0.0,
            py::arg("smoother") = "jacobi",
            py::arg("jacobi_omega") = 0.6667,
            py::arg("collect_timing") = false,
            py::arg("mass_diag_inv") = Eigen::VectorXd())
        .def("num_levels", &GravoMG::TetMultigridSolver::numLevels)
        .def("is_hierarchy_built", &GravoMG::TetMultigridSolver::isHierarchyBuilt)
        .def("is_vcycle_hierarchy_built", &GravoMG::TetMultigridSolver::isVCycleHierarchyBuilt)
        .def("get_coarse_solver_name", &GravoMG::TetMultigridSolver::getCoarseSolverName)
        .def_readwrite("min_verts", &GravoMG::TetMultigridSolver::minVerts)
        .def_readwrite("max_levels", &GravoMG::TetMultigridSolver::maxLevels)
        .def_readwrite("search_radius_factor", &GravoMG::TetMultigridSolver::searchRadiusFactor)
        .def_readwrite("verbose", &GravoMG::TetMultigridSolver::verbose)
        .def_readwrite("feature_preserve", &GravoMG::TetMultigridSolver::featurePreserve)
        .def_readwrite("method", &GravoMG::TetMultigridSolver::method)
        .def_readwrite("use_dense_coarse_solver", &GravoMG::TetMultigridSolver::use_dense_coarse_solver_)
        .def_readonly("all_vertices", &GravoMG::TetMultigridSolver::all_vertices)
        .def_readonly("all_tetrahedra", &GravoMG::TetMultigridSolver::all_tetrahedra)
        .def_readonly("all_P", &GravoMG::TetMultigridSolver::allP)
        .def_readonly("nr_points", &GravoMG::TetMultigridSolver::nr_points);

    m.def("version", []() { return "supplement-0.1.0"; });
}
