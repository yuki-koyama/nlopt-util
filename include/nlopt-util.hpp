/*
 MIT License

 Copyright (c) 2018 Yuki Koyama

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef NLOPTUTIL_HPP
#define NLOPTUTIL_HPP

#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <nlopt.hpp>
#include <sstream>
#include <string>

namespace nloptutil
{
    namespace internal
    {
        constexpr double constraint_tol = 1e-10;
    }

    inline Eigen::VectorXd solve(const Eigen::VectorXd&           x_initial,
                                 const Eigen::VectorXd&           upper,
                                 const Eigen::VectorXd&           lower,
                                 const nlopt::vfunc               objective_function,
                                 const std::vector<nlopt::vfunc>& equality_constraints,
                                 const std::vector<nlopt::vfunc>& inequality_constraints,
                                 nlopt::algorithm                 algorithm          = nlopt::LD_TNEWTON,
                                 void*                            data               = nullptr,
                                 bool                             is_maximization    = false,
                                 int                              max_evaluations    = 1000,
                                 double                           relative_func_tol  = 1e-06,
                                 double                           relative_param_tol = 1e-06,
                                 bool                             verbose            = false,
                                 double                           initial_step_scale = 1.0)
    {
        const unsigned M = static_cast<unsigned>(x_initial.rows());

        nlopt::opt solver(algorithm, M);

        if (upper.rows() != 0)
        {
            const std::vector<double> u(upper.data(), upper.data() + upper.rows());
            solver.set_upper_bounds(u);
        }

        if (lower.rows() != 0)
        {
            const std::vector<double> l(lower.data(), lower.data() + lower.rows());
            solver.set_lower_bounds(l);
        }

        solver.set_maxeval(max_evaluations);
        solver.set_ftol_rel(relative_func_tol);
        solver.set_xtol_rel(relative_param_tol);

        if (is_maximization)
        {
            solver.set_max_objective(objective_function, data);
        }
        else
        {
            solver.set_min_objective(objective_function, data);
        }

        for (auto func : equality_constraints)
        {
            solver.add_equality_constraint(func, data, internal::constraint_tol);
        }

        for (auto func : inequality_constraints)
        {
            solver.add_inequality_constraint(func, data, internal::constraint_tol);
        }

        std::vector<double> x_star(x_initial.data(), x_initial.data() + x_initial.rows());

        // Record the cost value for the initial solution
        double initial_cost_value;
        if (verbose)
        {
            std::vector<double> dummy(M);
            initial_cost_value = objective_function(x_star, dummy, data);
        }

        // Scale the initial step size (only for derivative-free algorithms such as nlopt::LN_COBYLA)
        const std::vector<double> step = [&]() {
            std::vector<double> step(M, 0.0);
            solver.get_initial_step(x_star, step);
            for (auto& d : step)
            {
                d *= initial_step_scale;
            }
            return step;
        }();
        solver.set_initial_step(step);

        // Start timing measurement
        const auto t_start = std::chrono::system_clock::now();

        // Run the optimization
        double final_cost_value;
        try
        {
            solver.optimize(x_star, final_cost_value);
        }
        catch (nlopt::roundoff_limited)
        {
            // Ignore roundoff_limited exceptions
        }
        catch (std::invalid_argument e)
        {
            if (verbose)
            {
                std::cerr << e.what() << std::endl;
            }
            assert(false);
        }
        catch (std::runtime_error e)
        {
            if (verbose)
            {
                std::cerr << e.what() << std::endl;
            }
            return x_initial;
        }

        // Stop timing measurement
        const auto t_end = std::chrono::system_clock::now();

        // Show statistics if "verbose" is set as true
        if (verbose)
        {
            const std::string elapsed_time_message = [&]() {
                const auto t_elapsed_in_microsec =
                    std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
                const auto t_elapsed_in_millisec = t_elapsed_in_microsec / 1000.0;
                const auto t_elapsed_in_sec      = t_elapsed_in_millisec / 1000.0;

                std::ostringstream sstream;
                if (t_elapsed_in_sec > 10.0)
                {
                    sstream << std::fixed << std::setprecision(3) << t_elapsed_in_sec << " [s]";
                }
                else if (t_elapsed_in_millisec > 10.0)
                {
                    sstream << std::fixed << std::setprecision(3) << t_elapsed_in_millisec << " [ms]";
                }
                else
                {
                    sstream << t_elapsed_in_microsec << " [us]";
                }
                return sstream.str();
            }();

            std::cout << "---- nlopt-util ----" << std::endl;
            std::cout << "Dimensions     : " << M << std::endl;
            std::cout << "Function value : " << initial_cost_value << " => " << final_cost_value << std::endl;
            std::cout << "Elapsed time   : " << elapsed_time_message << std::endl;
            std::cout << "Function evals : " << solver.get_numevals() << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        return Eigen::Map<Eigen::VectorXd>(&x_star[0], M);
    }

    inline Eigen::VectorXd solve(const Eigen::VectorXd& x_initial,
                                 const Eigen::VectorXd& upper,
                                 const Eigen::VectorXd& lower,
                                 const nlopt::vfunc     objective_function,
                                 nlopt::algorithm       algorithm          = nlopt::LD_TNEWTON,
                                 void*                  data               = nullptr,
                                 bool                   is_maximization    = false,
                                 int                    max_evaluations    = 1000,
                                 double                 relative_func_tol  = 1e-06,
                                 double                 relative_param_tol = 1e-06,
                                 bool                   verbose            = false,
                                 double                 initial_step_scale = 1.0)
    {
        return solve(x_initial,
                     upper,
                     lower,
                     objective_function,
                     {},
                     {},
                     algorithm,
                     data,
                     is_maximization,
                     max_evaluations,
                     relative_func_tol,
                     relative_param_tol,
                     verbose,
                     initial_step_scale);
    }
} // namespace nloptutil

#endif // NLOPTUTIL_HPP
