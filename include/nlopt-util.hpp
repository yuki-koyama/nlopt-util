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

#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <nlopt.hpp>

namespace nloptutil
{
    using obj_func = std::function<double(const std::vector<double>&, std::vector<double>&, void*)>;
    
    inline Eigen::VectorXd compute(const Eigen::VectorXd& x_initial,
                                   const Eigen::VectorXd& upper,
                                   const Eigen::VectorXd& lower,
                                   const obj_func&        objective_function,
                                   void*            data               = nullptr,
                                   nlopt::algorithm algorithm          = nlopt::LD_TNEWTON,
                                   int              max_evaluations    = 1000,
                                   double           relative_func_tol  = 1e-06,
                                   double           relative_param_tol = 1e-06,
                                   bool             verbose            = true,
                                   double           initial_step_scale = 1.0
                                   )
    {
        const unsigned M = static_cast<unsigned>(x_initial.rows());
        
        const std::vector<double> l(lower.data(), lower.data() + lower.rows());
        const std::vector<double> u(upper.data(), upper.data() + upper.rows());
       
        nlopt::opt solver(algorithm, M);
        solver.set_lower_bounds(l);
        solver.set_upper_bounds(u);
        solver.set_maxeval(max_evaluations);
        solver.set_min_objective(*objective_function.target<nlopt::vfunc>(), data);
        solver.set_ftol_rel(relative_func_tol);
        solver.set_xtol_rel(relative_param_tol);
        
        std::vector<double> x_star(x_initial.data(), x_initial.data() + x_initial.rows());

        // Record the cost value for the initial solution
        double initial_cost_value;
        if (verbose)
        {
            std::vector<double> dummy(M);
            initial_cost_value = objective_function(x_star, dummy, data);
        }
        
        // Scale the initial step size (only for derivative-free algorithms such as nlopt::LN_COBYLA)
        std::vector<double> step(M, 0.0);
        solver.get_initial_step(x_star, step);
        for (auto& d : step) d *= initial_step_scale;
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
            std::cerr << e.what() << std::endl;
            assert(false);
        }
        catch (std::runtime_error e)
        {
            std::cerr << e.what() << std::endl;
            assert(false);
        }

        // Stop timing measurement
        const auto t_end = std::chrono::system_clock::now();

        // Show statistics if "verbose" is set as true
        if (verbose)
        {
            const double t_elapsed_in_sec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
            
            std::cout << "---- nlopt-util ----" << std::endl;
            std::cout << "Dimensions     : " << M << std::endl;
            std::cout << "Function value : " << initial_cost_value << " => " << final_cost_value << std::endl;
            std::cout << "Elapsed time   : " << t_elapsed_in_sec << " [s]" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        
        return Eigen::Map<Eigen::VectorXd>(&x_star[0], M);
    }
}

#endif // NLOPTUTIL_HPP
