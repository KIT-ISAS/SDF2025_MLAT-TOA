# Fast Multilateration based on Time of Arrival (TOA) measurements (without taking differences / TDOA)
# Code example for:
# Frisch, Hanebeck, MFI 2025: "Why You Shouldn’t Use TDOA for Multilateration"

# Only once: install packages
# using Pkg; Pkg.add("LinearAlgebra"); Pkg.add("Statistics"); Pkg.add("ForwardDiff"); Pkg.add("LeastSquaresOptim")

using LinearAlgebra: norm
using Statistics: mean
import ForwardDiff
using LeastSquaresOptim

# Ground Truth
"ground truth TTT, [1]"
gt_t = .2
"ground truth position, [3]"
gt_x = [3,1,5]

# Sensors 
"number of sensors, [1]"
N = 100 
"sensor positions, [3 x N]"
s = rand(3,N)*10 

# measurements
"propagation speed, [1]"
c = 1; 
"measurement model, [3 x N] → [N]"
h(x) = 1/c * [norm(x-s[:,i]) for i in 1:N] 
"noise-free measurements, [N]"
gt_ti = gt_t .+ h(gt_x) 
"drawn noise values, [N]"
noise = randn(N) * 1 
"actual, noisy measurements, [N]"
ti = gt_ti + noise 

"""
Objective function, in-place. 
# Arguments
- `θ::Vector` : return values (vector whose square sum Levenberg-Marquardt minimizes), [N]
- `x::Vector` : current state vector, [3]
"""
function θ_TOA_LM!(θ, x)
    # use θ as temporary variable to avoid allocations 
    θ .= 1/c * [norm(x .- s[:,i]) for i in 1:N] # compute: 1/c * ||x-si||; part of Eq. (42), [N]
    "optimally estimated target transmission time"
    t0 = mean(ti .- θ) # compute t0-hat according to Eq. (42), [1]
    θ .+= t0 .- ti # Levenberg-Marquardt objective according to (43), [N]
end

"initial guess"
x0 = rand(3)*10 

# Test run
# θ = rand(N); θ_TOA_LM!(θ, x0)

# Optimize
x = optimize!(LeastSquaresProblem(x=copy(x0), f! = θ_TOA_LM! , output_length=N, autodiff=:forward), LevenbergMarquardt()).minimizer
display(x)

