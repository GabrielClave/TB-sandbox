using LinearAlgebra
# Experiment 1: Discrete Legendre Polynomials


x = collect(-128:128) / 128        # Discretization of [-1, 1]
A = [x.^0 x.^1 x.^2 x.^3]          # Vandermonde matrix
Q, R = qr(A) |> qr -> (qr.Q, qr.R) # Reduced QR factorization

scale = Q[end, :]                 # Last row of Q
Q = Q * Diagonal(1.0 ./ scale)    # Rescale columns

using Plots
plot(x, Q, legend=false)


# Experiment 2: Classical vs. Modified Gram–Schmidt

using Random
Random.seed!(0)

U = qr(randn(80, 80)).Q          # Random orthogonal matrix
V = qr(randn(80, 80)).Q          # Random orthogonal matrix
S = Diagonal(2.0 .^ (-1:-1:-80)) # Exponentially decaying singular values
A = U * S * V'                   # Matrix with prescribed SVD

include("QR.jl")

QC, RC = clgs(A)   # Classical Gram–Schmidt
QM, RM = mgs(A)    # Modified Gram–Schmidt


j = 1:size(A, 2)
r_cgs = abs.(diag(RC))
r_mgs = abs.(diag(RM))

plot(
    j, r_cgs;
    yscale = :log10,
    marker = :circle,
    label = "Classical GS",
    xlabel = "j",
    ylabel = "|r_{jj}|"
)

plot!(
    j, r_mgs;
    marker = :x,
    label = "Modified GS"
)

