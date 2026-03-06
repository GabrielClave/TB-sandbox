include("QR.jl")
using LinearAlgebra
using Plots
using Random
# Experiment 1: Discrete Legendre Polynomials


x = collect(-128:128) / 128        # Discretization of [-1, 1]
A = [x.^0 x.^1 x.^2 x.^3]          # Vandermonde matrix
Q, R = qr(A) |> qr -> (qr.Q, qr.R) # Reduced QR factorization

scale = Q[end, :]                 # Last row of Q
Q = Q * Diagonal(1.0 ./ scale)    # Rescale columns

plot(x, Q, legend=false)


# Experiment 2: Classical vs. Modified Gram–Schmidt
Random.seed!(0)

U = qr(randn(80, 80)).Q          # Random orthogonal matrix
V = qr(randn(80, 80)).Q          # Random orthogonal matrix
S = Diagonal(2.0 .^ (-1:-1:-80)) # Exponentially decaying singular values
A = U * S * V'                   # Matrix with prescribed SVD


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

# Experiment 3: Numerical Loss of Orthogonality

A = [0.70000  0.70711
     0.70001  0.70711]

Qh = qr(A).Q

norm(Qh' * Qh - I) # e-16

Qh_mgs, _ = mgs(A)
Qh_clgs, _ = clgs(A)

norm(Qh_mgs' * Qh_mgs - I) # e-11
norm(Qh_clgs' * Qh_clgs - I) # e-11
# in 2D modified and Classical are the same

W, _ = house(A)
Q_home_house = formQ(W)

norm(Q_home_house' * Q_home_house - I) # exactly 0 !
# orthogonality is enforced by the reconstruction process of Q (applying the same reflector backward)
# in the library package, orthogonality is numerically enforced: it is only guaranteed to ≈ machine precision