using LinearAlgebra, SpecialFunctions, Plots
using Printf
include("QR.jl")

N = 10                     # number of grid points 
x = range(1, 2; length=N)    

# we have Ac = b
# c is 3 dimensional, the 3 coefficients of the decomposition in the basis exp, sin , Γ
# A is "infinite dimensional" x 3 that we approximate by the finite range of points
# in each line we have a.exp(xk) + b.sin(xk) + c.Γ(xk) more or less equal to 1/xk

f = 1.0 ./ x

A = hcat(
    exp.(x),
    sin.(x),
    gamma.(x)
)

c = A \ f # coefficients of the projection on F = vect(exp, sin , Γ)
#  -0.10777441520910237
#   0.009228723223275464
#   1.2871460188327204

f_approx = A * c

# error
Δx = step(x)
L2_error = sqrt(sum((f .- f_approx).^2) * Δx)

# plots

plot(x, f; label="f(x)", lw=2)
plot!(x, f_approx; label="least squares fit", lw=2)

N = 100                     # number of grid points 
x = range(0, 1; length=N)[2:end]

f = 1.0 ./ x

A = hcat(
    exp.(x),
    sin.(x),
    gamma.(x)
)

c = A \ f # coefficients of the projection on F = vect(exp, sin , Γ)
#   0.6500286008428717
#  -1.8969039383507265
#   0.998933765190686 almost one, because near 0 we have Γ(x) ∼ 1/x
# coefficients are wildly different because we changed the inner product = the geometry

f_approx = A * c

# error
Δx = step(x)
L2_error = sqrt(sum((f .- f_approx).^2) * Δx)

# plots

plot(x, f; label="f(x)", lw=2)
plot!(x, f_approx; label="least squares fit", lw=2)

# 12 degree polynomial approximation of cos(4t)

m = 50
n = 12
t = range(0.0, 1.0; length=m) |> collect

A = [t[i]^(j-1) for i in 1:m, j in 1:n] # flipped Vandermonde

b = cos.(4 .* t)

# method a: normal equations
# A*Ax = A*b

x_normal = (A' * A) \ (A' * b)

# method b: mgs
# y = QQ*b
# QRx = y
# Rx = Q*b

Q, R = mgs(A)

x_mgs = R \ (Q' * b)

# method c: QR householder

W, R = house(A)
Q = formQ(W)[:,1:12]
R = R[1:12,:]

x_house = R \ (Q' * b)

# method e: Julia solver

x_backslash = A \ b

# method f: SVD
# A = USV*
# P = UU*
# USV*x = UU*b
# x = V S^-1 U*b

U, S, V = svd(A)

x_svd = V * (Diagonal(1 ./ S) * (U' * b))

X = hcat(x_normal, x_mgs, x_house, x_backslash, x_svd)

errors = [norm(X[:,j] - x_svd) / norm(x_svd) for j in 1:size(X,2)]

# normal: horrible
# mgs / house / backslash similar to e-9 digits
# svd = reference