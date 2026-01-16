using LinearAlgebra, SpecialFunctions, Plots

N = 10                     # number of grid points 
x = range(1, 2; length=N)    

# we have Ac = b
# c is 3 dimentional, the 3 coefficients of the decomposition in the basis exp, sin , Γ
# A is "infinite dimentional" x 3 that we approximate by the finite range of points
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
# coefficients are widly different because we changed the inner product = the geometry

f_approx = A * c

# error
Δx = step(x)
L2_error = sqrt(sum((f .- f_approx).^2) * Δx)

# plots

plot(x, f; label="f(x)", lw=2)
plot!(x, f_approx; label="least squares fit", lw=2)