using LinearAlgebra, SpecialFunctions, Plots, Random, Statistics

n, m = 30, 59
# interpolation nodes
x = range(-1, 1; length=n)
y = range(-1, 1; length=m)

Vx = [x[j]^(k-1) for j in 1:n, k in 1:n]
Vy = [y[i]^(k-1) for i in 1:m, k in 1:n]

# Vx*c = p(x)
# Vy*c = p(y)
# p(y) = Vy.Vx^-1.p(x)
# p(y) = A.p(x)  A = Vy.Vx^-1

A = Vy * (Vx \ I)

# We fix interpolation nodes,
# and study how large the resulting interpolating polynomial can become anywhere in [−1,1].

vals = Float64[]

for n in 2:30
    m = 2n - 1

    x = range(-1, 1; length=n)
    y = range(-1, 1; length=m)

    Vx = [x[j]^(k-1) for j in 1:n, k in 1:n]
    Vy = [y[i]^(k-1) for i in 1:m, k in 1:n]

    A = Vy * (Vx \ I)

    push!(vals, opnorm(A, Inf))
end


plot(1:29, log.(vals);
    xlabel="n",
    ylabel="||A||_∞",
    marker=:o)
# exponential growth

# properties of random matrix

random_matrix(m) = randn(m, m) / sqrt(m)

m = 128
ntrials = 1000

allλ = ComplexF64[]
allnorm = Float64[]

for k in 1:ntrials
    A = random_matrix(m)
    append!(allλ, eigvals(A))
    append!(allnorm, opnorm(A, 2))
end

sum(real.(allλ) .< 1e-15)
# is that even a good criteria ?

scatter(
    real.(allλ),
    imag.(allλ),
    markersize = 2,
    aspect_ratio = :equal,
    xlabel = "Re",
    ylabel = "Im",
    legend = false
)
# lots of zeros, the rest somewhat uniform in the unit circle

histogram(allnorm, bins=30, xlabel="‖A‖₂", ylabel="frequency")
# binomial-ish around 1.95

ratios = [
    maximum(abs.(eigvals(random_matrix(m)))) / opnorm(random_matrix(m))
    for _ in 1:ntrials
]

histogram(ratios, bins=30, xlabel="ρ(A)/‖A‖₂", ylabel="frequency")
# right tail looks bigger, centered around 0.55

σmins = [minimum(svdvals(random_matrix(m))) for _ in 1:ntrials]

thresholds = [2.0^(-k) for k in 1:10]

proportions = [
    mean(σmins .<= τ)
    for τ in thresholds
]

A = [10 1 0; 1 2 1; 0 1 -5]

# Get only the eigenvalues
values = eigvals(A)

# Get both eigenvalues and eigenvectors
factors = eigen(A)
values = factors.values
vectors = factors.vectors

A = [ 5 1 1
    0 10 2
    2 0 -2]

values = eigvals(A)