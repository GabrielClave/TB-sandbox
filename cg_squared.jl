using LinearAlgebra
using Plots
using IncompleteLU
using SparseArrays

# conjugate gradient squared

# using the adjoint property we get rid of A' entirely
# the promise is "double progress" per iteration

# the issues are "overshooting" and stability

function cgs(
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    maxiter::Int = 50, 
    tol::T = 1e-12
) where {T<:AbstractFloat}
    m, _ = size(A)
    x = zeros(T, m)
    q = zeros(T, m) # auxiliary q = u - α*A*p
    d = zeros(T, m) # auxiliary d = u + q
    
    residuals = zeros(T, maxiter+1)
    mvp_tmp = zeros(T, m)
    
    # Initialization: r = b - Ax (assumes x=0)
    r = copy(b)
    u = copy(b) # auxiliary u = P(A)S(A)*r₀
    p = copy(b) # P(A)²*r₀
    
    ρ_prev = dot(r, b)
    residuals[1] = norm(r)

    for k in 1:maxiter

        mul!(mvp_tmp, A, p)
        α = ρ_prev / dot(mvp_tmp, b)

        q .= u .- α.*mvp_tmp

        d .= u .+ q # direction update

        mul!(mvp_tmp, A, d)

        x .+= α.*d # solution update
        r .-= α.*mvp_tmp # residual update

        residuals[k+1] = norm(r)
        if residuals[k+1] < tol
            println("Converged in $k steps.")
            return x, k, residuals
        end

        ρ_cur = dot(r, b)

        if abs(ρ_cur) < tol
            println("inner product is zero at step $k, algorithm can't progress. residual: $(residuals[k+1])")
            return x, k, residuals
        end

        β = ρ_cur/ρ_prev

        u .= r .+ β.*q
        p .= u .+ β.*(q .+ β.*p)

        ρ_prev = ρ_cur
    end

    println("hit $maxiter iterations. residual: $(residuals[maxiter])")
    return x, maxiter, residuals

end

# bad matrix
m = 300
A = randn(m, m)
b = randn(m)

x, n, residuals = cgs(A,b)

res_history = residuals[1:n+1]

p = plot(
    0:n, 
    res_history, 
    yaxis=:log, 
    linewidth=2,
    marker=:circle,
    markersize=3,
    title="CGS Convergence",
    xlabel="Iteration (k)",
    ylabel="||r_k|| (Log Scale)",
    label="CGS Residual",
    legend=:topright,
    color=:crimson
)
hline!([1e-12], linestyle=:dash, color=:black, label="Tolerance")

# good matrix
m = 300
max_k = 100
X = randn(m, m)/sqrt(m)
A = X'*X + 5I + randn(m, m)/sqrt(m)
A[1,1] = 12
b = randn(m)

x, n, residuals = cgs(A,b)

res_history = residuals[1:n+1]

p = plot(
    0:n, 
    res_history, 
    yaxis=:log, 
    linewidth=2,
    marker=:circle,
    markersize=3,
    title="CGS Convergence",
    xlabel="Iteration (k)",
    ylabel="||r_k|| (Log Scale)",
    label="CGS Residual",
    legend=:topright,
    color=:crimson
)
hline!([1e-12], linestyle=:dash, color=:black, label="Tolerance")
# we see the residual increasing for some iteration

# include("bcg.jl")
# x, n = bcg(A,b)

# with preconditioner

function cgs(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    M;
    maxiter::Int = 50, 
    tol::T = 1e-12
) where {T<:AbstractFloat}
    m, _ = size(A)
    x = zeros(T, m)
    q = zeros(T, m) # auxiliary q = u - α*A*p
    d = zeros(T, m) # auxiliary d = u + q

    z_tmp = zeros(T, m) # preconditioned p and d
    
    residuals = zeros(T, maxiter+1)
    mvp_tmp = zeros(T, m)
    
    # Initialization: r = b - Ax (assumes x=0)
    r = copy(b)
    u = copy(b) # auxiliary u = P(A)S(A)*r₀
    p = copy(b) # P(A)²*r₀
    
    ρ_prev = dot(r, b)
    residuals[1] = norm(r)

    for k in 1:maxiter

        ldiv!(z_tmp, M, p) # z_tmp = M-1*p
        mul!(mvp_tmp, A, z_tmp)

        α = ρ_prev / dot(mvp_tmp, b)

        q .= u .- α.*mvp_tmp

        d .= u .+ q # direction update

        ldiv!(z_tmp, M, d) # z_tmp = M-1*d
        mul!(mvp_tmp, A, z_tmp)

        x .+= α.*z_tmp # solution update
        r .-= α.*mvp_tmp # residual update

        residuals[k+1] = norm(r)
        if residuals[k+1] < tol
            println("Converged in $k steps.")
            return x, k, residuals
        end

        ρ_cur = dot(r, b)

        if abs(ρ_cur) < tol
            println("inner product is zero at step $k, algorithm can't progress. residual: $(residuals[k+1])")
            return x, k, residuals
        end

        β = ρ_cur/ρ_prev

        u .= r .+ β.*q
        p .= u .+ β.*(q .+ β.*p)

        ρ_prev = ρ_cur
    end

    println("hit $maxiter iterations. residual: $(residuals[maxiter])")
    return x, maxiter, residuals

end

# generate some sparse matrix to try ILU
function generate_example(N)
    n = N^2
    # Standard 1D Laplacian 2nd order finite difference
    D1 = spdiagm(-1 => -ones(N-1), 0 => 2ones(N), 1 => -ones(N-1))
    # 2D Laplacian (Poisson operator)
    A = kron(I(N), D1) + kron(D1, I(N))
    
    # Add a convection term (non-symmetric: -1 on lower diag, +1.1 on upper)
    convection = spdiagm(-1 => -0.5ones(n-1), 1 => 1.5ones(n-1))
    A = A + convection    
    return A
end

N_grid = 20 # 20x20 grid = 400x400 matrix
A_sparse = generate_example(N_grid)
A_sparse[100, 100] = 1e3 # artificially creates one huge eigenvalue
b = randn(size(A_sparse, 1))

# 1. No Precond
x1, k1, res1 = cgs(A_sparse, b; maxiter=200)

# 2. ILU
M_ilu = ilu(A_sparse, τ=0.1)
x2, k2, res2 = cgs(A_sparse, b, M_ilu; maxiter=200)

# 3. Jacobi (Diagonal)
M_jac = Diagonal(A_sparse)
x3, k3, res3 = cgs(A_sparse, b, M_jac; maxiter=200)

p = plot(yaxis=:log, title="CGS on Sparse Convection-Diffusion", xlabel="Iter", ylabel="||r||")
plot!(p, 0:k1, res1[1:k1+1], label="No Precond", color=:red)
plot!(p, 0:k2, res2[1:k2+1], label="ILU (τ=0.1)", color=:green, lw=2)
plot!(p, 0:k3, res3[1:k3+1], label="Jacobi", color=:blue)