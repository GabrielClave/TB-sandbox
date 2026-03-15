using LinearAlgebra

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
        p .= u .+ β.*(q + β.*p)

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

# good matrix
m = 300
max_k = 100
X = randn(m, m)/sqrt(m)
A = X'*X + 5I + randn(m, m)/sqrt(m)
A[1,1] = 12
b = randn(m)

x, n, residuals = cgs(A,b)