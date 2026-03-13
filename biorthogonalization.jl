using LinearAlgebra

# Tridiagonal Biorthogonalization

# A = V T V^-1 V is not unitary
# A* = W T* W^-1  with W = (V^-1)*
# V*W = I
# AV = VT yields 3-terms recurrence
# A*W = WT*

function tridiagonal_biorthogonalization!(
    A::AbstractMatrix{T}, 
    V::AbstractMatrix{T},
    W::AbstractMatrix{T},
    T_mat::Tridiagonal{T}; 
    maxiter::Int = 50, 
    tol::T = 1e-12
) where {T<:AbstractFloat}
    m, _ = size(A)
    # any starting bi-orthogonal v and w
    v1 = view(V, :, 1)
    w1 = view(W, :, 1)

    v1 .= randn(T, m)
    normalize!(v1)
    w1 .= randn(T, m)
    normalize!(w1)

    w1 ./= dot(w1, v1) # <w1, v1> = 1

    alpha = T_mat.d
    beta = T_mat.dl
    gamma = T_mat.du

    for k in 1:maxiter-1

        v_prev = view(V, :, k)
        w_prev = view(W, :, k)

        v_cur = view(V, :, k+1)
        w_cur = view(W, :, k+1)

        mul!(v_cur, A, v_prev) # A*vn
        mul!(w_cur, A', w_prev) # A'*wn

        alpha[k] = dot(w_prev, v_cur)

        v_cur .-= alpha[k]*v_prev
        w_cur .-= alpha[k]*w_prev

        if k>1
            v_cur .-= gamma[k-1]*view(V, :, k-1)
            w_cur .-= beta[k-1]*view(W, :, k-1)
        end

        if (norm(v_cur) < tol) || (norm(w_cur) < tol) # "lucky" breackdown
            println("Found invariant subspace in $k steps.")
            return k
        end

        # normalization: gamma[k]*beta[k] = w_cur*v_cur

        δ = dot(w_cur,v_cur)
        beta[k] = norm(v_cur) # keep V normalized
        gamma[k] = δ/beta[k] # enforce gamma[k]*beta[k] = δ

        if abs(δ) < tol # serious breackdown
            println("serious breackdown after $k steps.")
            return k
        end

        v_cur ./= beta[k]
        w_cur ./= gamma[k]

    end

    alpha[maxiter] = dot(view(W, :, maxiter), A, view(V, :, maxiter))

    println("hit $maxiter iterations.")
    return(maxiter)
end

struct LanczosWorkspace{T}
    V::Matrix{T}
    W::Matrix{T}
    T_mat::Tridiagonal{T}
end

# A "Constructor"
function LanczosWorkspace(m::Int, max_k::Int, T=Float64)
    V = zeros(T, m, max_k)
    W = zeros(T, m, max_k)
    alpha = zeros(T, max_k)
    beta = zeros(T, max_k - 1)
    gamma = zeros(T, max_k - 1)
    return LanczosWorkspace(V, W, Tridiagonal(beta, alpha, gamma))
end

m = 300
max_k = 100
A = randn(m, m)
lw = LanczosWorkspace(m, max_k)

# n = tridiagonal_biorthogonalization!(A, lw.V, lw.W, lw.T_mat, maxiter = max_k)

# println("Bi-orthonormality (||W'v - I||): ", norm(view(lw.W, :, 1:n)' * view(lw.V, :, 1:n) - I))
# 20: yes
# 50: not quite !
# 100: garbage

# biconjugate gradient
# solve Ax = b
# does not minimize |r_n|
# instead enforces r_n ⟂ K(A*,b)

function bcg(
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    maxiter::Int = 50, 
    tol::T = 1e-12
) where {T<:AbstractFloat}
    m, _ = size(A)
    x = zeros(T, m)
    
    # Initialization: r = b - Ax (assumes x=0)
    r = copy(b)
    s = copy(b) # Shadow residual r*
    p = copy(r)
    q = copy(s) # Shadow search direction p*
    
    Ap = zeros(T, m)
    Atq = zeros(T, m)
    
    rho_prev = dot(s, r)
    residual = norm(r)

    for k in 1:maxiter

        mul!(Ap, A, p)
        mul!(Atq, A', q)
        
        denom = dot(q, Ap) # qnApn
        
        if abs(denom) < tol
            println("serious breackdown after $k steps. residual: $residual")
            return x, k
        end
        
        alpha = rho_prev / denom

        x .+= alpha .* p
        r .-= alpha .* Ap
        s .-= alpha .* Atq

        residual = norm(r)

        if norm(r) < tol
            println("Converged in $k steps.")
            return x, k
        end

        rho_cur = dot(s, r)

        if abs(rho_cur) < tol
            println("inner product is zero at step $k, algortithm can't progress. residual: $residual")
            return x, k
        end
        
        beta = rho_cur / rho_prev
        
        p .= r .+ beta .* p
        q .= s .+ beta .* q
        
        rho_prev = rho_cur
    end

    println("hit $maxiter iterations. residual: $residual")
    return x, maxiter
end

# bad matrix
m = 300
A = randn(m, m)
b = randn(m)

x, n = bcg(A,b)

# good matrix
m = 300
max_k = 100
X = randn(m, m)/sqrt(m)
A = X'*X + 5I + randn(m, m)/sqrt(m)
A[1,1] = 12
b = randn(m)

x, n = bcg(A,b)
# beta = 0 after around 10 steps
# at that point dot(s, r) ≃ norm(r)²
# because the tolerance is e-12, we stop at rnorm(r) ≃ e-6