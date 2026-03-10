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

m = 300
max_k = 100
A = randn(m, m)

V = zeros(m,max_k)
W = zeros(m,max_k)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
gamma_up = zeros(max_k - 1)
T_mat = Tridiagonal(beta_sub, alpha, gamma_up)

n = tridiagonal_biorthogonalization!(A, V, W, T_mat, maxiter = max_k)

println("Bi-orthonormality (||W'v - I||): ", norm(view(W, :, 1:n)' * view(V, :, 1:n) - I))
# 20: yes
# 50: not quite !
# 100: garbage