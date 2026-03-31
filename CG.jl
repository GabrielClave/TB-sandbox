using LinearAlgebra

# conjugate gradient

# A is SPD
# we minimize e_n = x - x_n in the A-norm
# using conjugate direction: e_n is A orthogonal to K_n
# equivalent to r_n = b - Ax_n orthogonal to K_n

function cg!(A::Symmetric, b, x, P, T_mat::SymTridiagonal; maxiter = 50, tol = 1e-14)

    r = b - A*x
    r_sq = dot(r, r)
    P[:,1] .= r # first search direction

    alpha = T_mat.dv
    β = T_mat.ev # /!\ length maxiter-1

    for k in 1:maxiter-1

        p_cur = view(P, :, k) # current search direction
        Ap = view(P, :, k+1) # temp storage for K_n expansion. Careful if loops ends, this is garbage

        mul!(Ap, A , p_cur)

        denom = dot(p_cur, Ap)
        alpha[k] = r_sq / denom # r*r/p*Ap

        x .+= alpha[k]*p_cur # new estimate
        r .-= alpha[k]*Ap # new residual = b-Ax

        new_r_sq = dot(r, r)
        residual = sqrt(new_r_sq)

        if residual < tol
            println("Converged in $k steps. Residual: $residual")
            return k
        end

        β[k] = new_r_sq/r_sq # r*r/r_prev*r_prev

        P[:,k+1] .= r + β[k]*p_cur # actual new conjugate direction
        r_sq = new_r_sq
    end

    p_final = view(P, :, maxiter)
    Ap_final = A * p_final # Temporary vector
    alpha[maxiter] = r_sq / dot(p_final, Ap_final)
    x .+= alpha[maxiter] .* p_final
    r .-= alpha[maxiter] .* Ap_final
    new_r_sq = dot(r, r)
    residual = sqrt(new_r_sq)

    println("hit $maxiter iterations. Residual: $residual")
    return(maxiter)
end

# https://discourse.julialang.org/t/using-axpy/7072/12

# bad matrix
m = 300
max_k = 100
X = randn(m, m)
A = Symmetric(X'*X)
b = randn(m)

x_true = A \ b
norm(A*x_true - b)

x = zeros(m)
P = zeros(m, max_k + 1)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
T_mat = SymTridiagonal(alpha, beta_sub)

n = cg!(A, b, x, P, T_mat; maxiter=max_k)
# awful, 100 iterations

# good matrix
m = 300
max_k = 100
X = randn(m, m)/sqrt(m)
A = Symmetric(X'*X + 5I) 
A[1,1] = 12
b = randn(m)

x_true = A \ b
norm(A*x_true - b)

x = zeros(m)
P = zeros(m, max_k + 1)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
T_mat = SymTridiagonal(alpha, beta_sub)

n = cg!(A, b, x, P, T_mat, maxiter=max_k)
# 17 iterations, excellent precision

# Check P
println("Orthonormality (||P'P - I||): ", norm(view(P, :, 1:n)' * view(P, :, 1:n) - I))
# no orthogonality

P_view = view(P, :, 1:n)
for pk in eachcol(P_view)
    pk ./= sqrt(dot(pk, A, pk))
end

println("A - orthogonality (||P'AP - I||): ", norm(P_view' * A * P_view - I))
# okay but slipping just like lanczos

eigvals(A)
eigvals(T_mat[1:n,1:n]) # NOT eigenvalues, P is A-orthogonal base not orthonormal

# CGN
# apply CG to A*A

function cgn!(
    A::AbstractMatrix{T}, 
    b::AbstractVector{T}, 
    x::AbstractVector{T}, 
    P::AbstractMatrix{T}, 
    T_mat::SymTridiagonal{T}; 
    maxiter::Int = 50, 
    tol::T = 1e-12
) where {T<:AbstractFloat}

    r = b - A*x
    r_sq = dot(r, r)
    P[:,1] .= r # first search direction

    alpha = T_mat.dv
    β = T_mat.ev # /!\ length maxiter-1

    tmp = zeros(size(b))

    for k in 1:maxiter-1

        p_cur = view(P, :, k) # current search direction
        Ap = view(P, :, k+1) # temp storage for K_n expansion

        mul!(tmp, A , p_cur) # Ap
        mul!(Ap, A' , tmp) # A*Ap

        denom = dot(p_cur, Ap)
        alpha[k] = r_sq / denom # r*r/p*Ap

        x .+= alpha[k]*p_cur # new estimate
        r .-= alpha[k]*Ap # new residual = b-Ax

        new_r_sq = dot(r, r)
        residual = sqrt(new_r_sq)

        if residual < tol
            println("Converged in $k steps. Residual: $residual")
            return k
        end

        β[k] = new_r_sq/r_sq # r*r/r_prev*r_prev

        P[:,k+1] .= r + β[k]*p_cur # actual new conjugate direction
        r_sq = new_r_sq
    end

    p_final = view(P, :, maxiter)
    Ap_final = A' * (A * p_final) # Temporary vector
    alpha[maxiter] = r_sq / dot(p_final, Ap_final)
    x .+= alpha[maxiter] .* p_final
    r .-= alpha[maxiter] .* Ap_final
    new_r_sq = dot(r, r)
    residual = sqrt(new_r_sq)

    println("hit $maxiter iterations. Residual: $residual")
    return(maxiter)
end

# Bad: Singular values decay like 1, 1/2, 1/3... 1/m
m = 300
max_k = 200
U, _ = qr(randn(m, m))
V, _ = qr(randn(m, m))
S = diagm([1.0/k for k in 1:m])
A = U * S * V'

x_true = A \ b
norm(A*x_true - b)

x = zeros(m)
P = zeros(m, max_k + 1)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
T_mat = SymTridiagonal(alpha, beta_sub)

n = cgn!(A, b, x, P, T_mat, maxiter=max_k)
# very slow but we're getting there

# good
U, _ = qr(randn(m, m))
V, _ = qr(randn(m, m))
S = diagm([5 + 1.0/k for k in 1:m])
S[m] = 12
A = U * S * V'

x_true = A \ b
norm(A*x_true - b)

x = zeros(m)
P = zeros(m, max_k + 1)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
T_mat = SymTridiagonal(alpha, beta_sub)

n = cgn!(A, A'*b, x, P, T_mat, maxiter=max_k)
# 12 steps

println("x - x_n relative error: ", norm(x_true - x)/norm(x_true))