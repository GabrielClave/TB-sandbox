using LinearAlgebra

# Arnoldi iterations

# in place version with views
function arnoldi!(A, b, Q, H; maxiter = 50, tol = 1e-12)
  
    @views Q[:, 1] .= b ./ norm(b)

    for k in 1:maxiter

        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        
        mul!(v, A, qk) # correctly placed in Q[: , k+1]

        # Orthogonalization (Modified Gram-Schmidt)
        for i in 1:k
            qi = view(Q, :, i)
            H[i, k] = dot(qi, v)
            v .-= H[i, k] .* qi  # In-place subtraction
        end

        a = norm(v)
        H[k+1, k] = a

        if a < tol
            println("lucky breakdown in $k steps")
            return k
        end

        v ./= a
    end

    println("hit $maxiter iterations")
    return maxiter
end

# pedagogical
function arnoldi(A,b; maxiter = 50, tol = 1e-12)
    m = length(b)
    Q = zeros(eltype(b), m, maxiter + 1)
    Q[:,1] = normalize(b)

    H = zeros(eltype(b), maxiter + 1, maxiter)

    for k in 1:maxiter
        v = A * Q[:, k] # m . 1

        # Orthogonalization (Modified Gram-Schmidt)
        for i in 1:k
            H[i, k] = dot(Q[:, i], v) # qi*Aqk
            v = v - H[i, k] * Q[:, i]
        end

        a = norm(v)
        H[k+1,k] = a

        if a < tol
            println("lucky breakdown in $k steps")
            return Q[:, 1:k], H[1:k, 1:k], k
        end


        Q[:, k+1] = v ./ a
    end

    println("hit $maxiter iterations")
    return(Q, H, maxiter+1)
end

m = 100
A = randn(m, m)
b = randn(m, 1)

Q, H, n = arnoldi(A,b)
# Check if Q is orthonormal
println("norm(Q'*Q - I): ", norm(Q'*Q - I(size(Q,2))))

# check if AQn = Qn+1H
println("norm(AQn - Qn+1H): ", norm(A*Q[:,1:n-1] - Q*H))

m = 300
max_k = 50
A = randn(m, m)
b = randn(m)

# Pre-allocate outside the function
T = promote_type(eltype(A), eltype(b))
Q = zeros(T, m, max_k + 1)
H = zeros(T, max_k + 1, max_k)

num_iters = arnoldi!(A, b, Q, H, maxiter=max_k)

# Truncate results to the actual iterations performed
Q_final = Q[:, 1:num_iters+1]
H_final = H[1:num_iters+1, 1:num_iters]

println("Orthonormality check: ", norm(Q_final' * Q_final - I))

@time num_iters = arnoldi!(A, b, Q, H, maxiter=max_k) # 50 allocations in kB
@time Q_created, H_created, n = arnoldi(A,b) # 10,000 allocations in MB

H_n = @view H[1:num_iters,1:num_iters]

# using H to approximate eigenvalues of A
ritz_values = eigvals(H_n)

# GMRES

# we want to solve Ax = b
# using the approximation x_n ∈ K_n that minimizes the residuals
# This is equivalent to the least square problem Hy = ||b||e1 that has dim n+1 x n

x_true = A \ b

e1 = zeros(num_iters + 1)
e1[1] = 1.0

y_n = H_final \ (norm(b).*e1)
Qn = @view Q[:,1:num_iters]
x_n = Qn*y_n

println("x - x_n relative error: ", norm(x_true - x_n)/norm(x_true))

max_k = 100
Q = zeros(T, m, max_k + 1)
H = zeros(T, max_k + 1, max_k)

num_iters = arnoldi!(A, b, Q, H, maxiter=max_k)

# Truncate results to the actual iterations performed
Q_final = Q[:, 1:num_iters+1]
H_final = H[1:num_iters+1, 1:num_iters]

H_n = @view H[1:num_iters,1:num_iters]

x_true = A \ b

e1 = zeros(num_iters + 1)
e1[1] = 1.0

y_n = H_final \ (norm(b).*e1)
Qn = @view Q[:,1:num_iters]
x_n = Qn*y_n

println("x - x_n relative error: ", norm(x_true - x_n)/norm(x_true))
# complete garbage
# random matrix is ill-adapted to Krylov subspace ?

# Setup a "Friendly" Problem
m = 300
max_k = 50

A = randn(m, m)/sqrt(m) + 5I # Diagonally dominant matrix, eigenvalues are clustered far from 0
b = randn(m)
x_true = A \ b

T = promote_type(eltype(A), eltype(b))
Q = zeros(T, m, max_k + 1)
H = zeros(T, max_k + 1, max_k)

n = arnoldi!(A, b, Q, H, maxiter=max_k)

β = norm(b)
e₁ = zeros(T, n + 1)
e₁[1] = 1.0

H_tilde = @view H[1:n+1, 1:n]
y_n = H_tilde \ (β .* e₁)

Q_n = @view Q[:, 1:n]
x_n = Q_n * y_n

println("Relative Error: ", norm(x_true - x_n) / norm(x_true)) # excellent
println("Residual Norm:  ", norm(b - A*x_n) / norm(b)) # excellent

# GMRES using a stopping criteria

function incremental_gmres!(A, b, Q, H, x; maxiter = 50, tol = 1e-12)
    β = norm(b)
    @views Q[:, 1] .= b ./ β

    # g tracks the rotated right-hand side (initially β*e1)
    g = zeros(eltype(b), maxiter + 1)
    g[1] = β
    
    # givens rotation
    cs = zeros(maxiter)
    sn = zeros(maxiter)

    for k in 1:maxiter
        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        mul!(v, A, qk)

        for i in 1:k
            qi = view(Q, :, i)
            H[i, k] = dot(qi, v)
            v .-= H[i, k] .* qi
        end
        H[k+1, k] = norm(v)
        v ./= H[k+1, k]

        # Apply previous rotations (n-1) to the new column of H
        for i in 1:k-1
            h_old = H[i, k]
            H[i, k]   =  cs[i] * h_old + sn[i] * H[i+1, k]
            H[i+1, k] = -sn[i] * h_old + cs[i] * H[i+1, k]
        end

        # compute the new givens rotation to make H triangular
        r = sqrt(H[k,k]^2 + H[k+1,k]^2)
        cs[k] = H[k,k] / r
        sn[k] = H[k+1,k] / r

        H[k,k] = r
        H[k+1,k] = 0.0

        # apply the new rotation to the right-hand side vector g
        g_old = g[k]
        g[k] = cs[k]*g_old + sn[k]*g[k+1]
        g[k+1] = -sn[k]*g_old + cs[k]*g[k+1]
        
        residual = abs(g[k+1]) # The residual is the last element of the transformed g!
        
        if residual < tol
            # ready to solve by backsubstitution
            H_k = @view H[1:k, 1:k]
            y = UpperTriangular(H_k) \ g[1:k]
            
            mul!(x, view(Q, :, 1:k), y) # x = Q_k * y
            println("Converged in $k steps. Residual: $residual")
            return k
        end
    end

    # If we hit maxiter, solve anyway
    H_k = @view H[1:maxiter, 1:maxiter]
    y = UpperTriangular(H_k) \ g[1:maxiter]
    mul!(x, view(Q, :, 1:maxiter), y)
    println("hit $maxiter iterations. Residual: $residual")
    return maxiter
end

x = zeros(m)

k = incremental_gmres!(A,b,Q,H,x) # 19 steps

println("Relative Error: ", norm(x_true - x) / norm(x_true)) # excellent

# Lanczos iteration

# in the special case of A symetric, H becomes tridiagonal symmetric, and the recurrence has 3 terms instead of n

function lanczos!(A, b, Q, alpha, beta; maxiter = 50, tol = 1e-12)
    
    @views Q[:, 1] .= b ./ norm(b)

    for k in 1:maxiter

        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        
        mul!(v, A, qk) # A*qk = βk-1 qk-1 + αk qk + βn+1 qn+1

        # Lanczos Orthogonalization
        alpha[k] = dot(qk, v) # qk*Aqk
        v .-= alpha[k]*qk

        if k>1
            qk_prev = view(Q, :, k-1)
            v .-= beta[k-1]*qk_prev
        end

        beta[k] = norm(v)
        
        if beta[k] < tol
            println("lucky breakdown in $k steps")
            return k
        end
        
        v ./= beta[k]
    end

    println("hit $maxiter iterations")
    return maxiter
end

m = 500
max_k = 200
X = randn(m, m)
A = X'*X
b = randn(m)

# Pre-allocate outside the function
T = promote_type(eltype(A), eltype(b))
Q = zeros(T, m, max_k + 1)
alpha = zeros(T, max_k)
beta = zeros(T, max_k)

num_iters = lanczos!(A, b, Q, alpha, beta, maxiter=max_k)

# Truncate results to the actual iterations performed
Q_final = Q[:, 1:num_iters+1]

println("Orthonormality check: ", norm(Q_final' * Q_final - I))
# Orthonormality for 20 steps
# completely lost at 200 steps
# we never "manually enforce" orthogonality of qn and qn-2 like we do for Arnoldi

# with selective re-orthogonalization
function lanczos_SO!(A, b, Q, T_mat::SymTridiagonal; maxiter = 50, tol = 1e-12, eta = 1e-5)

    alpha = T_mat.dv # that's a view
    beta  = T_mat.ev # /!\ length maxiter-1
    
    @views Q[:, 1] .= b ./ norm(b)

    for k in 1:maxiter
        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        
        mul!(v, A, qk)
        alpha[k] = dot(qk, v)
        
        v .-= alpha[k] .* qk
        if k > 1
            v .-= beta[k-1] .* view(Q, :, k-1)
        end

        # SELECTIVE RE-ORTHOGONALIZATION
        # We check if q_{k+1} is still orthogonal to q_1

        if abs(dot(view(Q, :, 1), v)) > eta
            println("Orthogonality slip detected at step $k.")
            # modified GS
            for i in 1:k
                qi = view(Q, :, i)
                v .-= dot(qi, v) .* qi
            end
        end

        b_val = norm(v)
        
        if b_val < tol
            println("Lucky breakdown in $k steps")
            return k
        end

        if k < maxiter # beta[maxiter] is undefined
            beta[k] = b_val
            v ./= b_val
        end
    end

    return maxiter
end

m = 300
max_k = 100
X = randn(m, m)
A = X'*X
b = randn(m)

Q = zeros(m, max_k + 1)
alpha = zeros(max_k)
beta_sub = zeros(max_k - 1)
T_mat = SymTridiagonal(alpha, beta_sub)

n = lanczos_SO!(A, b, Q, T_mat, maxiter=max_k)

# Check Q
println("Orthonormality (||Q'Q - I||): ", norm(view(Q, :, 1:n)' * view(Q, :, 1:n) - I))
# better but not great