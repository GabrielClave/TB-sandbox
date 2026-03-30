using LinearAlgebra

# Arnoldi iterations

function arnoldi!(A, b, Q, H; maxiter = 50, tol = 1e-12)
  
    @views Q[:, 1] .= b ./ norm(b)

    for k in 1:maxiter

        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        
        mul!(v, A, qk) # placed in Q[: , k+1]

        # Orthogonalization (Modified Gram-Schmidt)
        for i in 1:k
            qi = view(Q, :, i)
            H[i, k] = dot(qi, v)
            v .-= H[i, k] .* qi  # In-place
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

T = promote_type(eltype(A), eltype(b))
Q = zeros(T, m, max_k + 1)
H = zeros(T, max_k + 1, max_k)

num_iters = arnoldi!(A, b, Q, H, maxiter=max_k)

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