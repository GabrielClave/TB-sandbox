using LinearAlgebra

# Arnoldi iterations

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

    return maxiter
end

# --- Usage ---
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