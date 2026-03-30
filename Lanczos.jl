using LinearAlgebra
# Lanczos iteration

# in the special case of A symmetric, H becomes tridiagonal symmetric, and the recurrence has 3 terms instead of n

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

T = promote_type(eltype(A), eltype(b))
Q = zeros(T, m, max_k + 1)
alpha = zeros(T, max_k)
beta = zeros(T, max_k)

num_iters = lanczos!(A, b, Q, alpha, beta, maxiter=max_k)

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