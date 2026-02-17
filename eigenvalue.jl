using LinearAlgebra

# reduction to Hessemberg form

function hessemberg_householder(A)
        m = size(A,1)
        H = copy(float(A))

        for k in 1:m-2
            x = copy(H[k+1:m,k]) # vector to reflect
            nx = norm(x)

            s = sign(x[1]) == 0 ? one(eltype(x)) : sign(x[1])
            x[1] += s * nx

            v = x / norm(x) # reflexion vector

            H[k+1:m,k+1:m] .-= 2v*v'H[k+1:m,k+1:m] # left multiplication # beware of the order this will be expensive 
            H[1:m,k+1:m] .-= 2H[1:m,k+1:m]*v*v' # right multiplication
            
            H[k+1,k] = -s*nx
            H[k+2:m,k] .=0

        end
    return(H)
end

function hessenberg_reduction!(A)
    m = size(A, 1)
    T = eltype(A)
    
    for k in 1:m-2
        x = @view A[k+1:m, k] # this will modify A, but we don't care
        nx = norm(x)
        
        if nx != 0
            # reflection vector v
            s = sign(x[1]) == 0 ? one(T) : sign(x[1])
            
            # Create v in-place to avoid extra allocations
            v = copy(x)
            v[1] += s * nx
            normalize!(v)
            
            # Left multiplication: H = (I - 2vv')H
            # Apply to trailing submatrix [k+1:m, k+1:m]
            sub_trailing = @view A[k+1:m, k+1:m]
            # sub_trailing -= 2 * v * (v' * sub_trailing)
            v_tl = v' * sub_trailing # Vector-Matrix: O(p^2) 
            sub_trailing .-= 2 .* v * v_tl # Outer product update: O(p^2) (still allocates "tmp matrix" v * v_tl)
            
            # Right multiplication: H = H(I - 2vv')
            # Apply to ALL rows for columns [k+1:m]
            full_trailing = @view A[1:m, k+1:m]
            # full_trailing -= 2 * (full_trailing * v) * v'
            tr_v = full_trailing * v
            full_trailing .-= 2 .* tr_v * v'
            
            A[k+1, k] = -s * nx
            A[k+2:m, k] .= 0
        end
    end
    return A
end

n = 5
A_orig = randn(n, n)
A = copy(A_orig)

H = hessemberg_householder(A_orig)
H2 = hessenberg_reduction!(A)

# similarity transformation should preserve eigenvalues
eigvals(A_orig) ≈ eigvals(H)
eigvals(A_orig) ≈ eigvals(H2)

function tridiagonal_reduction!(A)
    m = size(A, 1)
    T = eltype(A)
    
    for k in 1:m-2
        x = @view A[k+1:m, k] # this will modify A, but we don't care
        nx = norm(x)
        
        if nx != 0
            # reflection vector v
            s = sign(x[1]) == 0 ? one(T) : sign(x[1])
            
            # Create v in-place to avoid extra allocations
            v = copy(x)
            v[1] += s * nx
            normalize!(v)
            
            # Left multiplication: H = (I - 2vv')H
            # Apply to trailing submatrix [k+1:m, k+1:m]
            sub_trailing = @view A[k+1:m, k+1:m] # we could use the symmetry to do half as many operations
            # sub_trailing -= 2 * v * (v' * sub_trailing)
            v_tl = v' * sub_trailing # Vector-Matrix: O(p^2) 
            sub_trailing .-= 2 .* v * v_tl # Outer product update: O(p^2) (still allocates "tmp matrix" v * v_tl)
            
            # Right multiplication: H = H(I - 2vv')
            # Now applied to trailing submatrix [k+1:m, k+1:m] 
            optimized_trailing = @view A[k+1:m, k+1:m] # we could use the symmetry to do half as many operations
            # optimized_trailing -= 2 * (optimized_trailing * v) * v'
            tr_v = optimized_trailing * v
            optimized_trailing .-= 2 .* tr_v * v'
            
            A[k+1, k] = -s * nx
            A[k, k+1] = -s * nx # by symmetry

            A[k+2:m, k] .= 0
            A[k, k+2:m] .= 0 # rows are zeroed by symmetry
        end
    end
    return A
end

n = 6
X = randn(n, n)
A_orig = X + X'
A = copy(A_orig)
issymmetric(A)

T = tridiagonal_reduction!(A)

issymmetric(T) #false !
T ≈ T' # True !
eigvals(A_orig) ≈ eigvals(T)

# power and inverse iteration

function power_iteration(A, v; tol=1e-4, maxiter=100) # absolute tolerance is fishy
    v = copy(v) 
    normalize!(v)
    
    λ = zero(eltype(A))
    
    for k in 1:maxiter
        v = A * v # power iteration
        normalize!(v)
        λ_new = dot(v, A, v)  # Rayleigh quotient: v' * A * v
        
        # Check convergence
        residual = norm(A*v - λ_new*v) # more expensive
        if residual < tol
            println(k)
            return v, λ_new
        end        
        λ = λ_new
    end
    println("hit max iterations")
    return v, λ
end

n = 10
X = randn(n, n)
A = X + X'
v = randn(n, 1)

v, λ = power_iteration(A,v)

eigvals(A)

function inverse_iteration(A, v, μ ; tol=1e-4, maxiter=100)
    v = copy(v) 
    normalize!(v)
    
    λ = zero(eltype(A))

    F = factorize(A - μ*I) # not to repeat it every loop
    
    for k in 1:maxiter
        # solve (A - μI)w = v
        v = F \ v
        normalize!(v)
        λ_new = dot(v, A, v)  # Rayleigh quotient: v' * A * v
        
        # Check convergence
        residual = norm(A*v - λ_new*v)
        if residual < tol
            println(k)
            return v, λ_new
        end        
        λ = λ_new
    end
    println("hit max iterations")
    return v, λ
end

n = 50
X = randn(n, n)
A = X + X'
v = randn(n, 1)

v, λ = power_iteration(A,v) # 89 iterations
v, λ = inverse_iteration(A,v,1) # 2 iterations from a garbage guess, but did not return the max λ

eigvals(A)

# Rayleigh quotient iteration

function rqi(A, v; tol=1e-4, maxiter=100)
    v = copy(v) 
    normalize!(v)
    
    λ = dot(v, A, v)
    
    for k in 1:maxiter
        # solve (A - μI)w = v
        v = (A - λ*I) \ v
        normalize!(v)
        λ_new = dot(v, A, v)  # Rayleigh quotient: v' * A * v
        
        # Check convergence
        residual = norm(A*v - λ_new*v)
        if residual < tol
            println("Converged in $k iterations")
            return v, λ_new
        end        
        λ = λ_new
    end
    println("hit max iterations")
    return v, λ
end

n = 10
X = randn(n, n)
A = X + X'
v = randn(n, 1)

v, λ = inverse_iteration(A,v,3, tol = 1e-12) # 80 iteration
v, λ = rqi(A, v, tol = 1e-12) # 1 iteration !

eigvals(A)

# Pure QR

function qr_eigen(A; tol = 1e-6, maxiter = 500)
    Ak = copy(A)

    for k in 1:maxiter
        F = qr(Ak) # QR factorisation of A
        Ak = F.R * F.Q
        if norm(tril(Ak, -1)) < tol # elements below the diagonal
            println("Converged in $k iterations")
            return diag(Ak)
        end    
    end
    println("hit max iterations")
    return diag(Ak)
end

n = 10
X = randn(n, n)
A = X + X'
v = randn(n, 1)

eigvals(A)
eigenvalues = qr_eigen(A) # stopped at 458, but with excellent accuracy

function qr_shift_rqs(A; tol = 1e-6, maxiter = 500)
    Ak = copy(A)
    m = size(A,1)
    
    for k in 1:maxiter
        μ = Ak[m,m] # rayleigh quotient shift: q*Aq = t(q)
        F = qr(Ak - μ*I) # QR factorisation of A - μI
        Ak = F.R * F.Q + μ*I # we still have Ak = Q* Ak-1 Q

        if norm(tril(Ak, -1)) < tol # elements below the diagonal
            println("Converged in $k iterations")
            return diag(Ak)
        end    
    end
    println("hit max iterations")
    return diag(Ak)
end

eigenvalues = qr_shift_rqs(A) # either converged in #100 iterations, or get stuck

function qr_shift_ws(A; tol = 1e-6, maxiter = 500)
    Ak = copy(A)
    n = size(A,1)
    
    for k in 1:maxiter
        # Look at the bottom right 2x2: [a b; b c]
        a = Ak[n-1, n-1]
        b = Ak[n-1, n]
        c = Ak[n, n]
        
        δ = (a - c) / 2
        # Sign-preserving shift formula
        μ = c - (sign(δ + eps()) * b^2) / (abs(δ) + sqrt(δ^2 + b^2))


        F = qr(Ak - μ*I) # QR factorisation of A - μI
        Ak = F.R * F.Q + μ*I # we still have Ak = Q* Ak-1 Q

        if norm(tril(Ak, -1)) < tol # elements below the diagonal
            println("Converged in $k iterations")
            return diag(Ak)
        end    
    end
    println("hit max iterations")
    return diag(Ak)
end

function check_accuracy(my_ev, true_ev)
    # Sort both by absolute magnitude to align them
    mine_sorted = sort(my_ev, by=abs, rev=true)
    true_sorted = sort(true_ev, by=abs, rev=true)
    
    error_norm = norm(mine_sorted - true_sorted)
    max_error = maximum(abs.(mine_sorted - true_sorted))
    
    println("--- Accuracy Check ---")
    println("Residual Norm: ", error_norm)
    println("Max Absolute Error: ", max_error)
end

n = 10
X = randn(n, n)
A = X + X'
v = randn(n, 1)

ev_true = eigvals(A)
eigenvalues = qr_eigen(A) # stopped at 458, but with excellent accuracy
check_accuracy(eigenvalues, ev_true)
eigenvalues = qr_shift_rqs(A)
check_accuracy(eigenvalues, ev_true)
eigenvalues = qr_shift_ws(A)
check_accuracy(eigenvalues, ev_true)

# in practise we apply tridiagonal_reduction!(A)
# although the code does not account for it anyway

tridiagonal_reduction!(A)

ev_true ≈ eigvals(A)
check_accuracy(eigvals(A), ev_true) # sanity check

eigenvalues = qr_eigen(A) # 182
eigenvalues = qr_shift_rqs(A) # 230
eigenvalues = qr_shift_ws(A) # 206

# we are not seeing great progress because we aren't doing deflation (we are super optimizing the conbergence on the "last" eigenvalue but that's about it) 