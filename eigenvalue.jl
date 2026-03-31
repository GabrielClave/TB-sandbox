using LinearAlgebra

# reduction to Hessenberg form

function hessenberg_reduction!(A)
    m = size(A, 1)
    T = eltype(A)

    v_buf = zeros(T, m)    # Householder vectors
    work_v = zeros(T, m)   # row products (v_tl)
    work_col = zeros(T, m)   # column products (tr_v)
    
    for k in 1:m-2
        x = @view A[k+1:m, k] # this will modify A, but we don't care
        nx = norm(x)
        
        if nx != 0
            # reflection vector v
            s = sign(x[1]) == 0 ? one(T) : sign(x[1])
            
            vk = @view v_buf[1:m-k]
            vk .= x
            vk[1] += s * nx
            normalize!(vk)
            
            # Left multiplication: H = (I - 2vv')H
            # Apply to trailing submatrix [k+1:m, k+1:m]
            sub_trailing = @view A[k+1:m, k+1:m]
            v_tl = @view work_v[1:m-k]
            # sub_trailing -= 2 * v * (v' * sub_trailing)
            mul!(v_tl, sub_trailing', vk)
            BLAS.ger!(-T(2.0), vk, v_tl, sub_trailing)
            
            # Right multiplication: H = H(I - 2vv')
            # Apply to ALL rows for columns [k+1:m]
            full_trailing = @view A[1:m, k+1:m]
            tr_v = @view work_col[1:m]
            # full_trailing -= 2 * (full_trailing * v) * v'
            mul!(tr_v, full_trailing, vk)
            BLAS.ger!(-T(2.0), tr_v, vk, full_trailing)
            
            A[k+1, k] = -s * nx
            A[k+2:m, k] .= 0
        end
    end
    return A
end

n = 5
A_orig = randn(n, n)
A = copy(A_orig)

H2 = hessenberg_reduction!(A)

# similarity transformation should preserve eigenvalues
eigvals(A_orig) ≈ eigvals(H2)

function tridiagonal_reduction!(A)
    m = size(A, 1)
    T = eltype(A)

    v_buf = zeros(T, m)      # Householder vector
    y_buf = zeros(T, m)      # 2 * A * v
    w_buf = zeros(T, m)      # y - (v'y)v
    
    for k in 1:m-2
        x = @view A[k+1:m, k] # this will modify A
        nx = norm(x)
        
        if nx != 0
            # reflection vector v
            s = sign(x[1]) == 0 ? one(T) : sign(x[1])
            
            vk = @view v_buf[1:m-k]
            vk .= x
            vk[1] += s * nx
            normalize!(vk)

            # Submatrix to update
            sub = @view A[k+1:m, k+1:m]
            yk = @view y_buf[1:m-k]
            wk = @view w_buf[1:m-k]

            # (I - 2vv')H(I - 2vv') ⟺ rank 2 update A - vw' - wv'

            # y = 2 * A * v
            BLAS.symv!('U', T(2.0), sub, vk, T(0.0), yk) 
            # w = y - (v'y)v
            wk .= yk .- dot(vk, yk) .* vk

            BLAS.ger!(-one(T), vk, wk, sub) # we should write a loop
            BLAS.ger!(-one(T), wk, vk, sub) # we're doing twice the work
            
            val = -s * nx
            A[k+1, k] = val
            A[k, k+1] = val # by symmetry

            A[k+2:m, k] .= 0
            A[k, k+2:m] .= 0 # rows are zeroed by symmetry
        end
    end
    return A
end

n = 200
X = randn(n, n)
A_orig = X + X'
A = copy(A_orig)
issymmetric(A)

@time T = tridiagonal_reduction!(A)

issymmetric(T) #false !
T ≈ T' # True !
eigvals(A_orig) ≈ eigvals(T)

# power and inverse iteration

function power_iteration(A, v; tol=1e-4, maxiter=100)
    T = eltype(A)
    m = size(A, 1)
    
    v = Vector{T}(vec(v)) # issues with randn
    Av = zeros(T, m)
    res_vec = zeros(T, m)
    
    normalize!(v)
    λ = zero(T)
    
    for k in 1:maxiter

        mul!(Av, A, v)
        λ_new = dot(v, Av)  # Rayleigh quotient: v' * A * v (v normalized)
        
        # Check convergence
        res_vec .= Av .- λ_new .* v
        residual = norm(res_vec) # more expensive
        if residual < tol * (abs(λ_new) + 1)
            println(k)
            return v, λ_new
        end

        v .= Av ./ norm(Av)
        λ = λ_new
    end
    println("hit max iterations")
    return v, λ
end

n = 100
X = randn(n, n)
A = X + X'
A[1,1] = 100
v = randn(n, 1)

@time v, λ = power_iteration(A,v)

eigvals(A)

function inverse_iteration(A, v, μ ; tol=1e-4, maxiter=100)
    T = eltype(A)
    m = size(A, 1)
    
    v = Vector{T}(vec(v)) # issues with randn
    w = zeros(T, m)
    v_prev = zeros(T, m)
    
    normalize!(v)
    λ = zero(T)
    F = factorize(A - μ*I) # once here
    
    for k in 1:maxiter
        copyto!(v_prev, v)

        # solve (A - μI)w = v
        ldiv!(w,F,v_prev)
        
        # Update and Normalize
        nw = norm(w)
        v .= w ./ nw
        
        dot_product = dot(v, v_prev) # Rayleigh quotient: v' * A * v
        λ_new = μ + dot_product / nw
        
        # Check convergence
        w .= (v_prev .- dot_product .* v) ./ nw
        residual = norm(w)

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
v = randn(n)
A[1,1] = 100

v, λ = power_iteration(A,v) # 7 iterations
v, λ = inverse_iteration(A,v,1) # 10 iterations from a garbage guess, but did not return the max λ

eigvals(A)

# Rayleigh quotient iteration

function rqi!(A, v, μ ; tol=1e-4, maxiter=100)
    T = eltype(A)
    m = size(A, 1)
    
    w = zeros(T, m)
    v_prev = zeros(T, m)
    
    normalize!(v)
    λ = T(μ)

    A_shifted = A - λ*I
    F = lu!(A_shifted)
    
    for k in 1:maxiter
        copyto!(v_prev, v)

        # solve (A - μI)w = v
        ldiv!(w,F,v_prev)
        
        # Update and Normalize
        nw = norm(w)
        v .= w ./ nw
        
        dot_product = dot(v, v_prev) # Rayleigh quotient: v' * A * v
        λ += dot_product / nw
        
        # Check convergence
        w .= (v_prev .- dot_product .* v) ./ nw
        residual = norm(w)

        if residual < tol
            println(k)
            return v, λ
        end        
        λ = λ
        A_shifted .= A - λ*I
        F = lu!(A_shifted) # O(n³)

    end
    println("hit max iterations")
    return v, λ
end

n = 10
X = randn(n, n)
A = X + X'
v = randn(n)
μ = 50

v, λ = inverse_iteration(A,v,3, tol = 1e-12) # 80 iteration
v, λ = rqi!(A, v, μ, tol = 1e-12) # 5 iteration !

v = randn(n)
@time rqi!(A, v, μ, tol = 1e-12)

eigvals(A)

# Pure QR

function qr_eigen(A; tol = 1e-6, maxiter = 500)
    Ak = copy(A)
    Bk = similar(Ak)
    F = qr!(Ak) # QR factorization of A
    mul!(Bk, F.R , F.Q)
    
    for k in 1:maxiter
        if norm(tril(Ak, -1)) < tol # elements below the diagonal
            println("Converged in $k iterations")
            return diag(Ak)
        end
        F = qr!(Ak) # QR factorization of A
        mul!(Bk, F.R , F.Q)
        Ak .= Bk
    end
    println("hit max iterations")
    return diag(Ak)
end

n = 100
X = randn(n, n)
A = X + X'

eigvals(A)
eigenvalues = qr_eigen(A) # stopped at maxiter

@time qr_eigen(A)

function qr_shift_rqs!(A, Bk; tol = 1e-6, maxiter = 500)
    m = size(A,1)
    
    for k in 1:maxiter

        off_diag_norm = 0.0
        for j in 1:m-1, i in j+1:m
            off_diag_norm += abs2(A[i, j]) # all elements below the diagonal
        end

        if sqrt(off_diag_norm) < tol 
            println("Converged in $k iterations")
            return diag(A)
        end

        μ = A[m,m] # rayleigh quotient shift: q*Aq = t(q)
        for i in 1:m
            A[i, i] -= μ
        end
        F = qr!(A) # QR factorization of A - μI
        mul!(Bk, F.R , F.Q)
        A .= Bk 
        for i in 1:m
            A[i, i] += μ
        end # we still have A = Q* A-1 Q
    end
    println("hit max iterations")
    return diag(A)
end

Ak = copy(A)
Bk = similar(Ak)
eigenvalues = qr_shift_rqs!(Ak, Bk) # either converged in #100 iterations, or get stuck
Ak = copy(A)
Bk = similar(Ak)
@time qr_shift_rqs!(Ak, Bk)

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


        F = qr(Ak - μ*I) # QR factorization of A - μI
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

# in practice we apply tridiagonal_reduction!(A)
# although the code does not account for it anyway

tridiagonal_reduction!(A)

ev_true ≈ eigvals(A)
check_accuracy(eigvals(A), ev_true) # sanity check

eigenvalues = qr_eigen(A) # 182
eigenvalues = qr_shift_rqs(A) # 230
eigenvalues = qr_shift_ws(A) # 206

# we are not seeing great progress because we aren't doing deflation (we are super optimizing the convergence on the "last" eigenvalue but that's about it)
# the idea is that once the last col as "converged", we stop caring about it and optimize the second last

function qr_shifted_deflation(A; tol=1e-12, maxiter=20)
    Ak = copy(A)
    n = size(Ak, 1)
    eigenvalues = zeros(eltype(A), n)
    total_iterations = 0 # useless
    
    current_n = n #the size of the "active" top-left submatrix
    
    while current_n > 1
        # Base case: 2x2 direct solve
        if current_n == 2
            a, b, c, d = Ak[1,1], Ak[1,2], Ak[2,1], Ak[2,2]
            tr = a + d
            det = a*d - b*c
            gap = sqrt(tr^2 - 4det + 0im)
            eigenvalues[1] = (tr + gap) / 2
            eigenvalues[2] = (tr - gap) / 2
            current_n = 0
            break
        end

        iter = 0 # iteration are PER EIGENVALUE not global
        while iter < maxiter
            # Wilkinson Shift on the active bottom-right 2x2
            m = current_n
            a = Ak[m-1, m-1]
            b = Ak[m-1, m]
            c = Ak[m, m]
            
            δ = (a - c) / 2
            μ = c - (sign(δ + eps()) * b^2) / (abs(δ) + sqrt(δ^2 + b^2))

            # Shifted QR step on the ACTIVE submatrix only
            active_block = @view Ak[1:m, 1:m]
            F = qr(active_block - μ*I)
            active_block .= F.R * F.Q + μ*I # Ak IS modified

            # Check for deflation at the bottom of the active block
            if abs(Ak[m, m-1]) < tol # O(1)
                eigenvalues[m] = Ak[m, m]
                current_n -= 1 # "Deflate": ignore the last row/col                
                break 
            end
            
            iter += 1
            total_iterations += 1
            if iter == maxiter
                println("Reached maxiter for eigenvalue $m")
                eigenvalues[m] = Ak[m, m]
                current_n -= 1
            end
        end
    end
    
    # If we shrunk down to a 1x1
    if current_n == 1
        eigenvalues[1] = Ak[1,1]
    end
    
    println("Converged in $total_iterations iterations")
    return eigenvalues
end

X = randn(50, 50)
A = X + X'

tridiagonal_reduction!(A)

ev_true = eigvals(A)

eigenvalues = qr_eigen(A) # max
eigenvalues = qr_shift_rqs(A) # max
eigenvalues = qr_shift_ws(A) # max
eigenvalues = qr_shifted_deflation(A) #55 iteration # approx one per row !

check_accuracy(eigenvalues, ev_true)