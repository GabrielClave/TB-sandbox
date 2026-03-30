using LinearAlgebra

# Jacobi

function jacobi_eigenvalues(A; tol = 1e-6, maxiter = 100)
    Ak = copy(A)
    n = size(A,1)

    for k in 1:maxiter # one "sweep"

        if norm(tril(Ak, -1)) < tol # elements below the diagonal
            println("Converged in $k iterations")
            return diag(Ak)
        end    

        for i in 1:n-1
            for j in i+1:n

                d = Ak[i,j]
                if abs(d) < 1e-15 continue end

                a = Ak[i,i]
                b = Ak[j,j]

                τ = (b - a)/(2*d) # cotan
                t = sign(τ)/(abs(τ) + sqrt(1 + τ^2)) # tan, solution of t² + 2τt - 1 = 0 (I love trigonometry)

                c = 1/(sqrt(1+t^2)) # cos
                s = c*t # sin

                # The only elements "modified twice" are Aii Ajj Aij = Aji
                c2 = c^2
                s2 = s^2
                Ak[i,i] = c2*a - 2*s*c*d + s2*b
                Ak[j,j] = s2*a + 2*s*c*d + c2*b
                Ak[i,j] = Ak[j,i] = 0 # that's the point

                # Now we apply J*AkJ
                # J*x modifies rows i and j
                # xJ modifies col i and j

                for r in 1:n
                    if r != i && r != j
                        # current values
                        Ari = Ak[r, i]
                        Arj = Ak[r, j]

                        # Apply rotation to the pair
                        new_ri = c * Ari - s * Arj
                        new_rj = s * Ari + c * Arj

                        # Update both keeping the symmetry
                        Ak[r, i] = Ak[i, r] = new_ri
                        Ak[r, j] = Ak[j, r] = new_rj
                    end
                end

            end
        end


    end

    return(diag(Ak))
    
end


# X = randn(50, 50)
# A = X + X'

# ev_true = eigvals(A)

# eigenvalues = jacobi_eigenvalues(A) # max
# check_accuracy(eigenvalues, ev_true)

function jacobi_eigenvalues!(A; tol=1e-10, maxiter=100)
    n = size(A, 1)

    for k in 1:maxiter
        off_norm = 0.0
        for j in 2:n, i in 1:j-1 # only look at the off diagonal O(n) instead of 0(n²)
            off_norm += A[i, j]^2
        end
        
        if sqrt(off_norm) < tol
            return diag(A)
        end

        for i in 1:n-1
            for j in i+1:n
                # Pivot element
                Apq = A[i, j]
                if abs(Apq) < 1e-14 continue end 

                # 2. rotation parameters
                App = A[i, i]
                Aqq = A[j, j]
                τ = (Aqq - App) / (2 * Apq)
                
                t = sign(τ) / (abs(τ) + hypot(1, τ)) # hypot has better numerical stability than sqrt(1 + τ^2)
                c = 1 / hypot(1, t)
                s = c * t

                # 3. Update the Diagonal/Pivot elements
                A[i, i] = c^2 * App - 2*s*c * Apq + s^2 * Aqq
                A[j, j] = s^2 * App + 2*s*c * Apq + c^2 * Aqq
                A[i, j] = A[j, i] = 0.0

                # 4. Row/Column updates                
                for r in 1:i-1
                    Ari, Arj = A[r, i], A[r, j]
                    A[r, i] = A[i, r] = c * Ari - s * Arj
                    A[r, j] = A[j, r] = s * Ari + c * Arj
                end

                for r in i+1:j-1
                    Ari, Arj = A[r, i], A[r, j]
                    A[r, i] = A[i, r] = c * Ari - s * Arj
                    A[r, j] = A[j, r] = s * Ari + c * Arj
                end

                for r in j+1:n
                    Ari, Arj = A[r, i], A[r, j]
                    A[r, i] = A[i, r] = c * Ari - s * Arj
                    A[r, j] = A[j, r] = s * Ari + c * Arj
                end
            end
        end
    end
    return diag(A)
end

# X = randn(50, 50)
# A = X + X'

# ev_true = eigvals(A)

# eigenvalues = jacobi_eigenvalues!(A) # max
# check_accuracy(eigenvalues, ev_true)


# Bisection
# binary search for eigenvalues

# A is real and symmetric

# if A is tridiagonal, there is a recurrence to compute det(Ak-xI)
# pk = (T[k,k] - x)*pk_1 - (T[k+1,k]^2)*pk_2
# but this grows or shrink very quickly
# so we keep the signs in mind q_k = (d_k - x) - e_{k-1}^2 / q_{k-1}

function count_eigenvalues_less_than(d, e, x)
    n = length(d)
    count = 0
    q = d[1] - x
    
    if q < 0
        count += 1
    end
    
    for k in 2:n
        # Prevent division by zero if q is exactly 0
        if q == 0
            q = 1e-14 
        end
        
        # The recurrence: q_k = (d_k - x) - e_{k-1}^2 / q_{k-1}
        q = (d[k] - x) - (e[k-1]^2) / q
        
        if q < 0
            count += 1
        end
    end
    return count
end

# find the n-th eigenvalue of A, starting in [a,b] where there is an eigenvalue (eg Gerschgorin)
function bisection_eigenvalue(d, e, a, b, n; tol=1e-10, maxiter=100)

    for k in 1:maxiter
        
        midpoint = (a+b)/2

        if (b-a) < tol
            println("converged in $k iterations")
            return(midpoint)
        end

        N = count_eigenvalues_less_than(d, e, midpoint) # number of eigenvalue less than midpoint

        if N < n
            a = midpoint # look to the right
        else
            b = midpoint # look to the left
        end
    end

    println("did not converged in $maxiter iterations")
    return(midpoint)

end

# X = randn(10, 10)
# A = X + X'

# ev_true = eigvals(A)

# tridiagonal_reduction!(A)

# n = 5
# ev = bisection_eigenvalue(diag(A), diag(A,-1), -10, 10, n)

# println("relative error: ", abs((ev - ev_true[n])/ev_true[n]))

# Divide and conquer

# dumb recursion
function recursive_dc_no_deflation(T)
    n = size(T,1)

    # direct solve
    if n == 1
        return ones(1, 1), [T[1, 1]]
    elseif n == 2
        sol = eigen(Matrix(T))
        return sol.vectors, sol.values
    end

    # divide, conquer, combine

    k = div(n,2)

    β = T[k, k+1]
    T1 = T[1:k, 1:k]
    T1[k, k] -= β

    T2 = T[k+1:n, k+1:n]
    T2[1,1] -= β

    Q1,D1 = recursive_dc_no_deflation(T1)
    Q2,D2 = recursive_dc_no_deflation(T2)

    D = vcat(D1, D2)

    # z = [last row of Q1' ; first row of Q2']
    z = vcat(Q1[end, :], Q2[1, :])

    # solve the secular equation, find the zeros of f(λ)
    new_eigenvalues = solve_secular(D, z, β)
    
    # Eigenvector for λ_j is proportional to (D - λ_j I)⁻¹ z
    # Then rotate back by Q_total = [Q1 0; 0 Q2]
    Q_combined = blockdiag(Q1, Q2)
    new_vectors = zeros(n, n)
    
    for j in 1:n
        λ = new_eigenvalues[j]
        # v_hat = (D - λI)⁻¹ z
        v = z ./ (D .- λ) # can divide by 0 ?
        v /= norm(v) # Normalize
        new_vectors[:, j] = Q_combined * v
    end
    
    return new_vectors, new_eigenvalues
end

# find roots of the secular equation
function solve_secular(D, z, β)
    n = length(D)
    roots = zeros(n)
    sorted_idx = sortperm(D)
    d_sorted = D[sorted_idx]
    z_sorted = z[sorted_idx]
    
    for i in 1:n
        # Define secular function f(λ)
        f(λ) = 1 + β * sum((z_sorted.^2) ./ (d_sorted .- λ))
        
        # Determine search interval [a, b] for bisection
        if i < n
            a, b = d_sorted[i] + 1e-10, d_sorted[i+1] - 1e-10
        else
            a, b = d_sorted[n] + 1e-10, d_sorted[n] + abs(β)*norm(z)^2 
        end
        
        # Simple bisection (Impractical)
        roots[i] = simple_bisection(f, a, b)
    end
    return roots
end

function blockdiag(Q1, Q2)
    n1, n2 = size(Q1, 1), size(Q2, 1)
    Q = zeros(n1+n2, n1+n2)
    Q[1:n1, 1:n1] = Q1
    Q[n1+1:end, n1+1:end] = Q2
    return Q
end

function simple_bisection(f, a, b; iters=50)
    for _ in 1:iters
        m = (a + b) / 2
        f(a) * f(m) < 0 ? b = m : a = m
    end
    return (a + b) / 2
end

T = Matrix(SymTridiagonal([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
Q, D = recursive_dc_no_deflation(T )# some "NAs" eigenvectors, eigenvalues are correct
println("Computed Eigenvalues: ", sort(D))
println("Actual Eigenvalues:   ", sort(eigen(Matrix(T)).values))