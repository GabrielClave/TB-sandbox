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


X = randn(50, 50)
A = X + X'

ev_true = eigvals(A)

eigenvalues = jacobi_eigenvalues(A) # max
check_accuracy(eigenvalues, ev_true)

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