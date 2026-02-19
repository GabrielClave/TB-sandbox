using LinearAlgebra

# Jacobi

function jacobi_eigenvalues(A; tol = 1e-6, maxiter = 500)
    M = copy(A)
    n = size(A,1)

    for k in 1:maxiter # one "sweep"
        for i in 1:n-1
            for j in i+1:n

                a = A[i,i]
                b = A[j,j]
                d = A[i,j]

                τ = (b - a)/(2*d) # cotan
                t = sign(τ)/(abs(τ) + sqrt(1 + τ^2)) # tan, solution of t² + 2τt - 1 = 0 (I love trigonometry)

                c = 1/(sqrt(1+t^2)) # cos
                s = c*t # sin

                # Now we apply J*AJ
                # J*x modifies rows i and j
                # xJ modifies col i and j

                # The only elements "modified twice" are Aii Ajj Aij = Aji
                c2 = c^2
                s2 = s^2
                A[i,i] = c2*a - 2*s*c*d + s2*b
                A[j,j] = s2*a + 2*s*c*d + c2*b
                A[i,j] = A[j,i] = 0 # that's the point

                # now the rest is modified once, either by left or right multiplication


            end
        end
    end
    
end