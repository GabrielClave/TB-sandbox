# reduction to Hessemberg form
using LinearAlgebra

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


