function symmetric_tridiagonal_reduction!(A)
    m = size(A, 1)
    T = eltype(A)
    
    for k in 1:m-2
        x = @view A[k+1:m, k]
        nx = norm(x)
        
        if nx > 1e-15 # if the column/row is already zero we have nothing to do
            s = sign(x[1]) == 0 ? one(T) : sign(x[1])
            v = copy(x)
            v[1] += s * nx
            normalize!(v)
            
            # Leveraging Symmetry: QAQ* = (I−2vv*)A(I−2vv*) = A - wq* - qw*
            # 1. Compute w = 2 * A_sub * v
            sub_A = @view A[k+1:m, k+1:m]
            # We use 'Symv' (Symmetric Matrix-Vector product)
            # This only reads the lower triangle of sub_A
            w = 2 .* (Symmetric(sub_A, :L) * v)
            
            # 2. Compute q = w - (v'w)v
            q = w - (v' * w) .* v
            
            # 3. Symmetric Rank-2 Update: sub_A -= (v*q' + q*v')
            # This is the key optimization for symmetric matrices
            low_trailing = @view A[k+1:m, k+1:m]
            p = length(q)
            for j in 1:p, i in j:p
                current_entry = low_trailing[i, j] - (w[i]*q[j] + q[i]*w[j])
                low_trailing[i, j] = current_entry
                low_trailing[j, i] = current_entry
            end
            
            # Clean up the known zeros/values
            A[k+1, k] = -s * nx
            A[k, k+1] = -s * nx # Keep it symmetric
            A[k+2:m, k] .= 0
            A[k, k+2:m] .= 0
        end
    end
    return A
end

n = 6
X = randn(n, n)
A_orig = X + X'
A = copy(A_orig)
A2 = copy(A_orig)
issymmetric(A)

T = symmetric_tridiagonal_reduction!(A)
T2 = tridiagonal_reduction!(A2)

issymmetric(T) #false !
T ≈ T' # True !
eigvals(A_orig) ≈ eigvals(T)


