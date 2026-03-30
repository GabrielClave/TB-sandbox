using LinearAlgebra

# GMRES using a stopping criteria
# using Givens rotation

function incremental_gmres!(A, b, Q, H, x; maxiter = 50, tol = 1e-12)
    β = norm(b)
    @views Q[:, 1] .= b ./ β

    # g tracks the "rotated" right-hand side (initially β*e1)
    g = zeros(eltype(b), maxiter + 1)
    g[1] = β
    
    # givens rotation
    cs = zeros(maxiter)
    sn = zeros(maxiter)

    for k in 1:maxiter
        qk = view(Q, :, k)
        v  = view(Q, :, k+1)
        mul!(v, A, qk)

        for i in 1:k
            qi = view(Q, :, i)
            H[i, k] = dot(qi, v)
            v .-= H[i, k] .* qi
        end
        H[k+1, k] = norm(v)
        v ./= H[k+1, k]

        # Apply previous rotations (n-1) to the new column of H
        for i in 1:k-1
            h_old = H[i, k]
            H[i, k]   =  cs[i] * h_old + sn[i] * H[i+1, k]
            H[i+1, k] = -sn[i] * h_old + cs[i] * H[i+1, k]
        end

        # compute the new givens rotation to make H triangular
        r = sqrt(H[k,k]^2 + H[k+1,k]^2)
        cs[k] = H[k,k] / r
        sn[k] = H[k+1,k] / r

        H[k,k] = r
        H[k+1,k] = 0.0

        # apply the new rotation to the right-hand side vector g
        g_old = g[k]
        g[k] = cs[k]*g_old + sn[k]*g[k+1]
        g[k+1] = -sn[k]*g_old + cs[k]*g[k+1]
        
        residual = abs(g[k+1]) # The residual is the last element of the transformed g!
        
        if residual < tol
            # ready to solve by back-substitution
            H_k = @view H[1:k, 1:k]
            y = UpperTriangular(H_k) \ g[1:k]
            
            mul!(x, view(Q, :, 1:k), y) # x = Q_k * y
            println("Converged in $k steps. Residual: $residual")
            return k
        end
    end

    # If we hit maxiter, solve anyway
    H_k = @view H[1:maxiter, 1:maxiter]
    y = UpperTriangular(H_k) \ g[1:maxiter]
    mul!(x, view(Q, :, 1:maxiter), y)
    println("hit $maxiter iterations. Residual: $residual")
    return maxiter
end

x = zeros(m)

k = incremental_gmres!(A,b,Q,H,x) # 19 steps

println("Relative Error: ", norm(x_true - x) / norm(x_true)) # excellent