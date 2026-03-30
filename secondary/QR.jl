using LinearAlgebra

# modified Gram Schmidt
function mgs_inplace(A)

    m, n = size(A)
    R = zeros(eltype(A), n, n)

    for i = 1:n-1
        R[i,i] = norm(A[:, i])
        A[:,i] = A[:,i]/R[i,i] # q_i
        R[i,i+1:n] = A[:,i]' * A[:,i+1:n] # q_i'a_j for j greater than i
        A[:,i+1:n] = A[:,i+1:n] -  A[:,i] * transpose(R[i, i+1:n]) # remove the composant along q_i from all a_j
    end

    R[n,n] = norm(A[:, n])
    A[:,n] = A[:,n]/R[n,n]

    return A, R
end

@time Q, R = mgs_inplace(A)


# Householder

function house(A)

    m, n = size(A)
    R = copy(A)
    W = zeros(eltype(A), m, n)

    for k = 1:n
        v = copy(R[k:m,k])
        if v[1] == 0
            v[1] += norm(v)
        else
            v[1] += v[1]/abs(v[1]) * norm(v)
        end
        v = v / norm(v)

        R[k:m,k:n] = R[k:m,k:n] - 2v*v'*R[k:m,k:n]
        W[k:m,k] = v
    end

    return W,R
end

function formQ(W)
    m, n = size(W)
    Q = Matrix{eltype(W)}(I, m, m)

    for k = n:-1:1
        wk = W[k:m, k]
        Q[k:m, :] .-= 2 .* wk * (wk' * Q[k:m, :])
    end

    return Q
end

A = rand(ComplexF64, 5, 5)

W, R = house(A)

Q = formQ(W)

println("Error in reconstruction: ", norm(Q*R - A))

# Check if Q is orthonormal
println("norm(Q'*Q - I): ", norm(Q'*Q - I(size(Q,2))))
println("max deviation from orthonormality: ", maximum(abs.(Q'*Q - I(size(Q,2)))))


# classical Gram Schmidt

function clgs(A)
    m, n = size(A)
    Q = zeros(eltype(A), m, n)
    R = zeros(eltype(A), n, n)

    for j in 1:n
        # Classical GS uses the ORIGINAL column
        for i in 1:j-1
            R[i, j] = dot(Q[:, i], A[:, j])
        end

        v = A[:, j] - Q[:, 1:j-1] * R[1:j-1, j]
        R[j, j] = norm(v)
        Q[:, j] = v / R[j, j]
    end

    return Q, R
end

A = rand(ComplexF64, 1000, 1000)

Q, R = clgs(A)

println("Error in reconstruction: ", norm(Q*R - A))

# Check if Q is orthonormal
println("norm(Q'*Q - I): ", norm(Q'*Q - I(size(Q,2))))
println("max deviation from orthonormality: ", maximum(abs.(Q'*Q - I(size(Q,2)))))