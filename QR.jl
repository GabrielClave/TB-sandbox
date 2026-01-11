function mgs(A)

    m, n = size(A)
    R = zeros(ComplexF64, n, n)
    Q = copy(A)

    for i = 1:n-1
        R[i,i] = norm(Q[:, i])
        Q[:,i] = Q[:,i]/R[i,i] # q_i
        R[i,i+1:n] = Q[:,i]' * Q[:,i+1:n] # q_i'a_j for j greater than i
        Q[:,i+1:n] = Q[:,i+1:n] -  Q[:,i] * transpose(R[i, i+1:n]) # remove the composant along q_i from all a_j
    end

    R[n,n] = norm(Q[:, n])
    Q[:,n] = Q[:,n]/R[n,n]

    return Q, R
end

using LinearAlgebra

# random complex matrix
A = rand(ComplexF64, 5, 5)

Q, R = mgs(copy(A))  # copy if you want to preserve A

# Check if Q*R ≈ A
println("Error in reconstruction: ", norm(Q*R - A))

# Check if Q is orthonormal
println("norm(Q'*Q - I): ", norm(Q'*Q - I(size(Q,2))))

println("max deviation from orthonormality: ", maximum(abs.(Q'*Q - I(size(Q,2)))))

A = rand(ComplexF64, 1_000, 1_000)

@time Q, R = mgs(A)  # copy if you want to preserve A

function mgs_inplace(A)

    m, n = size(A)
    R = zeros(ComplexF64, n, n)

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