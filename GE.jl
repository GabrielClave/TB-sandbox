using LinearAlgebra
using BenchmarkTools

# standard GE, no pivot

function ge_nopivot(A)
    m, n = size(A)
    L = Matrix{Float64}(I, m, n)
    U = copy(float(A))

    for k in 1:min(m, n) -1
        pivot = U[k,k]
        v = U[(k+1):m,k]/pivot
        w = U[k,(k+1):n]'
        U[(k+1):m,(k+1):n] .-= v*w # Schur complement

        U[(k+1):m,k] .= 0
        L[(k+1):m,k] = v
    end

    return L, U
end

A = [ 5.0 0 1
    0 10 2
    1 2 3]

L, U = ge_nopivot(A)

println("norm(LU - A): ", norm(L*U - A))

# GE with partial (row) pivoting

function ge_pivot(A)
    m, n = size(A)
    L = Matrix{Float64}(I, m, n)
    U = copy(float(A))
    P = Matrix{Float64}(I, m, n)

    for k in 1:min(m, n) -1

        column_segment = abs.(U[k:m, k])
        relative_idx = argmax(column_segment)
        pivot_row = relative_idx + k - 1 # actual index in A

        U[[k, pivot_row], k:n] = U[[pivot_row, k], k:n]
        L[[k, pivot_row], 1:k-1] = L[[pivot_row, k], 1:k-1]
        P[[k, pivot_row], :] = P[[pivot_row, k], :]
        
        pivot = U[k,k]
        v = U[(k+1):m,k]/pivot
        w = U[k,(k+1):n]'
        U[(k+1):m,(k+1):n] .-= v*w

        U[(k+1):m,k] .= 0
        L[(k+1):m,k] = v
    end

    return L, U, P
end

# Generate random L and U
n = 5
L_true = tril(randn(n, n), -1) + I 
U_true = triu(randn(n, n))

A = L_true * U_true

L_comp, U_comp, P_comp = ge_pivot(A)

residual_error  = norm(P_comp * A - L_comp * U_comp)

println("Residual (||PA - LU||): ", residual_error)

# 1. Setup
m = 1000
Z = randn(m, m)
A = Z' * Z
b = randn(m)

# (a/b) Baseline: HPD Matrix
# We wrap A in 'Hermitian' to ensure Julia uses the Cholesky path immediately
A_hpd = Hermitian(A)
t_hpd = @belapsed $A_hpd \ $b
println("(a/b) HPD (Cholesky path):         $(round(t_hpd, digits=4))s")

# (c) Break Symmetry (Forces LU)
A_nonsym = copy(A)
A_nonsym[m, 1] = A_nonsym[m, 1] / 2
t_lu = @belapsed $A_nonsym \ $b
println("(c) Non-Symmetric (LU path):      $(round(t_lu, digits=4))s (Expected ~2x slower)")

# (f) Triangular (Back-substitution)
A_tri = UpperTriangular(A)
t_tri = @belapsed $A_tri \ $b
println("(f) Upper Triangular (O(m^2)):    $(round(t_tri, digits=6))s (Expected much faster)")

# (g) Break Triangular
A_broken_tri = Matrix(A_tri)
A_broken_tri[m, 1] = A_broken_tri[1, m]
t_broken_tri = @belapsed $A_broken_tri \ $b
println("(g) Broken Triangular (Back to LU): $(round(t_broken_tri, digits=6))s")