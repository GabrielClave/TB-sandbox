using Plots, LinearAlgebra

# 1. Définition d'une matrice asymétrique pour bien voir l'effet
# Ligne 1 : centre 0, rayon 4
# Ligne 2 : centre 10, rayon 1
A = [0.0  4.0;
     1.0  4]

λs = eigvals(A)
centers = [A[1,1], A[2,2]]
radii = [abs(A[1,2]), abs(A[2,1])]

# Préparation de la grille pour le contour de Cassini
x_range = range(-6, 14, length=200)
y_range = range(-6, 6, length=200)

# Fonction de l'ovale : |z - a11| * |z - a22| - R1*R2
cassini(z) = abs(z - centers[1]) * abs(z - centers[2]) - (radii[1] * radii[2])

# 2. Plot
p = plot(title="Gerschgörin vs Cassini", xlabel="Re", ylabel="Im", aspect_ratio=:equal, legend=:outertopright)

# Dessin des disques de Gerschgörin (cercles)
θ = range(0, 2π, length=100)
plot!(p, centers[1] .+ radii[1]*cos.(θ), radii[1]*sin.(θ), label="Disque 1", color=:blue, lw=2)
plot!(p, centers[2] .+ radii[2]*cos.(θ), radii[2]*sin.(θ), label="Disque 2", color=:red, lw=2)

# Dessin de l'ovale de Cassini (Contour f(z) = 0)
contour!(p, x_range, y_range, (x,y) -> cassini(x + im*y), levels=[0], color=:black, lw=3, label="Ovale de Cassini")

# Ajout des valeurs propres
scatter!(p, real.(λs), imag.(λs), color=:green, markersize=6, label="Valeurs propres")

display(p)

