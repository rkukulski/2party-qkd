using Convex, SCS
using LinearAlgebra
using QuantumInformation
using Combinatorics
using SparseArrays
const MOI = Convex.MOI

# Function returning a matrix for permuting elements of 
# vector defined on dims according to permutation
function permutesystems_matrix(permutation, dims)
    M = sparse(zeros( prod(dims), prod(dims)))

    for i = 1:prod(dims)
        temp = zeros(Int, length(dims))
        j = i - 1
        for idx = length(dims):-1:1
            current = mod(j, dims[idx])
            temp[idx] = current
            j = Int((j - current) // dims[idx])
        end
        j = temp[permutation[1]]
        for idx = 2:length(dims)
            j = j * dims[permutation[idx]] + temp[permutation[idx]]
        end
        j = j+1
        M[j, i] = 1
    end
    return M
end

# Analitical version of sigma
tau0 = proj([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
tau1 = proj([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
tau2 = proj([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
tau3 = proj([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
tau4 = proj([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
sigma = 1//12*I(16) -( 1//15*(tau0 + tau4) + 1//60*(tau1 + tau3) + 1//90*tau2)
sigma = Array{ComplexF64}(sigma)
sqrt_sigma = sigma^(0.5)
v_sigma = vec(transpose(sqrt_sigma))
v_sigma_abc = (permutesystems_matrix([4, 1, 2, 3], [2, 2, 2, 2]) ⊗ I(16)) * v_sigma
v_sigma_bca = (permutesystems_matrix([4, 2, 1, 3], [2, 2, 2, 2]) ⊗ I(16)) * v_sigma

rho_sigma_abc = v_sigma_abc*v_sigma_abc'
rho_sigma_bca = v_sigma_bca*v_sigma_bca'

rho_sigma_bc_0 = ptrace(rho_sigma_abc , [2, 2, 64], [2])
rho_sigma_bc_1 = ptrace(rho_sigma_bca , [2, 2, 64], [2])

trace_distance(rho_sigma_bc_0, rho_sigma_bc_1)

rho_sigma_b_0_example1 = ptrace(rho_sigma_bc_0, [2,2,2,2,2,2,2], [3])
rho_sigma_b_1_example1 = ptrace(rho_sigma_bc_1, [2,2,2,2,2,2,2], [3])
rho_sigma_c_0_example1 = ptrace(rho_sigma_bc_0, [2,2,2,2,2,2,2], [2])
rho_sigma_c_1_example1 = ptrace(rho_sigma_bc_1, [2,2,2,2,2,2,2], [2])

trace_distance(rho_sigma_b_0_example1, rho_sigma_b_1_example1)
trace_distance(rho_sigma_c_0_example1, rho_sigma_c_1_example1)

# version 2

tau0 = proj([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
tau1 = proj([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
tau2 = proj([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])
tau3 = proj([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
tau4 = proj([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
sigma = 1//12*I(16) -( 1//15*(tau0 + tau4) + 1//60*(tau1 + tau3) + 1//90*tau2)
sigma = Array{ComplexF64}(sigma)
sqrt_sigma = sigma^(0.5)
v_sigma = vec(transpose(sqrt_sigma))

# List of system permutation matrices M_π, where the order of systems is
# S, A_I, B_I, C_I
L = [
    permutesystems_matrix([4, 1, 2, 3], [2, 2, 2, 2]),
    permutesystems_matrix([3, 1, 4, 2], [2, 2, 2, 2]),
    permutesystems_matrix([4, 3, 1, 2], [2, 2, 2, 2]),
    permutesystems_matrix([2, 4, 1, 3], [2, 2, 2, 2]),
    permutesystems_matrix([3, 4, 2, 1], [2, 2, 2, 2]),
    permutesystems_matrix([2, 3, 4, 1], [2, 2, 2, 2])
]

L_states = [(P ⊗ I(16))*v_sigma for P in L]
L_rhos = [x*x' for x in L_states]

# 

Omegas = [ComplexVariable(2^8, 2^8) for _=1:6]
constraints = [omega in :SDP for omega in Omegas]
constraints += [sum(Omegas) == I(2^8)]

f = real(sum(tr(Omegas[i]*L_rhos[i]) for i=1:6))
    
problem = maximize(f, constraints)
solve!(
    problem,
    MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
    silent_solver = true
)
problem.optval