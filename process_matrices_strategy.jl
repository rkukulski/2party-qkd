using Convex, SCS
using LinearAlgebra
using QuantumInformation
using Combinatorics
using SparseArrays
using Plots
const MOI = Convex.MOI

# Ai Ao Bi Bo

function Alice(a)
    return [proj(ket(1,2) ⊗ ket(a+1, 2)), proj(ket(2,2) ⊗ ket(a+1, 2))]
end

function Bob(b, b2)
    if b2 == 1
        return [proj(ket(1,2) ⊗ ket(b+1, 2)), proj(ket(2,2) ⊗ ket(b+1, 2))]
    end
    if b2 == 0
        ps = [1,1]/sqrt(2)
        ms = [1,-1]/sqrt(2)
        return [proj(ps ⊗ ket(b+1, 2)), proj(ms ⊗ ket(2-b, 2))]
    end
end

function Bob_S(b, b2)
    if b2 == 1
        return [ket(b+1,2)*bra(1,2), ket(b+1,2)*bra(2,2)]
    end
    if b2 == 0
        ps = [1,1]/sqrt(2)
        ms = [1,-1]/sqrt(2)
        return [ket(b+1,2)*ps', ket(2-b,2)*ms']
    end
end

function sdp_test(Q, p)

    # Eve (BI_in ⊗ BI_out), (e ⊗ z ⊗ b2)
    Eve = [[[ComplexVariable(4,4) for _=1:2] for _=1:2] for _=1:2]
    constraints = [X in :SDP for X in Eve[1][1]]
    constraints += [X in :SDP for X in Eve[1][2]]
    constraints += [X in :SDP for X in Eve[2][1]]
    constraints += [X in :SDP for X in Eve[2][2]]
    constraints += [sum(Eve[e][1][1] for e=1:2) == sum(Eve[e][1][2] for e=1:2)]
    constraints += [sum(Eve[e][1][2] for e=1:2) == sum(Eve[e][2][1] for e=1:2)]
    constraints += [sum(Eve[e][2][1] for e=1:2) == sum(Eve[e][2][2] for e=1:2)]
    constraints += [partialtrace(sum(Eve[e][z][b2] for e=1:2, z=1:2, b2=1:2)/4,2, [2,2]) == I(2)]


    WCNS = 1/4 * (I(16) + 1/sqrt(2) * (I(2) ⊗ sz ⊗ sz ⊗ I(2) + sz ⊗ I(2) ⊗ sx ⊗ sz))
    cond_prob(a,b,b2,x,y,e) = tr(
        WCNS * (Alice(a)[x+1] ⊗ partialtrace((Eve[e+1][rem((1-b2)*y+b,2)+1][b2+1] ⊗ I(2)) * (I(2) ⊗ Bob(b, b2)[y+1]),
        2, [2, 2, 2]))
    )

    Pab = 1/8*real(
        sum(
            sum(cond_prob(a,b,0,b,y,e) for y in [0,1], e in [0,1])+
            sum(cond_prob(a,b,1,x,a,e) for x in [0,1], e in [0,1])
            for a in [0,1], b in [0,1]))
    constraints += [Pab >= Q]

    if p=="a"
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,e,y,e) for y in [0,1], e in [0,1])+
                sum(cond_prob(a,b,1,x,y,a) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1]))
    elseif p=="b"
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,x,y,b) for x in [0,1], y in [0,1])+
                sum(cond_prob(a,b,1,x,y,y) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1]))
    end
    
    f = Pep
    problem = maximize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )
    print("$(problem.status): $(problem.optval) \n")
    return problem.optval
end

sdp_test((2+sqrt(2))/4-0.00005, "a")
sdp_test_EVE(0.8, "a")

function results()
    ListQ = 0.78865:0.0001:((2+sqrt(2))/4)
    ListR = [max(sdp_test(Q, "a"), sdp_test(Q, "b")) for Q in ListQ]
    return ListQ, ListR
end

function save_and_draw()
    ListQ, ListR = results()
    open("process_matrices_strategy.txt","w") do io
        println(io, "$(ListQ),\n$(ListR)")
    end
    Pl=plot(ListQ, ListQ, color="red", ls=:dash, lw=:1.5, aspect_ratio=((2+sqrt(2))/4-0.788)/((2+sqrt(2))/4-0.5), label = "\$\\mathbb{P}(K^A=K^B)\$")
    plot!(Pl, ListQ, ListR, color="blue", lw=:1.5, label = "\$ \\max \\{ \\mathbb{P}(K^A=K^E), \\mathbb{P}(K^B=K^E) \\}\$")
    plot!(Pl, ListQ, [0.78865 for i in eachindex(ListQ)], color="gray", lw=0.3, ls=:dot, label=false)
    plot!(Pl, ListQ, [0.8334 for i in eachindex(ListQ)], color="gray", lw=0.3, ls=:dot, label=false)
    plot!(Pl, ListQ, [0.6664 for i in eachindex(ListQ)], color="gray", lw=0.3, ls=:dot, label=false)
    
    xlims!(Pl, 0.7887, (2+sqrt(2))/4)
    xticks!(Pl, [0.7887, 0.8334, (2+sqrt(2))/4], ["0.7887", "0.8334", "0.8535"])
    ylims!(Pl, 0.5, (2+sqrt(2))/4)
    yticks!(Pl, [0.5, 0.6664, 0.7887, 0.8334, (2+sqrt(2))/4], ["0.5", "0.6664", "0.7887", "0.8334", "0.8535"])
    xlabel!("\$Q\$", fontsize=20)
    ylabel!("\$\\mathbb{P}\$", fontsize=20)
    savefig(Pl,"process_matrices_strategy.pdf")
end

# save_and_draw()

###########################

function sdp_test_EVE(Q, p)

    # Eve (BI_in ⊗ BI_out), (e ⊗ z ⊗ b2)
    Eve = [[[ComplexVariable(4,4) for _=1:2] for _=1:2] for _=1:2]
    constraints = [X in :SDP for X in Eve[1][1]]
    constraints += [X in :SDP for X in Eve[1][2]]
    constraints += [X in :SDP for X in Eve[2][1]]
    constraints += [X in :SDP for X in Eve[2][2]]
    constraints += [sum(Eve[e][1][1] for e=1:2) == sum(Eve[e][1][2] for e=1:2)]
    constraints += [sum(Eve[e][1][2] for e=1:2) == sum(Eve[e][2][1] for e=1:2)]
    constraints += [sum(Eve[e][2][1] for e=1:2) == sum(Eve[e][2][2] for e=1:2)]
    constraints += [partialtrace(sum(Eve[e][z][b2] for e=1:2, z=1:2, b2=1:2)/4,2, [2,2]) == I(2)]


    WCNS = 1/4 * (I(16) + 1/sqrt(2) * (I(2) ⊗ sz ⊗ sz ⊗ I(2) + sz ⊗ I(2) ⊗ sx ⊗ sz))
    cond_prob(a,b,b2,x,y,e) = tr(
        WCNS * (Alice(a)[x+1] ⊗ partialtrace((Eve[e+1][rem((1-b2)*y+b,2)+1][b2+1] ⊗ I(2)) * (I(2) ⊗ Bob(b, b2)[y+1]),
        2, [2, 2, 2]))
    )

    Pab = 1/8*real(
        sum(
            sum(cond_prob(a,b,0,b,y,e) for y in [0,1], e in [0,1])+
            sum(cond_prob(a,b,1,x,a,e) for x in [0,1], e in [0,1])
            for a in [0,1], b in [0,1]))
    constraints += [Pab >= Q]

    if p=="a"
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,e,y,e) for y in [0,1], e in [0,1])+
                sum(cond_prob(a,b,1,x,y,a) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1]))
    elseif p=="b"
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,x,y,b) for x in [0,1], y in [0,1])+
                sum(cond_prob(a,b,1,x,y,y) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1]))
    end
    
    f = Pep
    problem = maximize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )
    print("$(problem.status): $(problem.optval) \n")
    return problem.optval, [[[Eve[e][z][b2].value for b2=1:2] for z=1:2] for e=1:2]
end

Q = 0.8334
sol = sdp_test_EVE(Q, "b")
Eve = sol[2]

WCNS = 1/4 * (I(16) + 1/sqrt(2) * (I(2) ⊗ sz ⊗ sz ⊗ I(2) + sz ⊗ I(2) ⊗ sx ⊗ sz))
cond_prob(a,b,b2,x,y,e) = tr(
    WCNS * (Alice(a)[x+1] ⊗ partialtrace((Eve[e+1][rem((1-b2)*y+b,2)+1][b2+1] ⊗ I(2)) * (I(2) ⊗ Bob(b, b2)[y+1]),
    2, [2, 2, 2]))
)

Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,x,y,b) for x in [0,1], y in [0,1])+
                sum(cond_prob(a,b,1,x,y,y) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1])
)

b0 = maximum([real(sum(cond_prob(a,b,0,x,y,b) for x in [0,1], y in [0,1])) for a in [0,1], b in [0,1]])
b1 = maximum([real(sum(cond_prob(a,b,1,x,y,y) for x in [0,1], y in [0,1])) for a in [0,1], b in [0,1]])

[real(sum(cond_prob(a,b,b2,x,y,e) for x in [0,1], y in [0,1], e in [0,1])) for a in [0,1], b in [0,1], b2 in [0,1] ]


##########################################

function sdp_test_EVE_2(Q, p)
    # Eve (BI_in ⊗ BI_out ⊗ BI_in2 ⊗ BI_out2), (e ⊗ z ⊗ zz ⊗ b2)
    Eve = [[[[ComplexVariable(16,16) for _=1:2] for _=1:2] for _=1:2] for _=1:2]

    constraints = [X in :SDP for X in Eve[1][1][1]]
    constraints += [X in :SDP for X in Eve[1][2][1]]
    constraints += [X in :SDP for X in Eve[2][1][1]]
    constraints += [X in :SDP for X in Eve[2][2][1]]
    constraints += [X in :SDP for X in Eve[1][1][2]]
    constraints += [X in :SDP for X in Eve[1][2][2]]
    constraints += [X in :SDP for X in Eve[2][1][2]]
    constraints += [X in :SDP for X in Eve[2][2][2]]

    constraints += [sum(Eve[e][1][1][1] for e=1:2) == sum(Eve[e][1][2][1] for e=1:2)]
    constraints += [sum(Eve[e][1][2][1] for e=1:2) == sum(Eve[e][2][1][1] for e=1:2)]
    constraints += [sum(Eve[e][2][1][1] for e=1:2) == sum(Eve[e][2][2][1] for e=1:2)]
    
    constraints += [sum(Eve[e][1][1][1] for e=1:2) == sum(Eve[e][1][1][2] for e=1:2)]

    constraints += [sum(Eve[e][1][1][2] for e=1:2) == sum(Eve[e][1][2][2] for e=1:2)]
    constraints += [sum(Eve[e][1][2][2] for e=1:2) == sum(Eve[e][2][1][2] for e=1:2)]
    constraints += [sum(Eve[e][2][1][2] for e=1:2) == sum(Eve[e][2][2][2] for e=1:2)]


    temp = sum(Eve[e][z][zz][b2] for e=1:2, z=1:2, zz=1:2, b2=1:2)/8
    constraints += [partialtrace(temp,4, [2,2,2,2]) == partialtrace(temp,3, [2,2,4])/2 ⊗ I(2)]
    temp2 = partialtrace(temp,3, [2,2,4])/2
    constraints += [partialtrace(temp2,2, [2,2]) == I(2)]

    # WCNS = 1/4 * (I(16) + 1/sqrt(2) * (I(2) ⊗ sz ⊗ sz ⊗ I(2) + sz ⊗ I(2) ⊗ sx ⊗ sz))
    # WCNSrev = 1/4 * (I(16) + 1/sqrt(2) * (sz ⊗ I(2) ⊗ I(2) ⊗ sz + sx ⊗ sz ⊗ sz ⊗ I(2)))

    # cond_prob(a,b,b2,x,xx,y,yy,e) = tr(
    #     (WCNS ⊗ WCNSrev) * (Alice(a)[x+1] ⊗ ((I(2) ⊗ Bob_S(b, b2)[y+1] ⊗ I(2) ⊗ Bob_S(b, b2)[yy+1])*
    #     Eve[e+1][rem((1-b2)*y+b,2)+1][rem((1-b2)*yy+b,2)+1][b2+1]*(I(2) ⊗ Bob_S(b, b2)[y+1] ⊗ I(2) ⊗ Bob_S(b, b2)[yy+1])') 
    #      ⊗ Alice(a)[xx+1])
    # )

    ps = [1,1]/sqrt(2)
    ms = [1,-1]/sqrt(2)
    bs = [[ps*ps', ms*ms'],[ket(1,2)*bra(1,2), ket(2,2)*bra(2,2)]]
    W(a,b,b2,x,y) = 1/4 * (I(2) ⊗ bs[b2+1][y+1] + 1/sqrt(2) * 
    ( (-1)^a * sz ⊗ bs[b2+1][y+1] + (-1)^(x+b+y*(1-b2)) * sx ⊗ bs[b2+1][y+1]))

    cond_prob(a,b,b2,x,xx,y,yy,e) = tr(
        (W(a,b,b2,x,y) ⊗ W(a,b,b2,xx,yy)) * Eve[e+1][rem((1-b2)*y+b,2)+1][rem((1-b2)*yy+b,2)+1][b2+1]
    )

    Pab = 1/8*real(sum(
        sum(cond_prob(a,b,0,b,xx,y,yy,e) for y in [0,1], e in [0,1], xx in [0,1], yy in [0,1])+
        sum(cond_prob(a,b,1,x,xx,a,yy,e) for x in [0,1], e in [0,1], xx in [0,1], yy in [0,1])
        for a in [0,1], b in [0,1])+sum(
        sum(cond_prob(a,b,0,x,b,y,yy,e) for y in [0,1], e in [0,1], x in [0,1], yy in [0,1])+
        sum(cond_prob(a,b,1,x,xx,y,a,e) for x in [0,1], e in [0,1], xx in [0,1], y in [0,1])
        for a in [0,1], b in [0,1]))

    constraints += [Pab >= 2*Q]

    if p=="a"
        # TO DO CASE a AS WELL
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,e,y,e) for y in [0,1], e in [0,1])+
                sum(cond_prob(a,b,1,x,y,a) for x in [0,1], y in [0,1])
                for a in [0,1], b in [0,1]))
    elseif p=="b"
        Pep = 1/8*real(
            sum(
                sum(cond_prob(a,b,0,x,xx, y, yy, b) for x in [0,1], y in [0,1], xx in [0,1], yy in [0,1])+
                sum(cond_prob(a,b,1,x,xx,y,y,y) for x in [0,1], xx in [0,1], y in [0,1])+
                sum(cond_prob(a,b,1,x,xx,y,1-y,e) for x in [0,1], xx in [0,1], y in [0,1], e in [0,1])/2
                for a in [0,1], b in [0,1]))
    end

    f = Pep
    problem = maximize(f, constraints)

    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = false
    )
    print("$(problem.status): $(problem.optval) \n")
    return problem.optval
end

sdp_test_EVE_2(0.8334, "b")

wN = 0.7363745240769123 
wS = 0.666

ps1 = sum(binomial(17,er)*wS^(17-er)*(1-wS)^er for er=0:8)
ps2 = wS * sum(binomial(8,er)*wN^(8-er)*(1-wN)^er for er=0:4) +
(1-wS) * sum(binomial(8,er)*wN^(8-er)*(1-wN)^er for er=0:3)

sdp_test_EVE_2((2+sqrt(2))/4-0.00005, "b")