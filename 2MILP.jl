# ---- Packages to use
using vOptGeneric, JuMP, GLPK
using Combinatorics
using Printf
using LinearAlgebra
using LazySets


#=
backend = :pyplot
if backend == :pyplot
    using PyPlot
end
=#

backend = :plots
if backend == :plots
    using Plots
end


# =============================================================================
# loading a instance of 2-MILP
function loadInstance2MILP()
#=
nZ  = 2;  nR = 3;  nK = 4
c1Z = [1, 6]
c1R = [3, 1, 10]
c2Z = [5, 8]
c2R = [7, 9, 3]
A = [8 21 21 7 16; 19 17 18 21 16; 11 17 11 17 13; 18 11 24 15 17]
d = [84, 81, 83, 63]
=#

#=
nZ  = 2;  nR = 3;  nK = 4
c1Z = [3, 10]
c1R = [3, 5, 6]
c2Z = [2, 6]
c2R = [10, 6, 6]
A = [8 11 13 16 18; 13 11 17 15 8; 24 7 5 22 15; 18 5 5 19 25]
d = [93, 86, 54, 92]
=#

#=
nZ  = 2;  nR = 3;  nK = 4
c1Z = [8, 5]
c1R = [8, 10, 2]
c2Z = [4, 5]
c2R = [6, 10, 5]
A = [25 6 15 5 9; 7 11 16 17 16; 17 18 17 7 23; 15 12 5 22 19]
d = [95, 87, 98, 100]
=#

#=
nZ  = 2;  nR = 3;  nK = 4
c1Z = [5, 10]
c1R = [6, 3, 7]
c2Z = [3, 2]
c2R = [7, 8, 9]
A = [14 13 14 23 25; 6 8 24 16 5; 10 25 14 6 24; 24 25 17 16 6]
d = [60, 54, 76, 64]
=#


nZ  = 2;  nR = 3;  nK = 4
c1Z = [6, 5]
c1R = [1, 6, 9]
c2Z = [8, 5]
c2R = [9, 5, 5]
A = [23 7 23 24 18; 9 10 11 15 22; 24 12 15 19 19; 13 12 17 24 16]
d = [99, 99, 76, 78]

return c1Z, c1R, c2Z, c2R, A, d
end

# =============================================================================
# creating a random instance of 2-MILP
function createInstance2MILP(nZ::Int64, nR::Int64, nK::Int64)
    c1Z = rand(1:10,nZ)
    c1R = rand(1:10,nR)    
    c2Z = rand(1:10,nZ)
    c2R = rand(1:10,nR)  
    A  = rand(5:25,nK,nZ+nR)
    d  = rand(50:100,nK)
    return c1Z, c1R, c2Z, c2R, A, d
end

# =============================================================================
# displaying an instance
function displayInstance(nZ::Int64, nR::Int64, nK::Int64, 
                         c1Z::Array{Int,1}, c1R::Array{Int,1}, 
                         c2Z::Array{Int,1}, c2R::Array{Int,1},                             
                         A::Array{Int,2}, d::Array{Int,1})
    println("nZ  = $(nZ)  nR = $(nR)  nK = $(nK)")
    @show c1Z
    @show c1R
    @show c2Z
    @show c2R  
    @show A
    @show d  
    return nothing
end

# =============================================================================
# creating the JuMP model of a single objective MILP 
function createProblemMILP(nZ::Int64, nR::Int64, nK::Int64, 
                               c1Z::Array{Int,1}, c1R::Array{Int,1}, 
                               c2Z::Array{Int,1}, c2R::Array{Int,1},                             
                               A::Array{Int,2}, d::Array{Int,1},
                               fct::Int64)
    model = Model( GLPK.Optimizer )
    @variable(model, xZ[1:nZ] ≥0 , Int)
    @variable(model, xR[1:nR] ≥ 0)
    if fct==1
        @objective(model, Max, sum(c1Z[i]*xZ[i] for i in 1:nZ) + sum(c1R[i]*xR[i] for i in 1:nR))
    else
        @objective(model, Max, sum(c2Z[i]*xZ[i] for i in 1:nZ) + sum(c2R[i]*xR[i] for i in 1:nR))
    end
    @constraint(model, [i=1:nK], sum((A[i,j]*xZ[j]) for j in 1:nZ) + sum((A[i,nZ+j]*xR[j]) for j in 1:nR) ≤ d[i])
    return model, xZ, xR
end

# =============================================================================
# creating the JuMP model with 2 objectives MILP 
function createProblemBiMILP(nZ::Int64, nR::Int64, nK::Int64, 
                           c1Z::Array{Int,1}, c1R::Array{Int,1}, 
                           c2Z::Array{Int,1}, c2R::Array{Int,1},                             
                           A::Array{Int,2}, d::Array{Int,1})
    model = vModel( GLPK.Optimizer )
    @variable(model, xZ[1:nZ] ≥0 , Int)
    @variable(model, xR[1:nR] ≥ 0)
    @addobjective(model, Max, sum(c1Z[i]*xZ[i] for i in 1:nZ) + sum(c1R[i]*xR[i] for i in 1:nR))
    @addobjective(model, Max, sum(c2Z[i]*xZ[i] for i in 1:nZ) + sum(c2R[i]*xR[i] for i in 1:nR))
    @constraint(model, [i=1:nK], sum((A[i,j]*xZ[j]) for j in 1:nZ) + sum((A[i,nR-1+j]*xR[j]) for j in 1:nR) ≤ d[i])
    return model, xZ, xR
end

# =============================================================================
# evaluate a solution on the two objectives
function evaluateSolution2objectives(c1Z::Array{Int,1}, c1R::Array{Int,1}, 
                                     c2Z::Array{Int,1}, c2R::Array{Int,1},  
                                     xZ,  xR)
    nZ=length(c1Z)
    nR=length(c1R)
    f1 = 0
    f2 = 0
    for j in 1:nZ
        f1+=  c1Z[j]*value(xZ[j])
        f2+=  c2Z[j]*value(xZ[j])
    end
    for j in 1:nR
        f1+=  c1R[j]*value(xR[j])
        f2+=  c2R[j]*value(xR[j])
    end    
    return f1, f2
end

# =============================================================================
# display the output of a bi-objective resolution
function displaySolutions(model, nZ::Int64, nR::Int64)
    Y_N = getY_N( model )
    nS = length(Y_N)
    println("Number of solutions : ",nS)
    # for all solutions obtained
    for s in 1:length(Y_N)
        @printf("%3d   ",s)
        print("[")
        # for all discrete variables
        for v in 1:nZ
            @printf("%2d",round(getvOptData(model).X_E[s][v],digits=5))
            print(" ")
        end
        print(" |  ")
        # for all continuous variables
        for v in nZ+1:nZ+nR        
            @printf("%4.2f",round(getvOptData(model).X_E[s][v],digits=5))
            print(" ")
        end
        print("]   ( ")
        # for all points
        for p in 1:2
            @printf("%5.2f",round(Y_N[s][p], digits=5))
            print(" ")
        end
        println(")")
    end
end

# =============================================================================
# plot a set of points
function plotPoints(Y_N) #, usingmarker, usingcolor)
   # for i in 1:length(Y_N)
   #     scatter(Y_N[i][1], Y_N[i][2], marker=usingmarker, color = usingcolor)
   # end

    Z1=[]; Z2=[]
    for i in 1:length(Y_N)
        push!(Z1,Y_N[i][1]); push!(Z2,Y_N[i][2]) 
    end
    scatter!([Z1],[Z2], markershape=:star5 ,color=:yellow, markersize = 6, label = "Y_N")
end

# =============================================================================
# Put together the points sharing the same values on discrete variables
function analyzePoints(model, Y_N)  #, usingmarker, color1, color2)
    if length(Y_N) == 1
        println("Analyze not possible (|Y_N|=1)")
        return nothing
    end

  # plot the first point
  println("First")
  idColor = 1
  #scatter(Y_N[1][1], Y_N[1][2], marker=usingmarker, color = setColor)
  Z1=[];Z2=[]
  push!(Z1,Y_N[1][1]); push!(Z2,Y_N[1][2]) 
  scatter!([Z1], [Z2], markershape=:star4, color = :yellow, markersize = 6, legend=false) #color1

  for i in 2:length(Y_N)
    testEgal = (
                 (round(getvOptData(model).X_E[i-1][1],digits=5) == round(getvOptData(model).X_E[i][1],digits=5) )
                 &&
                 (round(getvOptData(model).X_E[i-1][2],digits=5) == round(getvOptData(model).X_E[i][2],digits=5) )
               )

    if testEgal
        println("Sharing")
    else
        println("Not sharing")
        idColor+=1          
    end
    #scatter(Y_N[i][1], Y_N[i][2], marker=usingmarker, color = setColor)
    Z1=[];Z2=[]
    push!(Z1,Y_N[i][1]); push!(Z2,Y_N[i][2])     
    if mod(idColor,2)!=0    
        scatter!([Z1], [Z2], markershape=:star4, color = :yellow, markersize = 6)    #color1     
    else
        scatter!([Z1], [Z2], markershape=:star5, color = :blue, markersize = 6)    #color2      
    end       
   end
   scatter!([0], [0], color=:black, markersize=0)

end

# =============================================================================
function isnonnegative(x::Array{Float64, 1})
  return length( x[ x .< 0] ) == 0
end

function enumerateFeasibleBases(c1Z, c1R, c2Z, c2R, xZopt, c, A, b)
  m, n = size(A)
  @assert rank(A) == m

  polytope = (solution)[]
  nR = length(c1R)

  for b_idx in combinations(1:n, m)
    B = A[:, b_idx]
    c_B = c[b_idx]
    x_B = inv(B) * b

    print("Basis:", b_idx, "  ")
    println(" nonnegative? ", isnonnegative(x_B))
    if isnonnegative(x_B)

        print("\t xB  = [ ")
        for i in 1:length(x_B)
            print(round(x_B[i],digits=3)," ")
        end
        println("]")

        z = dot(c_B, x_B)
        println("\t obj = ", round(z,digits=3))

        xR = zeros(nR)
        for i_B in 1:m # seek all index of variables in base
            if b_idx[i_B] ≤ nR # check if it is a variable of the problem
                xR[b_idx[i_B]] = x_B[i_B] # get the value of the variable
            end
        end

        print("\t xR  = [ ")
        for i in 1:length(xR)
            print(round(xR[i],digits=3)," ")
        end
        println("]")

        point = evaluateSolution2objectives(c1Z,c1R,c2Z,c2R,xZopt,xR)
        print("\t "); @show point
        sol = solution(xZopt, xR, point[1], point[2])
        print("\t "); @show sol
        push!(polytope,sol)
    end
  end

  return polytope
end

# -----------------------------------------------------------------------------
# Identify the sub-problem when the discrete variables are set to the opt values
# -----------------------------------------------------------------------------

mutable struct solution
    xZ :: Array{Int64}    # activity for discrete variables
    xR :: Array{Float64}  # activity for continuous variables
    z1 :: Float64         # value on the first objectives
    z2 :: Float64         # value on the second objective
end 

function computePolytope(nZ, nR, nK, c1Z, c1R, c2Z, c2R, A, d, xZOpt)
    
    # the part of RHS remaining after having fixed the discrete variables
    dreduced= Array{Float64}(undef,nK)
    for i=1:nK
        dreduced[i] = d[i]- dot(A[i,1:nZ],xZOpt)
    end
    
    # the sub-matrix remaining after having fixed the discrete variables extended by a identity matrix of size nK
    Areduced = hcat(A[:,nZ+1:nZ+nR],I)
    
    # the cost vector extended by a zero vector of size nK
    Creduced = vcat(c1R,zeros(nK))
    
    # enumerate all admissible bases corresponding to the sub-problem
    return enumerateFeasibleBases(c1Z, c1R, c2Z, c2R, xZOpt, Creduced, Areduced, dreduced)
end

# -----------------------------------------------------------------------------
# create an instance
# -----------------------------------------------------------------------------
# ---- Set up the dimension of an instance to create
nZ=2; nR=3 ; nK=4
# ---- Generate randomly an instance
#c1Z, c1R, c2Z, c2R, A, d = createInstance2MILP(nZ,nR,nK)
# ---- load an instance
c1Z, c1R, c2Z, c2R, A, d = loadInstance2MILP()
# ---- Displaying the instance
displayInstance(nZ,nR,nK,c1Z,c1R,c2Z,c2R,A,d)


# -----------------------------------------------------------------------------
# optimize f1
# -----------------------------------------------------------------------------
m_f1, xZ, xR = createProblemMILP(nZ,nR,nK,c1Z,c1R,c2Z,c2R,A,d,1)
optimize!(m_f1)
if termination_status(m_f1) == OPTIMAL
    xZ_f1opt = deepcopy(value.(xZ))
    y_f1max = evaluateSolution2objectives(c1Z,c1R,c2Z,c2R,xZ,xR)
    println("maximization of f1 ")
    println("f  = $(round(objective_value(m_f1),digits=3))")
    println("xZ = $(value.(xZ))")
    println("xR = $(value.(xR))")
    println("y  = ($(round(y_f1max[1],digits=3));$(round(y_f1max[2],digits=3)))")      
end

# -----------------------------------------------------------------------------
# optimize f2
# -----------------------------------------------------------------------------
m_f2, xZ, xR = createProblemMILP(nZ,nR,nK,c1Z,c1R,c2Z,c2R,A,d,2)
optimize!(m_f2)
if termination_status(m_f2) == OPTIMAL
    xZ_f2opt = deepcopy(value.(xZ))       
    y_f2max = evaluateSolution2objectives(c1Z,c1R,c2Z,c2R,xZ,xR)     
    println("maximization of f2 ")    
    println("f  = $(round(objective_value(m_f2),digits=3))")
    println("xZ = $(value.(xZ))")
    println("xR = $(value.(xR))")
    println("y  = ($(round(y_f2max[1],digits=3));$(round(y_f2max[2],digits=3)))")    
end

# -----------------------------------------------------------------------------
# display graphically the results
# -----------------------------------------------------------------------------

# ---- Set the axes orthonormed
vmax = max(y_f1max[1], y_f1max[2], y_f2max[1], y_f2max[2])

# ---- Plot y_f1max and y_f2max
#=
if backend == :pyplot
    # ---- Initialize the graphic
    figure("Trace of a run",figsize=(6,6)) # Create a new figure
    title("2-MILP | nZ=" * string(nZ) * " nR=" * string(nR) * " nK=" * string(nK))
    xlabel(L"$f^1(x_Z,x_R),>$")
    ylabel(L"$f^2(x_Z,x_R),>$")
    xlim(0.0,vmax+5)
    ylim(0.0,vmax+5)
    scatter(y_f1max[1],y_f1max[2],color="green")
    scatter(y_f2max[1],y_f2max[2],color="red")
end
=#
if backend == :plots
    scatter([y_f1max[1]],[y_f1max[2]],  markershape = :circle, markersize = 6, color = :green, label = "y_f1max", legend=:topleft)
    scatter!([y_f2max[1]],[y_f2max[2]], markershape = :circle, markersize = 6, color = :red,   label = "y_f2max")
    xaxis!("f1(xZ,xR),>",(0,vmax+5))
    yaxis!("f2(xZ,xR),>",(0,vmax+5))    
end


# -----------------------------------------------------------------------------
# compute the polytope attached to a given xZ
# -----------------------------------------------------------------------------
polytope_f1 = computePolytope(nZ, nR, nK, c1Z, c1R, c2Z, c2R, A, d, xZ_f1opt)
Z1=[];Z2=[];points_f1=(Vector{Float64})[]
for p in 1:length(polytope_f1)
    z1 = polytope_f1[p].z1
    z2 = polytope_f1[p].z2
    println(z1, " ", z2)
    #if backend == :pyplot
    #    scatter(z1,z2, marker="x",color="green",s=80)
    #end  
    push!(Z1,z1)
    push!(Z2,z2)  
    push!(points_f1,[z1,z2])    
end
scatter!([Z1],[Z2], markershape=:xcross ,color=:green,markersize = 6, label = "pol_f1max")

polytope_f2 = computePolytope(nZ, nR, nK, c1Z, c1R, c2Z, c2R, A, d, xZ_f2opt)
Z1=[];Z2=[];points_f2=(Vector{Float64})[]
for p in 1:length(polytope_f2)
    z1 = polytope_f2[p].z1
    z2 = polytope_f2[p].z2
    println(z1, " ", z2)
    #if backend == :pyplot
    #    scatter(z1,z2, marker="+",color="red",s=100)
    #end      
    push!(Z1,z1)
    push!(Z2,z2)
    push!(points_f2,[z1,z2])       
end
scatter!([Z1],[Z2], markershape=:cross ,color=:red, markersize = 6, label = "pol_f2max")

# -----------------------------------------------------------------------------
# Compute the convex set 
# -----------------------------------------------------------------------------

hull_f1 = convex_hull(points_f1)
pf1 = plot!([Singleton(vi) for vi in hull_f1])
plot!(pf1, VPolygon(hull_f1), alpha=0.7)

hull_f2 = convex_hull(points_f2)
pf2 = plot!([Singleton(vi) for vi in hull_f2])
plot!(pf2, VPolygon(hull_f2), alpha=0.2)


# -----------------------------------------------------------------------------
# optimize f1+f2 lexicographically
# -----------------------------------------------------------------------------
#=
m_f1f2lex, xZ, xR = createProblemBiMILP(nZ,nR,nK,c1Z,c1R,c2Z,c2R,A,d)
# ---- Invoking the solver (lexicographic method)
vSolve( m_f1f2lex, method=:lex )
# ---- Querying the results
Y_N = getY_N( m_f1f2lex )
# ---- Displaying the results for lex(1,2) and lex(2,1)
displaySolutions(m_f1f2lex,nZ,nR)
# ---- Plot the points
plotPoints(Y_N)    #, "o", "green")
=#


# -----------------------------------------------------------------------------
# optimize f1+f2 with epsilon constraint
# -----------------------------------------------------------------------------
m_f1f2epsi, xZ, xR = createProblemBiMILP(nZ,nR,nK,c1Z,c1R,c2Z,c2R,A,d)
# ---- Invoking the solver (lexicographic method)
vSolve(m_f1f2epsi, method=:epsilon, step = 1.0)
# ---- Querying the results
Y_N = getY_N( m_f1f2epsi )
# ---- Displaying the results for epsilon
displaySolutions(m_f1f2epsi,nZ,nR)
# ---- Plot the points
#plotPoints(Y_N)  #, "+", "blue")

analyzePoints(m_f1f2epsi, Y_N)  #, "+", "blue", "red")

