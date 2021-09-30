# This file contains the codes to solve DC OTS problem on tested instances with heuristic "big-M" values computed via k-shortest-path method
# To do so, do the following steps:
# 1. compute k shortest path for each edge in the network (call the function: comp_kSP())
# 2. compute k values for each edge in given tested intances (call the function: compute_ksp_k_val())
# 3. run DC OTS model with k-shortest-path "big-M" values on tested instances (call the function: OTS_ksp())
using Base: Int64, Int32, String, Float32
using LinearAlgebra
using CSV, DataFrames
using JuMP, Gurobi
using Dates
using PyCall

nx = pyimport("networkx")
py"""
def copy_nx_graph(nx_graph):
    return nx_graph.copy()
"""
copy_nx_graph = py"copy_nx_graph"

# The following function execute Yen's k shortest path algorithm (1971)
function k_shortest_paths(nx_graph, distmx, fbus, tbus, K)
    k_shortest_pathS = Vector{Vector{Int64}}()
    ksp_weightS = Vector{Float64}()

    # find the shortest path
    #-------------------------------------------------------------------
    shortest_path = nx.dijkstra_path(nx_graph, fbus, tbus)
    push!(k_shortest_pathS, shortest_path)
    # the weight of the path
    ksp_edges = Vector{Tuple{Int64,Int64}}()
    for i in 1:length(shortest_path)-1
        push!(ksp_edges, (shortest_path[i],shortest_path[i+1]))
    end
    push!(ksp_weightS, sum(distmx[f,t] for (f,t) in ksp_edges))
    #--------------------------------------------------------------------
    
    # find the 2 to K shortest paths
    #---------------------------------------------------------------------
    for k in 2:K
        k_graph = copy_nx_graph(nx_graph)
        # Step 1
        #-------------------------------------------------------------------
        k_minus_one_sp = k_shortest_pathS[k-1]
        list_B_paths = Vector{Vector{Int64}}()
        list_B_weights = Vector{Float64}()
        for i in 1:length(k_minus_one_sp)-1
            # (a) find the root path
            root_path_i = k_minus_one_sp[1:i]
            for path in k_shortest_pathS
                if i < length(path) && path[1:i] == root_path_i
                    k_graph.add_edge(path[i],path[i+1],weight=Inf) 
                end
            end
            # (b) find the spur path
            spur_node = root_path_i[end]
            # make sure the spur path does not visit any node in the root path
            i_graph = copy_nx_graph(k_graph)
            for node in root_path_i[1:end-1]
                i_graph.remove_node(node)
            end

            spur_path = nx.dijkstra_path(i_graph, spur_node, tbus)
            
            #(c) join the paths
            joint_path = Vector{Int64}()
            for node in root_path_i
                push!(joint_path, node)
            end
            for node in spur_path[2:end]
                push!(joint_path, node)
            end

            if !(joint_path in list_B_paths || joint_path in k_shortest_pathS)
                push!(list_B_paths, joint_path)
                path_edges = Vector{Tuple{Int64,Int64}}()
                for h in 1:length(joint_path)-1
                    push!(path_edges, (joint_path[h], joint_path[h+1]))
                end
                push!(list_B_weights, sum(distmx[f,t] for (f,t) in path_edges)) 
            end
        end
        #-------------------------------------------------------------------
        #Step 2
        if length(list_B_paths) == 0 || minimum(list_B_weights) == Inf
            break
        end
        # find the path with the minimum weight
        p = sortperm(list_B_weights)
        list_B_weights = list_B_weights[p]
        list_B_paths = list_B_paths[p]
        min_weight = list_B_weights[1]
        ksp_path = list_B_paths[1]
 
        # move the path from list B to list A
        if !(ksp_path in k_shortest_pathS)
            push!(ksp_weightS, min_weight)
            push!(k_shortest_pathS, ksp_path)
            popfirst!(list_B_weights)
            popfirst!(list_B_paths)
        end
    end
    #------------------------------------------------------------
    return k_shortest_pathS, ksp_weightS   
end

function comp_kSP(Branch_Data, K)
    num_branch = maximum(Branch_Data.index)
    output_paths_mx = Array{String, 2}(undef, num_branch, K)
    output_path_weights_mx = Array{Float64, 2}(undef, num_branch, K)
    
    # Construct the graph
    graph = nx.Graph()
    distmx = Dict()
    for idx in Branch_Data.index
        fbus = Branch_Data.node1[idx]
        tbus = Branch_Data.node2[idx]
        graph.add_edge(fbus, tbus, weight=Branch_Data.transLimit[idx]*Branch_Data.reactance[idx])
        distmx[fbus, tbus] = Branch_Data.transLimit[idx] * Branch_Data.reactance[idx]
        distmx[tbus, fbus] = Branch_Data.transLimit[idx] * Branch_Data.reactance[idx]
    end

    # Compute k-shortest-paths for each edge
    for idx in Branch_Data.index # for each branch in the network
        fbus = Branch_Data.node1[idx]
        tbus = Branch_Data.node2[idx]
        ksp_paths, ksp_weights = k_shortest_paths(graph, distmx, fbus, tbus, K)
        # write the paths into output
        for k in 1:K
            if k <= length(ksp_paths)
                path = ksp_paths[k]
                # Convert the array to a String
                s = "$(path[1])"
                for node in path[2:end]
                    s = s * "->$node"
                end
                # update the output Matrix
                output_paths_mx[idx,k] = s
                output_path_weights_mx[idx,k] = ksp_weights[k]
            else
                output_paths_mx[idx,k] = "None"
                output_path_weights_mx[idx,k] = -1
            end
        end
    end
    return output_paths_mx, output_path_weights_mx
end

function OPF(Bus_Data, Branch_Data, Generator_Data)
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer) );
    @variable(myMod, production[Generator_Data.index], lower_bound = 0)

    newBranches = []
    num_branches = length(Branch_Data.index)
    for idx in 1:num_branches
        push!(newBranches, (idx, Branch_Data.node1[idx], Branch_Data.node2[idx]) )
    end

    power_demand = Dict{Int64, Float64}()
    for i in 1:length(Bus_Data.index)
        power_demand[Bus_Data.index[i]] = Bus_Data.demand[i]
    end

    powerFlow = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()
    for (index, node1, node2) in newBranches
        global powerFlow[index, node1, node2] = @variable(myMod)
    end

    @variable(myMod, volAng[Bus_Data.index])

    # Define constraints
    #---------------------------------------------------------------------------------------------------------
    bus_with_generators = Set(Generator_Data.bus)

    generatorInBus = Dict()
    for index in Generator_Data.index
        generatorInBus[Generator_Data.bus[index]] = index
    end

    # production limits on power plants
    @constraint(myMod, [g in Generator_Data.index], production[g] <= Generator_Data.pmax[g])

    # Network flow balance
    #---------------------------------------------------------------------------------
    flow_balance = Dict()
    for bus in Bus_Data.index
        if bus in bus_with_generators
            flow_balance[bus] = @constraint(myMod, sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in newBranches if node2_ == bus)
            - sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in newBranches if node1_ == bus)
            + production[generatorInBus[bus]]
            == power_demand[bus]
            )
        else
            flow_balance[bus] = @constraint(myMod, sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in newBranches if node2_ == bus)
            - sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in newBranches if node1_ == bus)
            == power_demand[bus]
            )
        end
    end
    #--------------------------------------------------------------------------------

    # Kirchhoff's Law
    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[index, node1, node2]
                - 1/Branch_Data.reactance[index]*(volAng[node1] - volAng[node2]) == 0)

    # Power flow limits
    #--------------------------------------------------------------------------------
    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[index, node1, node2] + Branch_Data.transLimit[index] >= 0)

    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[index, node1, node2] - Branch_Data.transLimit[index] <= 0)
    #-----------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------------
    @objective(myMod, Min, sum(production[g]*Generator_Data.cost[g] for g in Generator_Data.index))
    set_silent(myMod)
    optimize!(myMod)

    if termination_status(myMod) == MOI.OPTIMAL
        dual_flow_balance = Dict{Int64, Float64}()
        for i in Bus_Data.index
            dual_flow_balance[i] = dual(flow_balance[i])
        end
        power_flow = Dict{Int64, Float64}()
        for (i, node1, node2) in newBranches
            power_flow[Branch_Data.index[i]] = value(powerFlow[i, node1, node2])
        end
        return termination_status(myMod), objective_value(myMod), power_flow, dual_flow_balance
    else
        return termination_status(myMod), "none", "none", "none"
    end
end

function OTS(busData, branchData, generatorData, bigM, idx_unswitchable_lines, cardinality_limit, timeLimit, gap, relaxation)
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer, "Seed" => 1, "TimeLimit" => timeLimit, "MIPGap" => gap) );
    
    power_demand = Dict{Int64, Float64}()
    for i in 1:length(busData.index)
        power_demand[busData.index[i]] = busData.demand[i]
    end

    # Define variables
    #--------------------------------------------------------------------------------
    @variable(myMod, production[generatorData.index], lower_bound = 0)

    branches = Array{Tuple{Int64,Int64,Int64},1}()
    switchable_branches = Array{Tuple{Int64,Int64,Int64},1}()
    unswitchable_branches = Array{Tuple{Int64,Int64,Int64},1}()
    for i in 1:length(branchData.index)
        push!(branches, (branchData.index[i], branchData.node1[i], branchData.node2[i]) )
        if i in idx_unswitchable_lines
            push!(unswitchable_branches, (branchData.index[i], branchData.node1[i], branchData.node2[i]) )
        else
            push!(switchable_branches, (branchData.index[i], branchData.node1[i], branchData.node2[i]) )
        end
    end

    powerFlow = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()
    for branch in branches
        global powerFlow[branch[1], branch[2], branch[3]] = @variable(myMod)
    end

    branchOpen = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()
    for branch in switchable_branches
        if relaxation
            global branchOpen[branch[1], branch[2], branch[3]] = @variable(myMod,lower_bound = 0,upper_bound = 1)
        else
            global branchOpen[branch[1], branch[2], branch[3]] = @variable(myMod, binary = true)
        end
    end

    @variable(myMod, volAng[busData.index])
    #------------------------------------------------------------------------------------

    # Define constraints
    #-----------------------------------------------------------------------------------
    # power flow balance for generator bus
    for g in generatorData.index
        bus = generatorData.bus[g]
        @constraint(myMod,
            sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in branches if node2_ == bus)
            - sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in branches if node1_ == bus)
            + production[g]
            == power_demand[bus]
            )
    end

    # power flow balance for non-generator bus
    nonGenBuses = setdiff(busData.index, generatorData.bus)
    for bus in nonGenBuses
        @constraint(myMod,
            sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in branches if node2_ == bus)
            - sum(powerFlow[index_, node1_, node2_] for (index_, node1_, node2_) in branches if node1_ == bus)
            == power_demand[bus]
            )
    end

    # Kirchhoff's Laws
    @constraint(myMod, [(index, node1, node2) in switchable_branches],
                -bigM[index]*(1-branchOpen[index, node1, node2])
                <= powerFlow[index, node1, node2]
                - (volAng[node1] - volAng[node2])/branchData.reactance[index]
                )

    @constraint(myMod, [(index, node1, node2) in switchable_branches],
                bigM[index]*(1-branchOpen[index, node1, node2])
                >= powerFlow[index, node1, node2]
                - (volAng[node1] - volAng[node2])/branchData.reactance[index]
                )
    @constraint(myMod, [(index, node1, node2) in unswitchable_branches],
                powerFlow[index, node1, node2]
                == (volAng[node1] - volAng[node2])/branchData.reactance[index]
                )
    # Power transmission limits
    @constraint(myMod, [(index, node1, node2) in switchable_branches],
                powerFlow[index, node1, node2]
                + branchOpen[index, node1, node2] * branchData.transLimit[index]
                >= 0
                )

    @constraint(myMod, [(index, node1, node2) in switchable_branches],
                powerFlow[index, node1, node2]
                - branchOpen[index, node1, node2] * branchData.transLimit[index]
                <= 0
                )
    @constraint(myMod, [(index, node1, node2) in unswitchable_branches],
                powerFlow[index, node1, node2]
                <=  branchData.transLimit[index]
                )
    @constraint(myMod, [(index, node1, node2) in unswitchable_branches],
                powerFlow[index, node1, node2]
                >=  -branchData.transLimit[index]
                )

    # Power production limits
    @constraint(myMod, [g in generatorData.index], production[g] <= generatorData.pmax[g])

    # Cardinality constraint
    @constraint(myMod, sum(1-branchOpen[edgeIdx, node1, node2] for (edgeIdx, node1, node2) in switchable_branches) <= cardinality_limit)
    #-----------------------------------------------------------------------------------

    @objective(myMod, Min, sum(production[g]*generatorData.cost[g] for g in generatorData.index))
    # set_silent(myMod)
    optimize!(myMod)
    if termination_status(myMod) != MOI.INFEASIBLE
        num_lines_open = length(switchable_branches) - sum(value(branchOpen[edgeIdx, node1, node2])  for (edgeIdx, node1, node2) in switchable_branches )
        return termination_status(myMod), objective_value(myMod), objective_bound(myMod), relative_gap(myMod), solve_time(myMod), num_lines_open
    else
        return termination_status(myMod), missing, missing, missing, solve_time(myMod), missing
    end
end

function dual_OPF_feasibility(bus_data, branch_data, gen_data)
    # build constraint matrix and vectors
    n = length(bus_data.index) # number of nodes
    m = length(branch_data.index) # number of edges

    A = zeros(n,m) # incidence matrix
    for i in 1:m
        A[branch_data.node1[i], i] = -1
        A[branch_data.node2[i], i] = 1
    end

    B = zeros(m,n) # susceptance matrix
    for i in 1:m
        B[i,branch_data.node1[i]] = 1/branch_data.reactance[i]
        B[i,branch_data.node2[i]] = -1/branch_data.reactance[i]
    end

    eye(n) = 1.0*Matrix(I,n,n)
    A_bar = [
        eye(n) zeros(n,n) A;
        zeros(m,n) B -eye(m)
    ] # contraint matrix of the primal feasible set Ax == b

    B = [
        zeros(m,n) zeros(m,n) eye(m);
        zeros(m,n) zeros(m,n) -eye(m);
        eye(n) zeros(n,n) zeros(n,m);
        -eye(n) zeros(n,n) zeros(n,m)
    ] # constraint matrix of primal feasible set Bx >= g

    num_gen = length(gen_data.bus) # number of generators
    p_max = zeros(n)
    p_min = zeros(n)
    for i in 1:num_gen
        p_max[gen_data.bus[i]] = gen_data.pmax[i]
        p_min[gen_data.bus[i]] = gen_data.pmin[i]
    end

    demand = zeros(n)
    for i in 1:n
        demand[i] = bus_data.demand[i]
    end
    f_bar = zeros(m)
    for i in 1:m
        f_bar[i] = branch_data.transLimit[i]
    end
    b = [
        demand;
        zeros(m)
    ] # contraint Vector of the primal feasible set A_bar * x == b

    g =[
        -f_bar;
        -f_bar;
        p_min;
        -p_max
    ] # constraint Vector of primal feasible set Bx >= g

    # build the optimization model
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer) );

    @variable(myMod, u[1:n+m])
    @variable(myMod, y[1:2*m+2*n] >= 0)
    @constraint(myMod, u'*A_bar + y'*B .== zeros(2n+m)')
    
    @objective(myMod, Max, y'*g + u'*b )

    # run the optimization model
    optimize!(myMod)

    return termination_status(myMod)
end

function find_edges_for_IIS(bus_data, branch_data, gen_data)
    # build constraint matrix and vectors
    n = length(bus_data.index) # number of nodes
    m = length(branch_data.index) # number of edges

    bus_name_to_index = Dict{Int32, Int32}()
    for i in 1:length(bus_data.index)
        bus_name_to_index[bus_data.index[i]] = i
    end

    A = zeros(n,m) # incidence matrix
    for i in 1:m
        A[bus_name_to_index[branch_data.node1[i]], i] = -1
        A[bus_name_to_index[branch_data.node2[i]], i] = 1
    end

    B = zeros(m,n) # susceptance matrix
    for i in 1:m
        B[i,bus_name_to_index[branch_data.node1[i]]] = 1/branch_data.reactance[i]
        B[i,bus_name_to_index[branch_data.node2[i]]] = -1/branch_data.reactance[i]
    end

    eye(n) = 1.0*Matrix(I,n,n)
    A_bar = [
        eye(n) zeros(n,n) A;
        zeros(m,n) B -eye(m)
    ] # contraint matrix of the primal feasible set Ax == b

    B = [
        zeros(m,n) zeros(m,n) eye(m);
        zeros(m,n) zeros(m,n) -eye(m);
        eye(n) zeros(n,n) zeros(n,m);
        -eye(n) zeros(n,n) zeros(n,m)
    ] # constraint matrix of primal feasible set Bx >= g

    num_gen = length(gen_data.bus) # number of generators
    p_max = zeros(n)
    p_min = zeros(n)
    for i in 1:num_gen
        p_max[bus_name_to_index[gen_data.bus[i]]] = gen_data.pmax[i]
        p_min[bus_name_to_index[gen_data.bus[i]]] = gen_data.pmin[i]
    end

    demand = zeros(n)
    for i in 1:n
        demand[i] = bus_data.demand[i]
    end
    f_bar = zeros(m)
    for i in 1:m
        f_bar[i] = branch_data.transLimit[i]
    end
    b = [
        demand;
        zeros(m)
    ] # contraint Vector of the primal feasible set A_bar * x == b

    g =[
        -f_bar;
        -f_bar;
        p_min;
        -p_max
    ] # constraint Vector of primal feasible set Bx >= g

    # build the optimization model
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer) );

    @variable(myMod, u[1:n+m])
    @variable(myMod, y[1:2*m+2*n] >= 0)
    @constraint(myMod, u'*A_bar + y'*B .== zeros(2n+m)')
    @constraint(myMod, norm_cons, y'*g + u'*b == 1)

    w = ones(2*m+2*n)
    @objective(myMod, Min, w'*y )

    # run the optimization model
    set_silent(myMod)
    optimize!(myMod)

    # identify the edges for IIS 
    edges_in_IIS = Vector{Int32}() # indices of the edges corresponding to the Kirchhoff constraints in IIS
    if termination_status(myMod) == MOI.OPTIMAL
        for i in n+1:n+m
            if value(u[i]) > 0.00001 || value(u[i]) < -0.00001
                push!(edges_in_IIS, branch_data.index[i-n])
            end
        end
    end
    return edges_in_IIS
end

function ksp_for_DC_OTS(bus_data, branch_data, gen_data, ksp_paths, max_k, max_check_edge)
    k_val = Vector{Int64}(undef,length(branch_data.index))

    for i in branch_data.index
        j = 0
        k_found = false # boolean of whether k value for the edge has been found 
        set_branch_rm = Vector{Int64}() # indices of branches that have been removed
    
        sol_status, obj_val, power_flow, dual_flow_balance = OPF(bus_data, branch_data, gen_data)
        # dual_flow_balance is Vector or dictionary corresponding to each node in the network
        # power_flow is a Vector or dictionary corresponding to each branch in the network

        while !k_found && j < max_k
            j += 1 # consider the next shortest path

            # read the path
            path_string = ksp_paths[i,j]
            if path_string == "None"
                break
            end
            path_nodes = split(path_string, "->")
            
            branch_in_path = Vector{Int64}() # indices of branches in the current shortest path
            for i in 1:length(path_nodes)-1
                push!( branch_in_path, branch_index[parse(Int64,path_nodes[i]),parse(Int64,path_nodes[i+1])] )
            end

            # skip this path if any edge in the path has been removed
            if length( intersect(branch_in_path, set_branch_rm) )>0
                continue
            end

            # rank the edges in the path according to alpha by fuller et al. (2012)
            candi_rm = setdiff(branch_in_path, set_branch_rm) # branches in the current shortest path that have not been removed yet
            if sol_status == MOI.OPTIMAL
                alpha = Vector{Float64}()
                for idx in candi_rm
                    push!(alpha, (dual_flow_balance[branch_data.node1[idx]] - dual_flow_balance[branch_data.node2[idx]]) *power_flow[idx])
                end
                ranked_candi_rm_idx = sortperm(alpha) # indices of the ordered candidate lines with alpha from small to big 
                if length(ranked_candi_rm_idx) > max_check_edge
                    ranked_candi_rm_idx = ranked_candi_rm_idx[1:max_check_edge]
                end
            else
                ranked_candi_rm_idx = 1:length(candi_rm)
            end
            
            # check if any edge can be removed
            #-------------------------------------------------------------------------------
            feasibility_found = false 

            for ord in ranked_candi_rm_idx # indices of the branches with the smallest alpha values
                new_rm_branch = union(candi_rm[ord], set_branch_rm)
                new_branch_data = branch_data[setdiff(1:end,new_rm_branch),:]
                sol_status, obj_val, new_power_flow, new_dual_flow_balance = OPF(bus_data, new_branch_data, gen_data)
                if sol_status == MOI.OPTIMAL
                    feasibility_found = true
                    set_branch_rm = new_rm_branch
                    power_flow = new_power_flow
                    dual_flow_balance = new_dual_flow_balance
                    break
                end
            end
        
            #-----------------------------------------------------------------------------------
            if !feasibility_found # if removing one edge in the path can not result in feasibility
                # try to achieve feasibility by removing one edge outside the path
                new_rm_branch = union(candi_rm[ ranked_candi_rm_idx[1] ], set_branch_rm)
                new_branch_data = branch_data[setdiff(1:end,new_rm_branch),:]
                edges_for_IIS = find_edges_for_IIS(bus_data, new_branch_data, gen_data) # indices of the edges corresponding to the Kirchhoff constraints in IIS
                
                if power_flow == "none" # can not rank the edges from existing OPF solution
                    restore_feasibility = false
                    for idx in edges_for_IIS
                        exper_rm_branch = union(idx, new_rm_branch)
                        exper_branch_data = branch_data[setdiff(1:end,exper_rm_branch),:]
                        sol_status, prod_cost, new_power_flow, new_dual_flow_balance = OPF(bus_data, exper_branch_data, gen_data)
                        if sol_status == MOI.OPTIMAL
                            restore_feasibility = true
                            push!(set_branch_rm, idx) # remove edge outside the path for feasibility
                            push!(set_branch_rm, candi_rm[ ranked_candi_rm_idx[1] ]) # remove edge in the path 
                            power_flow = new_power_flow # update the power flow for computing alpha values
                            dual_flow_balance =  new_dual_flow_balance # update dual of flow balance constraints for computing alpha values
                            break
                        end
                    end
                    if !restore_feasibility
                        k_found = true
                    end
                else # if the edges can be ranked for the greedy heuristic
                    #rank the edges
                    #----------------------------------------------------------------
                    alpha_vec = Vector{Float64}()
                    for idx in edges_for_IIS
                        push!(alpha_vec, (dual_flow_balance[branch_data.node1[idx]] - dual_flow_balance[branch_data.node2[idx]]) * power_flow[idx])
                    end
                    ranked_idx = sortperm(alpha_vec) # indices of the ordered candidate lines with alpha from small to big 
                    ranked_edges_for_IIS = edges_for_IIS[ranked_idx]
                    #-----------------------------------------------------------------
                    restore_feasibility = false
                    for idx in ranked_edges_for_IIS
                        exper_rm_branch = union(idx, new_rm_branch)
                        exper_branch_data = branch_data[setdiff(1:end,exper_rm_branch),:]
                        sol_status, prod_cost, power_flow, dual_flow_balance = OPF(bus_data, exper_branch_data, gen_data)
                        if sol_status == MOI.OPTIMAL
                            push!(set_branch_rm, candi_rm[ ranked_candi_rm_idx[1] ])# remove edge in the path 
                            push!(set_branch_rm, idx)# remove edge outside the path
                            restore_feasibility = true
                            break
                        end
                    end
                    if !restore_feasibility
                        k_found = true
                    end
                end
            end
        end
        k_val[i] = j
    end
    return k_val
end

# run k-shortest-path procedure to compute k values
# if network == 1, use 118 bus data
# if network == 2, use 300 bus data
# if merge == true, add the results to the existing ones
function compute_ksp_k_val(network,merge,load_factors,idx_load_factors,instances,branch_data,gen_data,ksp_paths,max_pathS,max_edgeS)
    time_df = DataFrame(data_instance = String[], max_path = Int32[], max_edge = Int32[], time = Float32[])
    for l in idx_load_factors
        for i in instances
            # set up the data matrix
            #---------------------------------------------------------------------
            num_branch = length(branch_data.index)
            num_max_edge = length(max_edgeS)
            num_max_path = length(max_pathS)
            k_val_mx = Matrix{Int64}(undef, num_branch, num_max_edge*num_max_path+1)
            k_val_mx[:,1] = branch_data.index
            #---------------------------------------------------------------------

            # read the bus data
            if network == 1
                data_file = "Data/118bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE118_P_$(load_factors[l]/100)_$i"
            elseif network == 2
                data_file = "Data/300bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE300_P_$(load_factors[l]/100)_$i"
            end
            new_bus_data = DataFrame!(CSV.File(data_file))

            # run kSP procedure
            for p in 1:num_max_path, j in 1:num_max_edge
                max_path = max_pathS[p]
                max_edge = max_edgeS[j]
                    
                base_time = Dates.now()
                # k values returned by the k-shortest-path procedure
                k_val = ksp_retrieve_feasibility(2, new_bus_data, branch_data, gen_data, ksp_paths, max_path, max_edge)
                    
                prepro_time = Dates.value(Dates.now()-base_time)/1000 # in seconds
                k_val_mx[:,p*j+1] = k_val # store the k values

                time_df_row = []
                push!(time_df_row, instance)
                push!(time_df_row, max_path)
                push!(time_df_row, max_edge)
                push!(time_df_row, prepro_time)
                push!(time_df, time_df_row) # record computational time of k-shortest-path procedure
            end

            k_val_df = DataFrame(k_val_mx)
            # name the output DataFrame
            #----------------------------------------------
            df_names = Vector{String}()
            push!(df_names, "branch_index")
            for mp in max_pathS, me in max_edgeS
                push!(df_names, "max_path_$(mp)_max_edge_$(me)")
            end
            rename!(k_val_df, df_names)
            #----------------------------------------------
            
            if network == 1
                output_file = "merged_branch_118/ksp_k_values/load_$(load_factors[l])_percent_instance_$(i).csv"
            elseif network == 2
                output_file = "merged_branch_300/ksp_k_values/load_$(load_factors[l])_percent_instance_$(i).csv"
            end
            CSV.write(output_file, k_val_df)
        end
    end

    if network == 1
        time_file = "merged_branch_118/ksp_time.csv"
    elseif network == 2
        time_file = "merged_branch_300/ksp_time.csv"
    end
    if merge
        input_time_df = DataFrame!(CSV.File(time_file))
        append!(time_df, input_time_df)
    end
    CSV.write(time_file, time_df)
end

# run OTS model with k-shortest-path methods while reading k values that are already computed
# if network == 1, use 118 bus data
# if network == 2, use 300 bus data
function OTS_ksp_bigM(network, load_factors, idx_load_factors, instances, card_limits, branch_data, gen_data, time_limit, MIP_gap, ksp_weights, max_pathS, max_edgeS, conv_levelS)
    if network == 1
        time_df = DataFrame!(CSV.File("merged_branch_118/ksp_time.csv"))
    elseif network == 2
        time_df = DataFrame!(CSV.File("merged_branch_300/ksp_time.csv"))
    end
    output_df = DataFrame(data_instance=String[], card = [], max_path = [], max_edge = [], conv_level = [], 
                             sol_status=[], obj_val=[], obj_bound=[], gap=[], time=[], num_open=[])
    for max_path in max_pathS, max_edge in max_edgeS, conv_level in conv_levelS
        for l in idx_load_factors, i in instances
            # read the data
            if network == 1
                data_file = "Data/118bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE118_P_$(load_factors[l]/100)_$i"
            elseif network == 2
                data_file = "Data/300bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE300_P_$(load_factors[l]/100)_$i"
            end

            new_bus_data = DataFrame!(CSV.File(data_file))
 
            # read big M values
            if network == 1
                k_val_file = "merged_branch_118/ksp_k_values/load_$(load_factors[l])_percent_instance_$(i).csv"               
            elseif network == 2
                k_val_file = "merged_branch_300/ksp_k_values/load_$(load_factors[l])_percent_instance_$(i).csv"
            end

            k_val_df = DataFrame!(CSV.File(k_val_file))
            if network == 1 # 118 bus case
                pre_k_val = k_val_df[:,"max_path_23_max_edge_$(max_edge)"]
            elseif network == 2 # 300 bus case
                pre_k_val = k_val_df[:,"max_path_40_max_edge_$(max_edge)"]
            end
            for idx in 1:length(pre_k_val)
                if pre_k_val[idx] >= max_path
                    pre_k_val[idx] = max_path
                end
            end
            k_val = pre_k_val .+ conv_level

            max_angle_diff = Vector{Float64}()
            for idx in branch_data.index
                if network == 1
                    push!( max_angle_diff, maximum(ksp_weights[idx,1:min(30, k_val[idx])]) )
                elseif network == 2
                    push!( max_angle_diff, maximum(ksp_weights[idx,1:min(50, k_val[idx])]) )
                end
            end
            bigM = max_angle_diff ./ branch_data.reactance
    
            # run OTS
            for card in card_limits
                results = OTS(new_bus_data, branch_data, gen_data, bigM, [], card, time_limit, MIP_gap, false)
                output_row = []
                push!(output_row, instance)
                push!(output_row, card)
                push!(output_row, max_path)
                push!(output_row, max_edge)
                push!(output_row, conv_level)

                instance_time_df = time_df[time_df.data_instance .== instance,:]
                instance_time_df = instance_time_df[instance_time_df.max_path .== max_path, :]
                instance_time_df = instance_time_df[instance_time_df.max_edge .== max_edge, :]
                #prepro_time = instance_time_df.time[1]
                #push!(output_row, prepro_time)
                for r in results
                    push!(output_row, r)
                end
                push!(output_df, output_row) 
            end
        end
    end
    return output_df
end

# An example of computing the k-shortest paths
#-----------------------------------------------------------------------------------------
#=
Branch_Data = DataFrame!(CSV.File("Data/118bus/Branch_Data_IEEE118_merged.csv"))
Bus_Data = DataFrame!(CSV.File("Data/118bus/Bus_Data_IEEE118.csv"))

K = 50
output_paths_mx, output_path_weights_mx = comp_kSP(Branch_Data, K)
output_paths_df = DataFrame(output_paths_mx)
CSV.write("merged_branch_118/ksp_paths.csv",output_paths_df)
output_path_weights_df = DataFrame(output_path_weights_mx)
CSV.write("merged_branch_118/ksp_weights.csv",output_path_weights_df)
=#
#--------------------------------------------------------------------------------------

# An example of computing the k values for big M values in DC OTS problem
#--------------------------------------------------------------------------------------
#=
branch_data =  DataFrame!(CSV.File("Data/300bus/IEEE300_branch_merged.csv"))
branch_index = Dict{Tuple{Int64,Int64},Int64}() # a dictionary that mapps buses into branch index
for i in branch_data.index
    fbus = branch_data.node1[i]
    tbus = branch_data.node2[i]
    branch_index[fbus,tbus] = i
    branch_index[tbus,fbus] = i
end
gen_data = DataFrame!(CSV.File("Data/300bus/IEEE300_gen.csv"))
ksp_paths = DataFrame!(CSV.File("merged_branch_300/ksp_paths.csv"))

load_factors = [90,95,100,105,110]
idx_load_factors = [2,3,4]
instances = 1:20
max_pathS = [50]
max_edgeS = [1,2]

compute_ksp_k_val(2,false,load_factors,idx_load_factors,instances,branch_data,gen_data,ksp_paths,max_pathS,max_edgeS)
=#
#--------------------------------------------------------------------------------------

# An example of run DC OTS model with k-shortest-path "big-M" values on tested instances
#--------------------------------------------------------------------------------------
#=
branch_data =  DataFrame!(CSV.File("Data/300bus/IEEE300_branch_merged.csv"))
gen_data = DataFrame!(CSV.File("Data/300bus/IEEE300_gen.csv"))
ksp_weights = DataFrame!(CSV.File("merged_branch_300/ksp_weights.csv"))

time_limit = 600
MIP_gap = 0.001

load_factors = [90,95,100,105,110]
idx_load_factors = [3]
instances = 1:20
card_limits = [10,15,20,25]
max_pathS = [15]
max_edgeS = [5]
conv_levelS = [3]
output_file = "merged_branch_300/OTS_ksp_card.csv"
output_df = OTS_ksp_bigM(2, load_factors, idx_load_factors, instances, card_limits, branch_data, gen_data, time_limit, MIP_gap, ksp_weights, max_pathS, max_edgeS, conv_levelS)
CSV.write(output_file, output_df)
=#
#--------------------------------------------------------------------------------------