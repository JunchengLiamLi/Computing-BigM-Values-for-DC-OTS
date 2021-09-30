# This file contains the codes to solve DC OTS problem on tested instances with big-M values computed via longest path method
# To do so, do the following steps:
# 1. compute voltage angle difference limit for each edge in given tested intances (call the function: compute_lwp_weights_MTZ())
# 2. run DC OTS model with  big-M values on given tested instances (call the function: OTS_lwp_bigM())

using CSV, DataFrames
using JuMP, Gurobi
using Dates

function MTZ_longest_path(edges, nodes, startNode, endNode, weights, time_limit, gap)
    lwpp = Model(optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => time_limit, "MIPGap" => gap))

    # x[i,j] = 1 if the flow goes from i to j, x[i,j] = 0 otherwise
    @variable(lwpp, x[nodes, nodes], binary=true)

    # u represent the position of the node in the tour
    remaining_nodes = filter(x->x≠startNode, nodes)
    @variable(lwpp, u[remaining_nodes])

    # flow balance: what goes in must goes out
    @constraint(lwpp, [node in nodes; node!=endNode && node!=startNode],
        sum(x[f_node, node] for f_node in nodes) == sum(x[node, f_node] for f_node in nodes) )    
    
    # flow balance for the starting node
    @constraint(lwpp, sum(x[node,startNode] for node in nodes) == 0)
    @constraint(lwpp, sum(x[startNode,node] for node in nodes) == 1)

    # flow balance for the ending node
    @constraint(lwpp, sum(x[node,endNode] for node in nodes) == 1)
    @constraint(lwpp, sum(x[endNode,node] for node in nodes) == 0)

    # if an edge (i,j) does not exist, then x[i,j] = x[j,i] = 0
    for n1 in nodes, n2 in nodes
        if !( (n1,n2) in edges || (n2,n1) in edges )
            @constraint(lwpp, x[n1,n2] == 0)
            @constraint(lwpp, x[n2,n1] == 0)
        end
    end

    # no cancellation
    @constraint(lwpp, [(fromNode, toNode) in edges], x[fromNode, toNode] + x[toNode, fromNode] <= 1 )
    
    # at most one edge in for each node
    @constraint(lwpp, [node in nodes], sum(x[f_node,node] for f_node in nodes) <= 1)

    # at most one edge out for each node
    @constraint(lwpp, [node in nodes], sum(x[node,t_node] for t_node in nodes) <= 1)

    # the edge (startNode, endNode) is disconnected
    # @constraint(lwpp, x[startNode,endNode] == 0)

    # MTZ constraints to eliminate subtours
    num_nodes = length(nodes)
    @constraint(lwpp, [n1 in remaining_nodes, n2 in remaining_nodes; n1 ≠ n2], (num_nodes-1)*x[n1,n2] + u[n1] - u[n2] <= num_nodes-2 )
    
    @objective(lwpp, Max, sum(weights[fromNode,toNode]*(x[fromNode,toNode] + x[toNode,fromNode]) for (fromNode, toNode) in edges) )
    #set_silent(lwpp)
    optimize!(lwpp)

    
    active_edges = Vector{Tuple{Int64, Int64}}()
    for (f_node, t_node) in edges
        if abs(value(x[f_node,t_node])-1) < 0.01
            push!(active_edges, (f_node,t_node))
        elseif abs(value(x[t_node,f_node])-1) < 0.01
            push!(active_edges, (t_node,f_node))
        end
    end
    # find out the path from startNode to endNode
    longest_path = "$startNode"
    current_node = startNode
    while !(current_node == endNode)
        for node in nodes
            if abs(value(x[current_node,node])-1) < 0.01
                filter!(x->x≠(current_node,node),active_edges) # move the arc to the path
                current_node = node
                longest_path = longest_path * "->$node"
                break
            end
        end
    end

    # test and debug
    subtours = Vector{Vector{Int64}}()
    #=
    #---------------------------------------------------------------------------------------
    # find out the subtours
    while length(active_edges) > 0
        subtour = Vector{Int64}()
        subtour_found = false
        begin_node = active_edges[1][1] # a random starting node for the cycle
        push!(subtour, begin_node)
        current_node = begin_node
        while !subtour_found
            for (f_node, t_node) in active_edges # for each active arc
                if f_node == current_node # if the arc start from the current node
                    current_node = t_node # update the current node
                    push!(subtour, t_node)
                    filter!(x->x ≠ (f_node, t_node),active_edges) # move the arc to the subtour
                    break
                end
            end
            if current_node == begin_node
                subtour_found = true # A subtour has been found
            end
        end
        push!(subtours, subtour)
    end

    # write down the subtours
    output_ar = Vector{String}()
    push!(output_ar, longest_path)
    for subtour in subtours
        subtour_str = "$(subtour[1])"
        for node in subtour[2:end]
            subtour_str = subtour_str * "->$node"
        end
        push!(output_ar, subtour_str)
    end
    output_df = DataFrame(paths = output_ar)
    return output_df
    #--------------------------------------------------------------------------------------
    =#

    if termination_status(lwpp) == MOI.INFEASIBLE || length(subtours) > 0
        return weights[startNode, endNode], "None", "None", "None"
    else    
        return objective_value(lwpp), relative_gap(lwpp), solve_time(lwpp), longest_path
    end
end

# compute weights of longest path for each edge
function compute_lwp_weights_MTZ(bus_data, branch_data, output_file, branch_range, ksp_weights)
    edges = Vector{Tuple{Int64,Int64}}()
    weights = Dict()
    for idx in branch_data.index
        fbus = branch_data.node1[idx]
        tbus = branch_data.node2[idx]
        push!(edges, (fbus,tbus))
        weights[fbus,tbus] = branch_data.transLimit[idx] * branch_data.reactance[idx]
        weights[tbus,fbus] = branch_data.transLimit[idx] * branch_data.reactance[idx]
    end

    obj_val_vec = []
    gap_vec = []
    sol_time_vec = []
    paths_vec = []
    branch_i = []
    for idx in branch_data.index[branch_range]
        long_path_known = false
        for j in ksp_weights[idx,:]
            if j < 0
                long_path_known = true
                break
            end
        end
        if long_path_known #if the longest path is known
            push!(obj_val_vec, maximum(ksp_weights[idx,:]))
            push!(gap_vec, missing)
            push!(sol_time_vec, missing)
            push!(paths_vec, "check_ksp")
            push!(branch_i, idx)
        else
            fbus = branch_data.node1[idx]
            tbus = branch_data.node2[idx]
            obj_val, gap, time, path= MTZ_longest_path(edges, bus_data.index, fbus, tbus, weights, 600, 0.01)
            push!(obj_val_vec, obj_val)
            push!(gap_vec, gap)
            push!(sol_time_vec, time)
            push!(paths_vec, path)
            push!(branch_i, idx)
        end
    end
    output_df = DataFrame(branch_i = branch_i, path_weights = obj_val_vec, opt_gap = gap_vec, sol_time = sol_time_vec, paths = paths_vec)
    CSV.write(output_file, output_df)        
end

# OTS 
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
    #set_silent(myMod)
    optimize!(myMod)
    if termination_status(myMod) != MOI.INFEASIBLE
        num_lines_open = length(switchable_branches) - sum(value(branchOpen[edgeIdx, node1, node2])  for (edgeIdx, node1, node2) in switchable_branches )
        return termination_status(myMod), objective_value(myMod), objective_bound(myMod), relative_gap(myMod), solve_time(myMod), num_lines_open
    else
        return termination_status(myMod), missing, missing, missing, solve_time(myMod), missing
    end
end

# if network == 1, use 118 bus data
# if network == 2, use 300 bus data
function OTS_lwp_bigM(network, input_lwp_results, load_scales, instances, branch_data, gen_data, cards, time_limit, gap, output_file)
    output_df = DataFrame(data_instance=String[],  card = [], sol_status=[], obj_val=[], obj_bound=[], gap=[], time=[], num_open=[])
    for load_scale in load_scales
        for i in instances
            # read the bus data
            if network == 1
                bus_data = CSV.read("Data/118bus/para_tune_loads/load_factor_$(load_scale)_percent/Bus_Data_$i.csv")
                instance = "IEEE118_P_$(load_scale)_$i"
            elseif network == 2
                bus_data = CSV.read("Data/300bus/para_tune_loads/load_factor_$(load_scale)_percent/Bus_Data_$i.csv")
                instance = "IEEE300_P_$(load_scale)_$i"
            end
            
            # read the big M values
            max_angle_diff = input_lwp_results[2]
            bigM = max_angle_diff ./ branch_data.reactance
            for card in cards
                # run OTS model
                results = OTS(bus_data, branch_data, gen_data, bigM, [], card, time_limit, gap, false)

                # record the results    
                output_row = []
                push!(output_row, instance)
                push!(output_row, card)
                for r in results
                    push!(output_row, r)
                end
                push!(output_df, output_row)
            end
        end
    end
    CSV.write(output_file, output_df)
end

# An example of computing longest-path big-M values on tested instances
#----------------------------------------------------------------------------------------
#=
bus_data = DataFrame!(CSV.File("Data/300bus/IEEE300_bus.csv"))
branch_data = DataFrame!(CSV.File("Data/300bus/IEEE300_branch_merged.csv"))
output_file = "merged_branch_300/lwp_max_angle_diff.csv"
branch_range = 1:10
ksp_weights = DataFrame!(CSV.File("merged_branch_300/ksp_weights.csv"))
compute_lwp_weights_MTZ(bus_data, branch_data, output_file, branch_range, ksp_weights)
=#
#----------------------------------------------------------------------------------------

# An example of solving DC OTS problem with longest-path big-M values on tested instances
#----------------------------------------------------------------------------------------
#=
branch_data = DataFrame!(CSV.File("Data/300bus/IEEE300_branch_merged.csv"))
gen_data = DataFrame!(CSV.File("Data/300bus/IEEE300_gen.csv"))
ksp_weights = DataFrame!(CSV.File("merged_branch_300/ksp_weights.csv"))

time_limit = 600
MIP_gap = 0.001
lwp_weights_df = DataFrame!(CSV.File("merged_branch_300/lwp_max_angle_diff.csv"))
output_file = "merged_branch_300/OTS_lwp_para_tune.csv"
OTS_lwp_bigM(2, lwp_weights_df, [95,100,105], 1:20, branch_data, gen_data, [45], time_limit, MIP_gap, output_file)
=#
#----------------------------------------------------------------------------------------