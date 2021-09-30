# This file contains the codes to solve DC OTS problem on tested instances with heuristic "big-M" values computed via kNN simulation method
# To do so, do the following steps:
# 1. compute voltage angle difference for each edge in given tested intances (call the function: comp_kNN_sim_angDiff())
# 2. run DC OTS model with kNN simulation "big-M" values on given tested instances (call the function: OTS_kNN_bigM())

using CSV, DataFrames
using JuMP, Gurobi
using LightGraphs, Random
using Dates

# OPF function
#-------------------------------------------------------------------------------------------
function OPF(Bus_Data, Branch_Data, Generator_Data, Node1, Node2)
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer) );
    @variable(myMod, production[Generator_Data.index], lower_bound = 0)

    num_branches = length(Branch_Data.index)
    newBranches = Vector{Tuple{Int32, Int32, Int32}}(undef, num_branches)
    for idx in 1:num_branches
        newBranches[idx] = (idx, Branch_Data.node1[idx], Branch_Data.node2[idx])
    end
    @variable(myMod, powerFlow[newBranches])
    #=
    powerFlow = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()
    for (index, node1, node2) in newBranches
        global powerFlow[index, node1, node2] = @variable(myMod)
    end
    =#

    @variable(myMod, volAng[Bus_Data.index])

    power_demand = Dict{Int64, Float64}()
    for i in 1:length(Bus_Data.index)
        power_demand[Bus_Data.index[i]] = Bus_Data.demand[i]
    end

    # Define constraints
    #---------------------------------------------------------------------------------------------------------
    bus_no_generators = setdiff(Bus_Data.index, Generator_Data.bus)

    generatorInBus = Dict()
    for index in Generator_Data.index
        generatorInBus[Generator_Data.bus[index]] = index
    end

    # production limits on power plants
    @constraint(myMod, [g in Generator_Data.index], production[g] <= Generator_Data.pmax[g])

    # Network flow balance
    #---------------------------------------------------------------------------------
    @constraint(myMod, [b in Generator_Data.bus],
        sum(powerFlow[(index_, node1_, node2_)] for (index_, node1_, node2_) in newBranches if node2_ == b)
        - sum(powerFlow[(index_, node1_, node2_)] for (index_, node1_, node2_) in newBranches if node1_ == b)
        + production[generatorInBus[b]]
        == power_demand[b]
        )

    @constraint(myMod, [bus in bus_no_generators],
        sum(powerFlow[(index_, node1_, node2_)] for (index_, node1_, node2_) in newBranches if node2_ == bus)
        - sum(powerFlow[(index_, node1_, node2_)] for (index_, node1_, node2_) in newBranches if node1_ == bus)
        == power_demand[bus]
        )
    #--------------------------------------------------------------------------------

    # Kirchhoff's Law
    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[(index, node1, node2)]
                - 1/Branch_Data.reactance[index]*(volAng[node1] - volAng[node2]) == 0)

    # Power flow limits
    #--------------------------------------------------------------------------------
    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[(index, node1, node2)] + Branch_Data.transLimit[index] >= 0)

    @constraint(myMod, [(index, node1, node2) in newBranches],
                powerFlow[(index, node1, node2)] - Branch_Data.transLimit[index] <= 0)
    #-----------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------------
    @objective(myMod, Min, sum(production[g]*Generator_Data.cost[g] for g in Generator_Data.index))
    set_silent(myMod)
    optimize!(myMod)
    sol_status = termination_status(myMod)
    if sol_status == MOI.OPTIMAL
        vol_ang_diff = abs( value(volAng[Node1]) - value(volAng[Node2]) )
    else
        vol_ang_diff = -1
    end
    return sol_status, vol_ang_diff, solve_time(myMod)
end
#---------------------------------------------------------------------------------

# DC OTS function
#---------------------------------------------------------------------------------------
function OTS(busData, branchData, generatorData, bigM, idx_unswitchable_lines, cardinality_limit, timeLimit, gap, relaxation)
    myMod = Model( optimizer_with_attributes(Gurobi.Optimizer, "Seed" => 1, "TimeLimit" => timeLimit, "MIPGap" => gap) );
    
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

    power_demand = Dict{Int64, Float64}()
    for i in 1:length(busData.index)
        power_demand[busData.index[i]] = busData.demand[i]
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
    generators = Set(generatorData.bus)
    buses = Set(busData.index)
    nonGenBuses = setdiff(buses, generators)
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
    set_silent(myMod)
    optimize!(myMod)
    if termination_status(myMod) != MOI.INFEASIBLE
        num_lines_open = length(switchable_branches) - sum(value(branchOpen[edgeIdx, node1, node2])  for (edgeIdx, node1, node2) in switchable_branches )
        return termination_status(myMod), objective_value(myMod), objective_bound(myMod), relative_gap(myMod), solve_time(myMod), num_lines_open
    else
        return termination_status(myMod), missing, missing, missing, solve_time(myMod), missing
    end
end
#----------------------------------------------------------------------------------------

# computing voltage angle difference limit for each edge in a give data instance via kNN simulation method
# each row in the output matrix corresponds to voltage angle differences of all edges under certain parameter
#--------------------------------------------------------------------------------------
function kNN_sim(data_instance, bus_data, branch_data, gen_data, sizes_neighbour, per_lines_rm, repeat, scal_factors, record_log)
    num_com_para = length(sizes_neighbour)*length(per_lines_rm)*length(scal_factors) # number of all combinations of parameters in concern
    num_row_log = length(sizes_neighbour)*length(per_lines_rm)*repeat
    num_branch = length(branch_data.index)
    num_neigh = length(sizes_neighbour)
    num_per = length(per_lines_rm)
    num_sca = length(scal_factors)
    output_mx = Matrix{Union{Float64,String,Int64}}(undef, num_com_para, num_branch+6) # data instance, size of neighbour, percentage of lines remove, scaling factor, total preprocessing time, maximum LP time 
    if record_log
        log_mx = Matrix{Union{Float64,String,Int64}}(undef, num_row_log, num_branch+4) # data instance, size of neighbour, percentage of lines removed, repetition
        log_mx[:,1] .= data_instance 
    end
    # Construct the julia graph
    #--------------------------------------------------
    num_nodes = maximum(bus_data.index)
    ju_graph = Graph(num_nodes)
    distmx = Inf*ones(num_nodes,num_nodes)
    
    for idx in branch_data.index
        add_edge!(ju_graph, branch_data.node1[idx], branch_data.node2[idx])
        distmx[branch_data.node1[idx], branch_data.node2[idx]] = 1
        distmx[branch_data.node2[idx], branch_data.node1[idx]] = 1
    end
    #---------------------------------------------------

    max_LP_time = 0
    sim_time_in_seconds = Dict{Tuple{Int64, Int64, Float64}, Float64}() # key: branch, size of neigh, percentage

    for idx in branch_data.index
        fbus = branch_data.node1[idx]
        tbus = branch_data.node2[idx]
        for s_idx in 1:num_neigh, p_idx in 1:num_per
            base_time = Dates.now() # the clock count begins
            s = sizes_neighbour[s_idx]
            p = per_lines_rm[p_idx]
            max_VAD = -1
            for r in 1:repeat
                # Apply Dijkstra's Algorithm to one of end nodes i of the edge
                distances_i = dijkstra_shortest_paths(ju_graph, fbus, distmx)
            
                # Form a set I by collecting the nodes close enough to node i from the results of Dijkstra
                close_nodes_i = Vector{Int64}()
                for node in bus_data.index
                    distanceFromNodeToI = length( enumerate_paths(distances_i, node) )
                    distanceFromNodeToI - 1 <= s && push!(close_nodes_i, node)
                end
            
                # Apply Dijkstra's Algorithm to one of end nodes j of the edge
                distances_j = dijkstra_shortest_paths(ju_graph, tbus, distmx)
            
                # Form a set J by collecting the nodes close enough to node j from the results of Dijkstra
                close_nodes_j = Vector{Int64}()
                for node in bus_data.index
                    distanceFromNodeToJ = length( enumerate_paths(distances_j, node) )
                    distanceFromNodeToJ - 1 <= s && push!(close_nodes_j, node)
                end
            
                # Union of Set I and Set J
                k_neighbors_nodes = union(close_nodes_i,close_nodes_j)
            
                # Find out the set of edges in the original graph that connect this set of nodes
                k_neighbors_edges = Array{Tuple{Int64, Int64, Int64},1}()
                for idx_ in branch_data.index
                    n1 = branch_data.node1[idx_]
                    n2 = branch_data.node2[idx_]
                    if n1 in k_neighbors_nodes && n2 in k_neighbors_nodes
                        push!( k_neighbors_edges, (idx_,n1,n2) )
                    end
                end
 
                removed_edges_idx = Array{Int64, 1}()
                num_neighbors_edges = length(k_neighbors_edges)
                perm_idx_neighbors = randperm(num_neighbors_edges)
                for i in 1:num_neighbors_edges
                    perm_idx_neighbors[i] <= num_neighbors_edges*p && push!(removed_edges_idx, k_neighbors_edges[i][1])
                end
   
                reduced_branch_data = branch_data[setdiff(1:end, removed_edges_idx),:]
                sol_status, VAD, time = OPF(bus_data, reduced_branch_data, gen_data, fbus, tbus)
                if time > max_LP_time
                    max_LP_time = time
                end

                if sol_status == MOI.OPTIMAL
                    if VAD > max_VAD
                        max_VAD = VAD
                    end
                end

                if record_log
                    row_idx = (s_idx-1)*(num_per*repeat) + (p_idx-1)*repeat + r
                    log_mx[row_idx, 2] = sizes_neighbour[s_idx]
                    log_mx[row_idx, 3] = per_lines_rm[p_idx]
                    log_mx[row_idx, 4] = r
                    log_mx[row_idx, idx+4] = VAD
                end
            end# for each repetition
            for sca_idx in 1:num_sca
                row_idx = (s_idx-1)*(num_per*num_sca) + (p_idx-1)*num_sca + sca_idx
                output_mx[row_idx,idx+6] = max_VAD * scal_factors[sca_idx]
            end# for each scaling factor  
            sim_time_in_seconds[idx,s_idx,p_idx] = Dates.value(Dates.now()-base_time)/1000 # the clock count ends
        end# for combination of parameters  
    end#for each branch
    
    for s_idx in 1:num_neigh, p_idx in 1:num_per, sca_idx in 1:num_sca
        row_idx = (s_idx-1)*(num_per*num_sca) + (p_idx-1)*num_sca + sca_idx
        output_mx[row_idx, 1] = data_instance
        output_mx[row_idx, 2] = sizes_neighbour[s_idx]
        output_mx[row_idx, 3] = per_lines_rm[p_idx]
        output_mx[row_idx, 4] = scal_factors[sca_idx]
        total_time = sum(sim_time_in_seconds[idx,s_idx,p_idx] for idx in branch_data.index) # total time of this process in seconds
        output_mx[row_idx, 5] = total_time
        output_mx[row_idx, 6] = max_LP_time
    end

    if record_log
        return output_mx, log_mx
    else
        return output_mx
    end
    
end
#-------------------------------------------------------------------------------------

# compute kNN simulation angle differences
# if network == 1, use 118 bus test data
# if network == 2, use 300 bus test data
function comp_kNN_sim_angDiff(network, load_scales, instances, branch_data, gen_data, sizes_neighbour, per_lines_rm, repeat, scal_factors, output_file, log_file)
    output_df = DataFrame()
    log_df = DataFrame()
    for load_scale in load_scales
        for i in instances
            # read the data
            if network == 1
                bus_data = CSV.read("Data/118bus/para_tune_loads/load_factor_$(load_scale)_percent/Bus_Data_$i.csv")
                data_instance = "IEEE118_P_$(load_scale)_$i"
            elseif network == 2
                bus_data = CSV.read("Data/300bus/para_tune_loads/load_factor_$(load_scale)_percent/Bus_Data_$i.csv")
                data_instance = "IEEE300_P_$(load_scale)_$i"
            end

            instance_mx, inst_log_mx = kNN_sim(data_instance, bus_data, branch_data, gen_data, sizes_neighbour, per_lines_rm, repeat, scal_factors, true)
            instance_df = DataFrame(instance_mx)
            inst_log_df = DataFrame(inst_log_mx)
            append!(output_df, instance_df)
            append!(log_df, inst_log_df)
        end
    end
    df_names = Vector{String}(undef, 6+length(branch_data.index))
    first_names = ["data_instance", "size_neigh", "per_lines_rm", "scal_factor", "time", "max_LP_time"]
    for idx in 1:6
        df_names[idx] = first_names[idx]
    end
    for idx in 7:(length(branch_data.index)+6)
        df_names[idx] = "branch_$(idx-6)"
    end
    rename!(output_df, df_names)
    CSV.write(output_file, output_df)
    CSV.write(log_file, log_df)
end

# if network == 1, use 118 bus test data
# if network == 2, use 300 bus test data
function OTS_kNN_bigM(network, input_kNN_results, lwp_weights_df, load_scales, instances, branch_data, gen_data, cards, time_limit, gap, output_file)
    output_df = DataFrame(data_instance=String[], size_neigh=Int64[], per_lines_rm=[], scal_factor=[], sim_time=[], max_LP = [], 
                            card = [], sol_status=[], obj_val=[], obj_bound=[], gap=[], time=[], num_open=[])
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
            
            # read kNN results and compute big M values
            instances_df = input_kNN_results[input_kNN_results.data_instance .== instance, :]
            for j in 1:nrow(instances_df)
                kNN_inf = instances_df[j,1:6] # kNN simulation information
                kNN_results = instances_df[j,7:end]
                kNN_results = convert(Vector, kNN_results)
                f(x,y) = x <= 0.01 ? y : min(x,y)
                num_edges = nrow(lwp_weights_df)
                lwp_weights = Vector{Float64}(undef, num_edges)
                for idx in 1:num_edges
                    lwp_weights[idx] = lwp_weights_df[2][idx]
                end
                max_ang_diff = f.(kNN_results, lwp_weights)
                kNN_bigM = max_ang_diff ./ branch_data.reactance
                for card in cards
                    results = OTS(bus_data, branch_data, gen_data, kNN_bigM, [], card, time_limit, gap, false)
                    output_row = []
                    for kNN_inf_piece in kNN_inf
                        push!(output_row, kNN_inf_piece)
                    end
                    push!(output_row, card)
                    for r in results
                        push!(output_row, r)
                    end
                    push!(output_df, output_row)
                end
            end
        end
    end
    CSV.write(output_file, output_df)
end

# An example of using kNN simulation to compute voltage angle difference limits 
#------------------------------------------------------------------------------------------
#=
load_scales = [100]
instances = 1:20
branch_data = DataFrame!(CSV.File("Data/118bus/Branch_Data_IEEE118_merged.csv"))
gen_data = DataFrame!(CSV.File("Data/118bus/Generator_Data_IEEE118.csv"))

sizes_neighbour = 3:4
per_lines_rm = [0.05, 0.1]
scal_factors = [5,10,15]
repeat = 30
output_file = "merged_branch_118/kNN_ang_diff_limits.csv"
log_file = "merged_branch_118/log_kNN_ang_diff_limits.csv"

comp_kNN_sim_angDiff(1, load_scales, instances, branch_data, gen_data, sizes_neighbour, per_lines_rm, repeat, scal_factors, output_file, log_file)
=#
#---------------------------------------------------------------------------------------------

# An example of running DC OTS model with kNN simulation "big-M" values on given tested instances
#---------------------------------------------------------------------------------------------
#=
load_scales = [95,100,105]
instances = 1:20
branch_data = DataFrame!(CSV.File("Data/118bus/Branch_Data_IEEE118_merged.csv"))
gen_data = DataFrame!(CSV.File("Data/118bus/Generator_Data_IEEE118.csv"))

input_kNN_results = DataFrame!(CSV.File("merged_branch_118/kNN_ang_diff_limits.csv"))
lwp_weights_df = DataFrame!(CSV.File("merged_branch_118/lwp_max_angle_diff.csv"))
time_limit = 600
instances = 1:20
gap = 0.001
cards = [45]
output_file = "merged_branch_118/OTS_kNN_demands.csv"

OTS_kNN_bigM(1, input_kNN_results, lwp_weights_df, load_scales, instances, branch_data, gen_data, cards, time_limit, gap, output_file)
=#
#---------------------------------------------------------------------------------------------