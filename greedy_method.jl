# This file contains the codes to solve DC OTS problem via greedy method proposed by Fuller et al.(2012)
# To do so, call the function: OTS_fuller_lp_heuristic()
using LinearAlgebra
using CSV, DataFrames
using JuMP, Gurobi
using Dates

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

function fuller_lp_heuristic(bus_data, branch_data, gen_data, m, L)
    # setting up
    #----------------------------------------------------------------------
    l = 0 # number of edges removed from the network
    idx_rm_branch = Vector{Int64}() # indices of edges removed from the network
    sol_status, obj_val, power_flow, dual_flow_balance = OPF(bus_data, branch_data, gen_data)
    if sol_status == MOI.OPTIMAL
        stop = false
    else
        stop = true
    end
    #------------------------------------------------------------------------
    
    while l < L && !stop
        branch_data = branch_data[setdiff(1:end, idx_rm_branch),:]
        # find the candidate set of edges to remove according to alpha values
        if sol_status == MOI.OPTIMAL
            alpha = Vector{Float64}()
            for i in 1:length(branch_data.index)
                branch_idx = branch_data.index[i]
                push!(alpha, (dual_flow_balance[branch_data.node1[i]] - dual_flow_balance[branch_data.node2[i]]) * power_flow[branch_idx])
            end
            ranked_branch_idx = branch_data.index[sortperm(alpha)]  
            candi_idx = ranked_branch_idx[1:m]
        else
            stop = true
            break
        end

        # find the best edge in the candidate set
        feasi_idx_vec = Vector{Int64}()
        obj_val_vec = Vector{Float64}()
        power_flow_vec = Vector{Dict{Int64, Float64}}()
        dual_flow_vec = Vector{Dict{Int64, Float64}}()
        for idx in candi_idx
            new_branch_data = branch_data[setdiff(1:end,idx),:]
            trial_sol_status, trial_obj_val, trial_power_flow, trial_dual_flow = OPF(bus_data, new_branch_data, gen_data)
            if trial_sol_status == MOI.OPTIMAL
                push!(feasi_idx_vec, idx)
                push!(obj_val_vec, trial_obj_val)
                push!(power_flow_vec, trial_power_flow)
                push!(dual_flow_vec, trial_dual_flow)
            end
        end

        # update the network topology and relevant information for the next iteration
        #---------------------------------------------------------------
        if length(feasi_idx_vec) > 0
            new_obj_val, vec_idx = findmin(obj_val_vec)
            if new_obj_val < obj_val # improvement found & update the network configuration
                obj_val = new_obj_val
                rm_branch_idx = feasi_idx_vec[vec_idx]
                power_flow = power_flow_vec[vec_idx]
                dual_flow_balance = dual_flow_vec[vec_idx]
                push!(idx_rm_branch, rm_branch_idx)
                l += 1
            else
                stop = true
            end
        else # no edge in the candidate set makes OPF problem feasible 
            stop = true
        end
        #---------------------------------------------------------------
    end
    if l > 0
        return obj_val, l, idx_rm_branch
    else
        return obj_val, l, missing
    end
end

function OTS_fuller_lp_heuristic(network, load_factors, idx_load_factors, instances, branch_data, gen_data, card_limits, max_edgeS)
    output_df = DataFrame(data_instance=String[], card = [], size_candi = [], time=[], obj_val=[], num_open=[], idx_rm_branch=[])
    for l in idx_load_factors
        for i in instances
            if network == 1
                data_file = "Data/118bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE118_P_$(load_factors[l]/100)_$i"
            elseif network == 2
                data_file = "Data/300bus/para_tune_loads/load_factor_$(load_factors[l])_percent/Bus_Data_$i.csv"
                instance = "IEEE300_P_$(load_factors[l]/100)_$i"
            end
            
            new_bus_data = DataFrame!(CSV.File(data_file))
            for card in card_limits, m in max_edgeS
                base_time = Dates.now()
                results = fuller_lp_heuristic(new_bus_data, branch_data, gen_data, m, card)
                comp_time = Dates.value(Dates.now()-base_time)/1000 # in seconds

                output_row = []
                push!(output_row, instance)
                push!(output_row, card)
                push!(output_row, m)
                push!(output_row, comp_time)
                for r in results
                    push!(output_row, r)
                end

                push!(output_df, output_row)
            end
            
        end
    end
    return output_df
end

# An example of solving DC OTS problem with Fuller heuristics
#---------------------------------------------------------------------------------------
#=
branch_data =  DataFrame!(CSV.File("Data/300bus/IEEE300_branch_merged.csv"))
gen_data = DataFrame!(CSV.File("Data/300bus/IEEE300_gen.csv"))

load_factors = [90,95,100,105,110]
idx_load_factors = [2,3,4]
card_limits = [45]
instances = 1:20
max_edgeS = [1,2]

output_df = OTS_fuller_lp_heuristic(2, load_factors, idx_load_factors, instances, branch_data, gen_data, card_limits, max_edgeS)
CSV.write("merged_branch_300/OTS_fuller_lp_heuristic.csv", output_df)
=#
#---------------------------------------------------------------------------------------