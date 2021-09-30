using CSV, DataFrames
using Statistics
using Plots

function scat_plot(network, load_factor, input_data, conv_levelS, best_para, shapeS, labelS, lg, colorS, xscaleS, output_file)
    # input the data
    #--------------------------------------------------------------
    if network == 2 # analyze 300-bus data
        load_data = input_data[input_data.load_factor .== load_factor,:]
        p = plot()
        for i in 1:length(conv_levelS)
            conv_level = conv_levelS[i]
            plot_data = load_data[load_data.conv_level .== conv_level,:]
            plot!(p, plot_data.avg_time, plot_data.avg_rel_gap .* 100, seriestype = :scatter, xlims = xscale, color = :blue, legend = lg, markershape = shapeS[i], label = labelS[i])
        end
        for i in 1:length(best_para)
            (p1,p2,p3) = best_para[i]
            best_para_data = input_data[input_data.max_path .== p1, :]
            best_para_data = input_data[input_data.max_edge .== p2, :]
            best_para_data = input_data[input_data.conv_level .== p3, :]
            plot!(p, best_para_data.avg_time, best_para_data.avg_rel_gap .* 100, seriestype = :scatter, xlims = xscale, color = colorS[i], legend = lg, markershape = :star5, label = "($p1,$p2,$p3)")
        end
        xlabel!(p, "average computation time (seconds)")
        ylabel!(p, "average relative gap(%)")
        if network == 1
            output_file_name = output_file*"_118bus_$(load_factor).png"
        elseif network == 2
            output_file_name = output_file*"_300bus_$(load_factor).png"
        end
        savefig(p, output_file_name)
    end
    #--------------------------------------------------------------
end

# input data
#--------------------------------------------------------------------------------------
input_data = DataFrame!(CSV.File("merged_branch_300/analyze_ksp_para_tuning_300.csv"))
network = 2
load_factor = 0.95
#---------------------------------------------------------------------------------------

#conservative levels
#----------------------------------------------------------------------------------
conv_levelS = [0,1]
shapeS = [:utriangle, :rect]
labelS = ["l=0","l=1"]
#----------------------------------------------------------------------------------

# best parameters
#-------------------------------------------------------------------------------
best_para = [(25,9,0)]
colorS = [:red]
#-------------------------------------------------------------------------------

xscale = (0,15)
lg = :left
output_file = "code and test instances/plots/scat_OTS_kSP_para_tune"

scat_plot(network, load_factor, input_data, conv_levelS, best_para, shapeS, labelS, lg, colorS, xscale, output_file)