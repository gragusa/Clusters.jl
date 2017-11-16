using Clusters
using DataFrames
using Base.Test

## Two test. 

#=
 1. - Make sure that the estimates obtained from Clusters.estimatemodel
      coincides with those of STATA
=#

df = readtable("test.txt")
ng = vec(readcsv("states.txt", header = true)[1])
ng = floor.(Int64, ng./5000)

opt = Clusters.LinearRegressionClusterOpt(length(ng), ng, 1, 1/9, 1, 1/9, 0.0, 0.0, 0.0, 1)
m = Clusters.initialize(LinearRegressionCluster, opt)
simulate!(m)
## Inject data from test.txt
copy!(m.y, df[:y])
copy!(m.X, df[:x])

## Make sure everything is fine
@test m.cl == df[:cl]
@test m.wts == df[:wts]

@test all(m.y .== df[:y])
@test all(m.X .== df[:x])

out = Clusters.estimatemodel(m)

@test out[1] ≈ -.0421262 atol = 0.0001  ## Unweighted OLS
@test out[2] ≈ .0275297 atol = 0.0001   ## iid std
@test out[3] ≈ .0276643 atol = 0.0001   ## HC std
@test out[4] ≈ .0296611 atol = 0.0001   ## CRHC1 std
@test out[5] ≈ .0296611 atol = 0.01     ## CRHC2 std 
@test out[6] ≈ .0296611 atol = 0.01     ## CRHC3 std

@test out[17] ≈  -.1051925 atol = 0.0001 ## Weighted OLS
@test out[18] ≈  .0269856 atol = 0.0001 ## iid std
@test out[19] ≈  .0427102 atol = 0.0001 ## HC std
@test out[20] ≈  .0487519 atol = 0.0001 ## CRHC1 std

#=
# 2. - Run a Small Montecarlo 
=#
ng = repeat([30], inner = 30)
opt = Clusters.LinearRegressionClusterOpt(length(ng), ng, 1, 1, 1, 1, 0.0, 0.0, 0.0, 1)
m = Clusters.initialize(LinearRegressionCluster, opt)
out = Clusters.montecarlo(LinearRegressionCluster, opt)


@test maximum(out[:theta_u] -  out[:theta_w]) ≈ 0 atol = 1e-10

## Construct WB confidence interval
ciwb = ConfInterval(out[:theta_w], out[:V3_w], out[:qw0_050], out[:qw0_950])
coverage(ciwb, 0)
ciwb = ConfInterval(out[:theta_u], out[:V3_u], out[:qu0_025], out[:qu0_975])
coverage(ciwb, 0)

