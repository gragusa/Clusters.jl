using Clusters
using DataFrames
using Base.Test

## Two test.

#=
 1. - Make sure that the estimates obtained from estimatemodel
      coincides with those of STATA
=#

df = readtable("test.txt")
ng = vec(readcsv("states.txt", header = true)[1])
ng = floor.(Int64, ng./5000)

opt = LinearRegressionClusterOpt(length(ng), ng, 1, 1/9, 1, 1/9, 0.0, 0.0, 0.0, 1)
m = initialize(LinearRegressionCluster, opt)
simulate!(m)
## Inject data from test.txt
copy!(m.y, df[:y])
copy!(m.X, df[:x])
out = estimatemodel(m)
## Make sure everything is fine
@testset "Set 1" begin
      @test m.cl == df[:cl]
      @test m.wts == df[:wts]
      @test all(m.y .== df[:y])
      @test all(m.X .== df[:x])
      @testset "Coefficients and Variances" begin
            @test out[1] ≈ -.0421262 atol = 0.0001  ## Unweighted OLS
            @test out[2] ≈ 0.0275297 atol = 0.0001  ## iid std
            @test out[3] ≈ 0.0276643 atol = 0.0001  ## HC std
            @test out[4] ≈ 0.0296611 atol = 0.0001  ## CRHC1 std
            @test out[5] ≈ 0.0296611 atol = 0.01    ## CRHC2 std
            @test out[6] ≈ 0.0296611 atol = 0.01    ## CRHC3 std

            @test out[17] ≈  -.1051925 atol = 0.0001 ## Weighted OLS
            @test out[18] ≈  0.0269856 atol = 0.0001 ## iid std
            @test out[19] ≈  0.0427102 atol = 0.0001 ## HC std
            @test out[20] ≈  0.0487519 atol = 0.0001 ## CRHC1 std
      end
end

#=
# 2. - Run a Small Montecarlo
=#
ng = repeat([30], inner = 30)
opt = LinearRegressionClusterOpt(length(ng), ng, 1, 1, 1, 1, 0.0, 0.0, 0.0, 1)
m = initialize(LinearRegressionCluster, opt)
out = montecarlo(LinearRegressionCluster, opt)

@testset "Set 2" begin
      @test maximum(out[:theta_u] -  out[:theta_w]) ≈ 0 atol = 1e-10
      ## Confidence interval
      ci = ConfInterval(out[:theta_u], out[:V3_u], 1.96)
      @test average_length(ci)
      ## Construct WB confidence interval
      ciwb = ConfInterval(out[:theta_w], out[:V3_w], out[:qw0_025], out[:qw0_975])
      @test coverage(ciwb, 0) ≈ 0.95 atol = 0.01
      ciwb = ConfInterval(out[:theta_u], out[:V3_u], out[:qu0_025], out[:qu0_975])
      coverage(ciwb, 0)
      @test coverage(ciwb, 0) ≈ 0.95 atol = 0.01
      ciwb = ConfInterval(out[:theta_w], out[:V3_w], out[:qw0_050], out[:qw0_950])
      @test coverage(ciwb, 0) ≈ 0.90 atol = 0.04
      ciwb = ConfInterval(out[:theta_u], out[:V3_u], out[:qu0_050], out[:qu0_950])
      @test coverage(ciwb, 0) ≈ 0.90 atol = 0.04
end
