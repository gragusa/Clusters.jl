using Clusters
using DataFrames
using Base.Test
using CovarianceMatrices

const sim = 300

## Two test.

#=
 1. - Make sure that the estimates obtained from estimatemodel
      coincides with those of STATA
=#

df = readtable("test.txt")
ng = vec(readcsv("states.txt", header = true)[1])
ng = floor.(Int64, ng./5000)

# G::Int64
# ng::Vector{Int64}
# ## Regressors std. dev.
# σ_z::Float64
# σ_ξ::Float64
# ## Errors std. dev
# σ_ϵ::Float64
# σ_α::Float64
# ## Third/Fourth design parms
# p::Float64
# γ::Float64
# δ::Float64
# σ_η::Float64
# μ_q::Float64
# ## Null value
# β₀::Float64
# design::Int

opt = LinearRegressionClusterOpt(length(ng), ng,
                                  1,  ## σ_z
                                1/9,  ## σ_ξ
                                  1,  ## σ_ϵ
                                1/9,  ## σ_α
                                0.0,  ## p
                                0.0,  ## γ
                                0.0,  ## δ
                                0.0,  ## σ_η
                                0.0,  ## μ_q
                                0.0,  ## β₀
                                  1)
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
      @testset "Check correctness of parameters" begin
            σ_ϵ = m.opt.σ_z
            σ_α = m.opt.σ_ξ
            σ_z = m.opt.σ_z
            σ_ξ = m.opt.σ_ξ
            γ   = m.opt.γ
            p   = m.opt.p
            ## The parameters are std. dev
            @test σ_ϵ == 1
            @test σ_α == 1/9
            @test σ_z == 1
            @test σ_ξ == 1/9
            ## Check ICC of regressors and errors
            @test icc_x(m) == σ_ξ^2/(σ_z^2+σ_ξ^2)
            @test icc_u(m) == σ_α^2/(σ_ϵ^2+σ_α^2)
      end
end

#=
# 2. - Run a Small Montecarlo
=#
ng = repeat([30], inner = 30)
opt = opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1.0, ## σ_z
                                       1.0, ## σ_ξ
                                       1.0, ## σ_ϵ
                                       1.0, ## σ_α
                                       0.0, ## p
                                       0.0, ## γ
                                       0.0, ## δ
                                       0.0, ## σ_η
                                       0.0, ## μ_q
                                       0.0, ## β₀
                                       1) ## Design

m = initialize(LinearRegressionCluster, opt)
srand(1234512345)
out = montecarlo(LinearRegressionCluster, opt)

@testset "Set 2" begin
      @test maximum(out[:theta_u] -  out[:theta_w]) ≈ 0 atol = 1e-10
      ## Confidence interval
      ci = ConfInterval(out[:theta_u], out[:V3_u], 1.96)
      @test average_length(ci) <= 0.4
      ## Construct WB confidence interval
      ciwb = ConfInterval(out[:theta_w], out[:V3_w], out[:qw0_025], out[:qw0_975])
      @test coverage(ciwb, 0) ≈ 0.93 atol = 0.01
      ciwb = ConfInterval(out[:theta_u], out[:V3_u], out[:qu0_025], out[:qu0_975])
      @test coverage(ciwb, 0) ≈ 0.93 atol = 0.01
      ciwb = ConfInterval(out[:theta_w], out[:V3_w], out[:qw0_050], out[:qw0_950])
      @test coverage(ciwb, 0) ≈ 0.90 atol = 0.04
      ciwb = ConfInterval(out[:theta_u], out[:V3_u], out[:qu0_050], out[:qu0_950])
      @test coverage(ciwb, 0) ≈ 0.90 atol = 0.04
end

function run_montecarlo(d, sim)
      dd = Array{Float64}(sim)
      od = similar(dd)
      N = length(d.opt.ng)
      ng = maximum(d.opt.ng)
      M = Array{Float64}(ng,ng);

      for i in 1:sim
            simulate!(d)
            fill!(M, 0.0)
            for j in d.bstarts
                  E = d.y[j]
                  M = M + E*E'./N
            end
            dd[i] = mean(diag(M))
            od[i] = M[1,2]
      end
      (dd, od)
end

ci_lower_lim(x, z = 2.5) = mean(x) - z*std(x)/sqrt(length(x))
ci_upper_lim(x, z = 2.5) = mean(x) + z*std(x)/sqrt(length(x))

theoretical_Λ_ii(d, ::Type{Val{1}}) = d.opt.σ_ϵ^2+d.opt.σ_α^2
theoretical_Λ_ik(d, ::Type{Val{1}}) = d.opt.σ_α^2


theoretical_Λ_ii(d, ::Type{Val{2}}) = theoretical_Λ_ii(d, Val{1})
theoretical_Λ_ik(d, ::Type{Val{2}}) = theoretical_Λ_ik(d, Val{1})


function theoretical_Λ_ik(d, ::Type{Val{3}})
      (σ_z, σ_ξ, σ_ϵ, σ_α, p, γ, δ, σ_η, μ_q) = Clusters.getparms(d)
      σ_α^2+γ^2*(1-p)*p*σ_ξ^2
end

function theoretical_Λ_ii(d, ::Type{Val{3}})
      (σ_z, σ_ξ, σ_ϵ, σ_α, p, γ, δ, σ_η, μ_q) = Clusters.getparms(d)
      σ_ϵ^2 + σ_α^2+γ^2*(1-p)*p*(1+σ_ξ^2)
end

function theoretical_Λ_ik(d, ::Type{Val{4}})
      (σ_z, σ_ξ, σ_ϵ, σ_α, p, γ, δ, σ_η, μ_q) = Clusters.getparms(d)
      a = σ_η^2 + δ*σ_ξ^2*(2*δ + 3*μ_q) + δ^2*σ_z^2
      γ^2*σ_ξ^2*a
end

function theoretical_Λ_ii(d, ::Type{Val{4}})
      (σ_z, σ_ξ, σ_ϵ, σ_α, p, γ, δ, σ_η, μ_q) = Clusters.getparms(d)
      a = 2*δ^2*σ_ξ^4
      b = σ_ξ^2*(σ_η^2 + μ_q^2 + δ^2*σ_z^2)
      c = σ_z^2*(σ_η^2 + μ_q^2)            
      1 + γ^2*(a+b+c)
end




@testset "Design 1" begin
      ng = repeat([2], outer=2000)
      opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1,    ## σ_z
                                       1/9,  ## σ_ξ
                                       1,    ## σ_ϵ
                                       1/9,  ## σ_α
                                       0.0,  ## p
                                       0.0,  ## γ
                                       0.0,  ## δ
                                       0.0,  ## σ_η
                                       0.0,  ## μ_q
                                       0.0,  ## β₀
                                        1)   ## Desing

      d = initialize(LinearRegressionCluster, opt)
      simulate!(d)
      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, sim)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{1})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{1})

      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{1})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{1})
end

@testset "Design 2" begin
      ng = repeat([2], outer=2000)

      opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1,    ## σ_z
                                       1/9,  ## σ_ξ
                                       1,    ## σ_ϵ
                                       1/9,  ## σ_α
                                       0.0,  ## p
                                       0.0,  ## γ
                                       0.0,  ## δ
                                       0.0,  ## σ_η
                                       0.0,  ## μ_q
                                       0.0,  ## β₀
                                        2)   ## Desing

      d = initialize(LinearRegressionCluster, opt)
      simulate!(d)
      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, sim)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{d.opt.design})

      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{d.opt.design})

      opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1,    ## σ_z
                                       1/9,  ## σ_ξ
                                       1,    ## σ_ϵ
                                       1/2,  ## σ_α
                                       0.0,  ## p
                                       0.0,  ## γ
                                       0.0,  ## δ
                                       0.0,  ## σ_η
                                       0.0,  ## μ_q
                                       0.0,  ## β₀
                                        2)   ## Desing

      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, sim)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{d.opt.design})

      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{d.opt.design})
end

@testset "Design 3" begin
      ng = repeat([2], outer=20000)
      ## ρᵤ = 0.2  ρₓ = 0.2
      opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1,    ## σ_z
                                       0.5,  ## σ_ξ
                                       1,    ## σ_ϵ
                                       0.5,  ## σ_α
                                       0.5,  ## p
                                       0.1,  ## γ
                                       0.0,  ## δ
                                       0.0,  ## σ_η
                                       0.0,  ## μ_q
                                       0.0,  ## β₀
                                        3)   ## Desing

      d = initialize(LinearRegressionCluster, opt)
      simulate!(d)
      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, sim)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{d.opt.design})
end

@testset "Design 4" begin
      ## This test that as n_g -> ∞
      ng = repeat([150], outer=2)

      opt = LinearRegressionClusterOpt(length(ng), ng,
                                       1, ## σ_z
                                     0.5, ## σ_ξ
                                       0, ## σ_ϵ
                                       0, ## σ_α
                                     0.0, ## p
                                    -1.0, ## γ
                       .9354143466934853, ## δ
                       .5361902647381804, ## σ_η
                                    -0.5, ## μ_q
                                     0.0, ## β₀
                                       4) ## Design


      d = initialize(LinearRegressionCluster, opt)
      simulate!(d)
      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, 300)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{d.opt.design})

      ng = repeat([250], outer=2)
      opt = LinearRegressionClusterOpt(length(ng), ng,
                                        1, ## σ_z
                                      0.5, ## σ_ξ
                                        0, ## σ_ϵ
                                        0, ## σ_α
                                      0.0, ## p
                                     -1.0, ## γ
                        .9354143466934853, ## δ
                        .5361902647381804, ## σ_η
                                     -0.5, ## μ_q
                                      0.0, ## β₀
                                        4) ## Design

      d = initialize(LinearRegressionCluster, opt)
      simulate!(d)
      srand(1234554321)
      Λ_ii, Λ_ik = run_montecarlo(d, sim)
      @test ci_lower_lim(Λ_ii) <= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ii) >= theoretical_Λ_ii(d, Val{d.opt.design})
      @test ci_lower_lim(Λ_ik) <= theoretical_Λ_ik(d, Val{d.opt.design})
      @test ci_upper_lim(Λ_ik) >= theoretical_Λ_ik(d, Val{d.opt.design})
end