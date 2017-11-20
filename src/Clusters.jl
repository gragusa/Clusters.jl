module Clusters

using DataFrames, GLM, CovarianceMatrices, Distributions, GrowableArrays

abstract type WildWeights end
struct Rademacher <: WildWeights end
struct TwoPoint <: WildWeights end
struct Mammen1 <: WildWeights end
struct Mammen2 <: WildWeights end

abstract type MonteCarloModel end
abstract type MonteCarloModelOpt end

struct LinearRegressionCluster{G} <: MonteCarloModel
    ## Containers
    y::Array{Float64, 1}
    X::Array{Float64, 2}
    epsilon::Array{Float64, 1}
    eta::Array{Float64, 1}
    D::Array{Float64, 1}
    sigma_e::Array{Float64, 1}
    cl::Array{Int64, 1}
    iter::Array{Int64, 1}
    bstarts::Array{UnitRange{Int64},1}
    wts::Array{Float64, 1}
    sqwts::Array{Float64, 1}
    opt::G
end

struct LinearRegressionClusterOpt <: MonteCarloModelOpt
    G::Int64
    ng::Vector{Int64}
    ## Regressors std. dev.
    σ_z::Float64
    σ_ξ::Float64
    ## Errors std. dev
    σ_ϵ::Float64
    σ_α::Float64
    ## Third design parms
    p::Float64
    γ::Float64
    ## Null value
    β₀::Float64
    design::Int
end

function dispersion_ng(m::LinearRegressionClusterOpt)
    sum(m.ng^2)/sum(ng)^2
end


function icc_x(m::LinearRegressionClusterOpt)
        σ_z = m.σ_z
        σ_ξ = m.σ_ξ
        σ_ξ^2/(σ_ξ^2+σ_z^2)
end

function icc_u(m::LinearRegressionClusterOpt)
    σ_ϵ = m.σ_z
    σ_α = m.σ_ξ
    σ_z = m.σ_z
    γ   = m.γ
    p   = m.p
    σ_ξ = m.σ_ξ    
    if m.design < 3        
        σ_α^2/(σ_α^2+σ_ϵ^2)
    else
        (σ_α^2 + (1-p)*p*γ^2*σ_ξ^2)/(σ_α^2 + σ_ϵ^2 + (1-p)*p*γ^2*(1+σ_ξ^2))
    end
end

icc_x(m::LinearRegressionCluster) = icc_x(m.opt)
icc_u(m::LinearRegressionCluster) = icc_u(m.opt)

struct ConfInterval
    lo::Array{Float64,1}
    hi::Array{Float64,1}
end

function confinterval(m::Real, s::Real, zl::Real, zh::Real)
    m = float(m)
    (m+zl*s, m+zh*s)
end

function ConfInterval(m::Array{Float64}, σ::Array{Float64}, z::Float64 = 1.96)
    zs = z.*σ
    ConfInterval(m - zs, m + zs)
end

function ConfInterval(m::Array{Float64}, σ::Array{Float64},
                      zl::Array{Float64}, zh::Array{Float64})
    @assert length(m) == length(σ)
    @assert length(m) == length(zh)
    @assert length(zl) == length(zh)

    lo = m .+ zl.*σ
    hi = m .+ zh.*σ

    @assert all(hi.>=lo)
    ConfInterval(m .+ zl.*σ, m .+ zh.*σ)
end

Base.length(i::ConfInterval) = length(i.lo)

function average_length(i::Clusters.ConfInterval)
    l = 0.0
    n = length(i)
    @simd for j in 1:n
        @inbounds l += i.hi[j]-i.lo[j]
    end
    l/n
end

function Base.in(i::ConfInterval, a::Number)
    isin = BitArray(length(i))
    for j in 1:length(i)
        if (i.lo[j] <= a && i.hi[j] >= a)
            isin[j] = true
        else
            isin[j] = false
        end
    end
    isin
end

function coverage(i::Clusters.ConfInterval, a::Number)
    ifin = in(i, a)
    mean(ifin)
end

function initialize(::Type{LinearRegressionCluster}, opt)
    G = opt.G
    ng = opt.ng
    n = sum(ng)
    cl = vcat(map((i, n) -> repeat(i:i, outer = n), 1:G, ng)...)
    wts = vcat(map((i, n) -> repeat(1/n:1/n, outer = n), 1:G, ng)...)
    sqwts = sqrt.(wts)
    iter = mapreduce(x -> collect(1:x), vcat, ng)
    bstarts = [searchsorted(cl, j[2]) for j in enumerate(unique(cl))]
    y = Array{Float64}(n)
    X = Array{Float64}(n, 1)
    epsilon = similar(y)
    eta = Array{Float64}(G)
    D = similar(eta)
    sigma_e = similar(y)
    LinearRegressionCluster(y, X, epsilon, eta, D, sigma_e, cl, iter, bstarts, wts, sqwts, opt)
end

function simulate!(m::LinearRegressionCluster)
    clus = m.cl
    iter = m.iter

    ## Container
    X = m.X
    y = m.y
    D = m.D
    η = m.eta
    ϵ = m.epsilon
    σₑ= m.sigma_e

    ## Parameters
    β₀  = m.opt.β₀
    σ_ξ = m.opt.σ_ξ
    σ_z = m.opt.σ_z
    σ_ϵ = m.opt.σ_ϵ
    σ_α = m.opt.σ_α
    γ  = m.opt.γ
    p  = m.opt.p
    ρₓ = icc_x(m)
    κ  = sqrt((π/2)*(1-ρₓ))
    ς  = sqrt(ρₓ/(1-ρₓ))

    ## Simulate regressors

    randn!(η)
    scale!(η, σ_ξ)
    randn!(X)
    scale!(X, σ_z)
    for (i, (n, g)) in enumerate(zip(iter, clus))
        X[i] += η[g]
    end

    ## Simulate error

    randn!(ϵ)
    scale!(ϵ, σ_ϵ)
    ## Design == 2 (heteroskedastic)
    if m.opt.design == 2
        for i in eachindex(ϵ)
            ϵ[i] = κ.*sqrt(abs(X[i]))
        end
    end

    randn!(η)
    scale!(η, σ_α)
    for (i, (n, g)) in enumerate(zip(iter, clus))
        ϵ[i] += η[g]
    end

    ## Design == 3 (heterogenous)
    if m.opt.design == 3
        rand!(η)
        for i in eachindex(D)
            D[i] = ifelse(η[i] .<= p, 1-p, p)
        end
        for (i, (n, g)) in enumerate(zip(iter, clus))
            ϵ[i] += γ*(X[i]*D[g])
        end
    end

    ## Change this when the power is sought
    @simd for i in eachindex(y)
        @inbounds y[i] = X[i].*β₀ + ϵ[i]
    end

end

function montecarlo(m::Type{T} where T <: MonteCarloModel, opt::MonteCarloModelOpt; simulations::Int64 = 1000)
    model = initialize(m, opt)
    simulate!(model)
    out = estimatemodel(model)
    res = GrowableArray(out)
    sizehint!(res,simulations)
    @inbounds for j in 1:simulations
        simulate!(model)
        push!(res, estimatemodel(model))
    end
    ## Add information about the experiment


    res = convert(DataFrame, convert(Array,res))
    names!(res,
    [:theta_u, :V1_u, :V2_u, :V3_u, :V4_u, :V5_u,
     :G_u, :k_u,
     :qu_025, :qu_975,
     :qu_050, :qu_950,
     :qu0_025, :qu0_975,
     :qu0_050, :qu0_950,
     :theta_w, :V1_w, :V2_w, :V3_w, :V4_w, :V5_w,
     :G_w, :k_w,
     :qw0_025, :qw0_975, :qw0_050, :qw0_950])

end


function estimatemodel(m::LinearRegressionCluster)
    y = m.y
    X = m.X
    cl = m.cl
    iter = m.iter
    w  = Val{:weighted}
    uw = Val{:unweighted}
    #try
        fitted_u = fit(GeneralizedLinearModel, X, y, Normal(), IdentityLink())

        theta_u = first(coef(fitted_u))
        V1_u = first(stderr(fitted_u))
        V2_u = first(stderr(fitted_u, HC1()))

        V3_u = faststderr(fitted_u, m, CRHC1(cl), uw)
        V4_u = faststderr(fitted_u, m, CRHC2(cl), uw)
        V5_u = faststderr(fitted_u, m, CRHC3(cl), uw)

        fitted_w = fit(GeneralizedLinearModel, X, y, Normal(), IdentityLink(), wts = m.wts)
        theta_w = first(coef(fitted_w))
        V1_w = fastiid(fitted_w, m)
        V2_w = first(stderr(fitted_w, HC1()))

        V3_w = faststderr(fitted_w, m, CRHC1(cl), w)
        V4_w = faststderr(fitted_w, m, CRHC2(cl), w)
        V5_w = faststderr(fitted_w, m, CRHC3(cl), w)

        G_u, k_u = kappastar(fitted_u, m)
        G_w, k_w = kappastar(fitted_w, m)

        _, _, qw = fastwildboot_nonull(fitted_w, m, Rademacher(), rep = 499)
        _, _, qu = fastwildboot_nonull(fitted_u, m, Rademacher(), rep = 499)

        _, _, qw0 = fastwildboot_null(fitted_w, m, Rademacher(),  rep = 499)
        _, _, qu0 = fastwildboot_null(fitted_u, m, Clusters.Rademacher(),  rep = 499)

        [theta_u, V1_u, V2_u, V3_u, V4_u, V5_u, G_u, k_u,
        qu[1], qu[4], qu[2], qu[3],
        qu0[1], qu0[4], qu0[2], qu0[3],
        theta_w, V1_w, V2_w, V3_w, V4_w, V5_w, G_w, k_w,
        qw0[1], qw0[4], qw0[2], qw0[3]]
    # catch
    #     fill(NaN, 28)
    # end
end

##=
## Standard Errors
##=
function fastiid(f::GLM.AbstractGLM, m::LinearRegressionCluster)
    r = f.rr.wrkresid.*m.sqwts
    ichol = inv(cholfact(f.pp))
    sqrt(ichol[1]*mean(abs2.(r)))
end

function faststderr(f::GLM.AbstractGLM, m::LinearRegressionCluster, v::CovarianceMatrices.CRHC, w::T) where T
    B = fastmeat(f, m, v, w)
    A = first(inv(f.pp.chol))
    sqrt(B*A^2)
end

function fastmeat(f::GLM.AbstractGLM, m::LinearRegressionCluster, v::CovarianceMatrices.CRHC,
                  w::Type{Val{:unweighted}})
    cl = m.cl
    bstarts = m.bstarts
    ichol  = inv(cholfact(f.pp))::Array{Float64, 2}
    e = copy(CovarianceMatrices.wrkresid(f.rr))
    CovarianceMatrices.adjresid!(v, m.X, e, ichol, bstarts)
    fastclusterize(m.X.*e, bstarts)
end

function fastmeat(f::GLM.AbstractGLM, m::LinearRegressionCluster, v::CovarianceMatrices.CRHC,
                  w::Type{Val{:weighted}})
    cl = m.cl
    bstarts = m.bstarts
    ichol  = inv(cholfact(f.pp))::Array{Float64, 2}
    e = copy(CovarianceMatrices.wrkresid(f.rr))
    X = copy(m.X)
    broadcast!(*, X, X, m.sqwts)
    broadcast!(*, e, e, m.sqwts)
    CovarianceMatrices.adjresid!(v, X, e, ichol, bstarts)
    fastclusterize(X.*e, bstarts)
end

function fastmeat(ichol, e, X, m::LinearRegressionCluster, v::CovarianceMatrices.CRHC)
    bstarts = m.bstarts
    cl  = m.cl
    CovarianceMatrices.adjresid!(v, X, e, ichol, bstarts)
    fastclusterize(X.*e, bstarts)
end

function fastclusterize(U, bstarts)
    M = 0.0
    G = length(bstarts)
    for g = 1:G
        s = 0.0
        for i = bstarts[g]
            @inbounds s += U[i, 1]
        end
        M += s^2
    end
    M
end

####################################################################################
##
##
####################################################################################
function fastgamma(f::GLM.AbstractGLM, m::LinearRegressionCluster)
    A = first(inv(f.pp.chol))
    X = copy(m.X)
    #w = f.rr.wts
    X .= X.*m.sqwts
    fastgamma(A, X, m.bstarts)
end


function kappastar(f::GLM.AbstractGLM, m::LinearRegressionCluster)
    G = m.opt.G
    Gamma = fastgamma(f, m)
    (Gamma, G/(1+Gamma))
end

function fastgamma(XXinv, X, bstarts)
    M = Array{Float64}(length(bstarts))
    kappaclusterize!(M, X, bstarts)
    gamma_g = XXinv^2.*M
    gamma_bar = mean(gamma_g)
    mean((gamma_g-gamma_bar).^2)/gamma_bar.^2
end


function kappaclusterize!(M, U, bstarts)
    ## Assumes that U is ng x 1
    G = length(bstarts)
    for m = 1:G
        s = 0.0
        for i = bstarts[m]
            @inbounds s += U[i, 1]
        end
        M[m] = s^2
    end
end

function fastwildboot_null(f::GLM.AbstractGLM, m::LinearRegressionCluster, WT::WildWeights; rep::Int = 499)
    ## Wildbootstrap with null imposed
    wts  = f.rr.wts
    is_weighted = !all(wts.==1.0)
    sqwts = m.sqwts
    betahat = first(coef(f))
    Hₙ = first(inv(f.pp.chol))
    if is_weighted
        Y = f.rr.y.*sqwts
        X = m.X.*sqwts
        uhat = copy(Y)
    else
        Y = f.rr.y
        X = copy(m.X)
        uhat = Y
    end
    β, σ = Clusters._wildboot_null(Y, X, uhat, Hₙ, WT, rep, m)
    q = quantile(β./σ, [.025, .05, .95,.975])
    (β, σ, q)
end

function fastwildboot_nonull(f::GLM.AbstractGLM, m::LinearRegressionCluster, WT::WildWeights; rep::Int = 999)
    wts  = f.rr.wts
    is_weighted = !all(wts.==1)
    sqwts = m.sqwts
    betahat = first(coef(f))
    Hₙ = first(inv(f.pp.chol))
    uhat = CovarianceMatrices.wrkresidwts(f.rr)
    if is_weighted
        Y = f.rr.y.*sqwts
        X = m.X.*sqwts
    else
        Y = f.rr.y
        X = copy(m.X)
    end
    β, σ = Clusters._wildboot_nonull(Y, X, uhat, betahat, Hₙ, WT, rep, m)
    q = quantile((β-betahat)./σ, [.025, .05, .95,.975])
    (β, σ, q)
end

function _wildboot_null(Y, X, uhat, Hₙ, WT, rep, m::LinearRegressionCluster)
    ustar = similar(uhat)
    bstarts = m.bstarts
    G = length(bstarts)
    ## Containers
    W  = Array{Float32}(G)
    β  = Array{Float64}(rep)
    σ  = Array{Float64}(rep)
    cr = CRHC1(m.cl)
    cX = similar(X)
    for h in 1:rep
        Clusters.wbweights!(WT, W)
        s = 0.0
        for r = 1:G
            @inbounds for i = bstarts[r]
                ustar[i] = uhat[i]*W[r]
                s += X[i]*ustar[i]
            end
        end
        ##
        β[h] = Hₙ*s
        ## Calculate the variance        
        @simd for j in eachindex(ustar)
            @inbounds ustar[j] -= X[j]*Hₙ*s
        end
        B = Clusters.fastmeat(Hₙ, ustar, copy!(cX, X), m, cr)
        σ[h] = sqrt(B*Hₙ^2)
    end
    (β, σ)
end

function _wildboot_nonull(Y, X, uhat, betahat, Hₙ, WT, rep, m::LinearRegressionCluster)
    ustar = similar(uhat)
    bstarts = m.bstarts
    G = length(bstarts)
    ## Containers
    W = Array{Float32}(G)
    β = Array{Float64}(rep)
    σ = Array{Float64}(rep)
    cr = CRHC1(m.cl)
    cX = similar(X)
    for h in 1:rep
        Clusters.wbweights!(WT, W)
        s = 0.0
        for r = 1:G
            @inbounds for i = bstarts[r]
                ustar[i] = uhat[i]*W[r]
                s += X[i]*ustar[i]
            end
        end
        ##
        β[h] = betahat .+ Hₙ*s
        ## Calculate the variance
        @simd for j in eachindex(ustar)
            @inbounds ustar[j] -= X[j]*Hₙ*s
        end
        B = Clusters.fastmeat(Hₙ, ustar, copy!(cX, X), m, cr)
        σ[h] = sqrt(B*Hₙ^2)
    end
    (β, σ)
end

function wbweights!(x::Rademacher, W::Vector{Float32})
    G = length(W)
    for m = 1:G
        W[m] = rand() < 0.5 ? 1.0 : -1.0
    end
end

function wbweights!(x::TwoPoint, W::Vector{Float32})
    G = length(W)
    s5 = √5
    p = (s5 + 1.0)/(2*s5)
    for m = 1:G
        s = 0.0
        W[m] = rand() < p ? (1 + s5)/2 : -(s5-1)/2
    end
end

export LinearRegressionCluster, simulate!, montecarlo, initialize!, ConfInterval, coverage, average_length

end
