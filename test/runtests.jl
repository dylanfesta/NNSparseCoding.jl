using NNSparseCoding ; const N=NNSparseCoding
using Test
using Statistics

@testset "Initialization" begin
    n = 200
    b = n
    k = 80
    A,S = N.initialize_mats(n,b,k)
    @test all(A .>=  0.)
    @test all(S .>=  0.)
    @test N.nnsc_objective(A*S,A,S,0.0) ≈ 0.
    @test N.nnsc_objective(rand(n,k),A,S,1.0) > 0.
    # projection
    M = randn(n,k)
    MM = copy(M)
    N.project!(M)
    @test all(M[MM .< 0.] .== 0.0)
    mynorms = mapslices(N.norm,M;dims=1)
    @test all(isapprox.(mynorms,1.0))
end

@testset "Convergence of S alone (basis coef)" begin
    n = 80
    b = n
    k = 100
    A,S = N.initialize_mats(n,b,k)
    # let's have a sparse S
    map!(s-> rand() < 0.8 ? 0.0 : 3s ,S,S)
    X = A*S
    _,S̃ = N.initialize_mats(n,b,k)
    lambda=0.1
    nsteps=5000
    for _ in 1:nsteps
      N.updateS!(S̃,A,X,lambda)
    end
    @test all(isapprox.(S,S̃;atol=0.5))
end

@testset "Convergence of S, new interface" begin
    n = 80
    b = n
    k = 100
    lambda=0.1
    A,S = N.initialize_mats(n,b,k)
    # let's have a sparse S
    map!(s-> rand() < 0.8 ? 0.0 : 3s ,S,S)
    X = A*S
    obj = N.NNSC_optimizer(X,b,lambda)
    copy!(obj.A,A)
    N.updateAS!(obj)
    nsteps=5000
    for _ in 1:nsteps
      N.updateS!(obj)
    end
    @test all(isapprox.(S,obj.S;atol=0.5))
end

@testset "Convergence of A alone" begin
    n = 80
    b = n
    k = 100
    A,S = N.initialize_mats(n,b,k)
    # let's have a sparse S
    map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
    X = A*S
    Ã,_ = N.initialize_mats(n,b,k)
    lambda=0.1
    nsteps=3_000
    stepsize=0.1
    for _ in 1:nsteps
      temp = N.updateA!(Ã,S,X,lambda,stepsize)
      stepsize=temp
    end
    @test all(isapprox.(A,Ã;atol=0.3))
end
# 
# @testset "Convergence of A, other interface" begin
#     n = 80
#     b = n
#     k = 100
#     lambda=0.1
#     A,S = N.initialize_mats(n,b,k)
#     # let's have a sparse S
#     map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
#     X = A*S
#     obj = N.NNSC_optimizer(X,b,lambda)
#     copy!(obj.S,S)
#     N.updateAS!(obj)
#     nsteps=5_000
#     stepsize=0.02
#     objval = N.nnsc_objective(obj)
#     for _ in 1:nsteps
#       stepsize,objval,_= N.updateA!(obj,stepsize,objval)
#     end
#     @test all(isapprox.(A,obj.A;atol=0.3))
# end
#
# @testset "Optimization of A and S" begin
#     n = 80
#     b = n
#     k = 300
#     A,S = N.initialize_mats(n,b,k)
#     # let's have a sparse S
#     map!(s-> rand() < 0.95 ? 0.0 : 3s ,S,S)
#     X = A*S
#     lambda=0.1
#     o = N.NNSC_optimizer(X,b,lambda)
#     N.nnsc_objective(o)
#     nsteps=10_000
#     stepsize=0.02
#     obj_temp = N.nnsc_objective(o)
#     for _ = 1:nsteps
#       (stepsize,obj_temp,_) = N.updateA!(o, stepsize,obj_temp)
#       N.updateS!(o)
#     end
#     @test all(isapprox.(X,o.AS;atol=1.0))
# end
#
@testset "Transformations" begin
    n = 80
    b = n
    k = 150
    A,S = N.initialize_mats(n,b,k)
    map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
    X = A*S
    broadcast!(-,X,X,mean(X;dims=2))
    whit,Mp,Mm = N.preprocess_plus_minus(X,N.PreprocessWhiten)
    @test (!any(Mp .< 0)) && (!any(Mm .< 0))
    X̃ = N.revert_plus_minus(Mp,Mm,whit)
    @test all(isapprox.(X,X̃))
    pcadims = 40
    whit, Mp,Mm = N.preprocess_plus_minus(X,N.PreprocessPCA(pcadims))
    @test (!any(Mp .< 0)) && (!any(Mm .< 0))
    X̃ = N.revert_plus_minus(Mp,Mm,whit)
    @test all(isapprox.(X,X̃ ; atol=2))
    whit,Mp,Mm = N.preprocess_plus_minus(X,N.PreprocessNot)
    @test (!any(Mp .< 0)) && (!any(Mm .< 0))
    X̃ = N.revert_plus_minus(Mp,Mm,whit)
    @test all(isapprox.(X,X̃))
end
