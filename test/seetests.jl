using Pkg
pkg"activate ."
using NNSparseCoding;
const N = NNSparseCoding;
using Test
using Plots;
theme(:dark);
using BenchmarkTools

function scatty(M1,M2)
  scatter(M1[:], M2[:]; leg = false, ratio = 1)
  return plot!([extrema(M1)...].* 1.1, identity; linewidth = 3)
end
##

n = 80
b = n
k = 100
lambda=0.1
A,S = N.initialize_mats(n,b,k)
# let's have a sparse S
map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
X = A*S
obj = N.NNSC_optimizer(X,b,lambda)
copy!(obj.S,S)
N.updateAS!(obj)
nsteps=5_000
_ = let stepsize=0.02, objval= N.nnsc_objective(obj)
  for _ in 1:nsteps
    stepsize,objval,_= N.updateA!(obj,stepsize,objval)
  end
end
scatty(obj.A,A)
@test all(isapprox.(A,obj.A;atol=0.3))





n = 80
b = n
k = 100
lambda=0.1
A,S = N.initialize_mats(n,b,k)
# let's have a sparse S
map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
X = A*S
obj = N.NNSC_optimizer(X,b,lambda)
copy!(obj.S,S)
N.updateAS!(obj)
nsteps=5_000
stepsize=0.02
_ = let stepsize=0.02, objval = N.nnsc_objective(obj)
  for _ in 1:nsteps
    stepsize,objval= N.updateA!(obj,stepsize,objval)
  end
end
@test all(isapprox.(A,obj.A;atol=0.3))


##


n = 80
b = n
k = 300
A,S = N.initialize_mats(n,b,k)
# let's have a sparse S
map!(s-> rand() < 0.95 ? 0.0 : 3s ,S,S)
X = A*S
lambda=0.1
o = N.NNSC_optimizer(X,b,lambda)
N.nnsc_objective(o)
nsteps=10_000
_ = let stepsize=0.02 , obj_temp = N.nnsc_objective(o)
  for _ = 1:nsteps
    (stepsize,obj_temp) = N.updateA!(o, stepsize,obj_temp)
    N.updateS!(o)
end end
scatty(X,o.AS)

extrema(X .- o.AS)

@test all(isapprox.(X,o.AS;atol=1.0))

extrema(o.S)
Stemp = copy(S)
Stemp[S .< 1E-9] .= 0.0

histogram(Stemp[:])

##
n = 80
b = n
k = 100
lambda=0.1
A,S = N.initialize_mats(n,b,k)
# let's have a sparse S
map!(s-> rand() < 0.85 ? 0.0 : 3s ,S,S)
X = A*S
obj = N.NNSC_optimizer(X,b,lambda)
copy!(obj.S,S)
N.updateAS!(obj)
nsteps=5_000
_ = let o=obj, stepsize=0.001 , obj_temp = N.nnsc_objective(o)
  for _ = 1:nsteps
    (stepsize,obj_temp) = N.updateA!(o, stepsize,obj_temp)
    #N.updateS!(o)
end end

obj.A

((o.AS-o.X)*o.S')
##
@test all(isapprox.(X,X̃;atol=1.5))

##
using BenchmarkTools

function f1(A)
  A[A.<0.] .= 123.0
  return A
end
function f2(A)
  for i in eachindex(A)
    @inbounds if A[i] < 0.0
     @inbounds A[i] = 123.0
    end
  end
  return A
end

function f3(A)
  f(x::Real)::Real = x < 0.0 ? 123. : x
  return map!(f, A,A)
end

A = -rand(1000,1000)
@time f1(A)
A = -rand(1000,1000)
@time f2(A)
A = -rand(1000,1000)
@time f3(A)

@btime f1($A)
@btime f2($A)
@btime f3($A)
