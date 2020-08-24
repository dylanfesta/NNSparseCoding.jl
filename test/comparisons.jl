using NMF
using Pkg
pkg"activate ."
using NNSparseCoding; const N = NNSparseCoding;
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
d = n
k = 300
A,S = N.initialize_mats(n,d,k)
# let's have a sparse S
map!(s-> rand() < 0.9 ? 0.0 : 3s ,S,S)
X = A*S
lambda=0.1
nsteps=10_000
stepsize=0.2
optthresh=1E-4
o = N.nnsc_optimize(X,d,lambda ;
  step_zero=stepsize, nstepmax=nsteps,
  opt_threshold=optthresh,verbose=true)

extrema(o.AS .- X)
##
# X2,res = let res = nnmf(X,d)
#   (res.W * res.H,res)
# end

(W,H) = let (W,H) = NMF.nndsvd(X,d ;variant=:a),
  alg = NMF.CoordinateDescent{Float64}(
      l₁ratio=0.8,α=10.0,regularization=:both)
  res = NMF.solve!(alg,X,W,H)
  (res.W,res.H)
end
X2=W*H

scatty(X2,X)
scatty(o.AS,X)

histogram(o.S[:])
histogram(H[:])

heatmap(o.S)
heatmap(H)
