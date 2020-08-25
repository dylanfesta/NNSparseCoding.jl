using Pkg
pkg"activate ."
using NNSparseCoding; const N = NNSparseCoding;
using Test
using Plots;
theme(:dark);
using BenchmarkTools
using StatsBase,Statistics
using MultivariateStats

function scatty(M1,M2)
  scatter(M1[:], M2[:]; leg = false, ratio = 1)
  return plot!([extrema(M1)...].* 1.1, identity; linewidth = 3)
end
# function to plot one or more black/mytransfe images
function showbw(mat::Matrix{<:Real},_title="")
  heatmap(mat, color=:grays,ratio=1,cbar=false,axis=false,title=_title)
end
function showbw_many(title::String,mats...)
  mats = [mats...] # convert tuple to array
  padval=maximum(maximum,mats)
  framesize = size(mats[1],1)
  padmat = fill(padval, framesize,round(Int,0.1framesize))
  padmats = repeat([padmat,],inner=length(mats))
  matspads = permutedims(hcat(mats,padmats))[:]
  imgrow = hcat(matspads[1:end-1]...)
  showbw(imgrow , title)
end
##
#
# const dirimg = abspath(@__DIR__,"..","data")
# const sizesq=315
# all_images = N.read_natural_images(dirimg,sizesq ; verbose=true)
#
# const npatches = 1000
# const patch_size = 40
#
# patches = N.get_random_patches(npatches,all_images,patch_size)
#
#
# patchmat = N.merge_mats(patches)
#
# mean(patchmat;dims=2) |> extrema
# std(patchmat;dims=2) |> extrema
# cov(patchmat;dims=2) |> heatmap
#
#
# ##
# showbw_many("done!",sample(patches,5)...)
##

n = 80
d = n
k = 500
A,S = N.initialize_mats(n,d,k)
map!(s-> rand() < 0.95 ? 0.0 : 3s ,S,S)
X = A*S
broadcast!(-,X,X,mean(X;dims=2))

##
mytransf,Mp,Mm = N.preprocess_plus_minus(X,N.PreprocessPCA(d))

##

lambda=0.1
nsteps=10_000
stepsize=0.2
optthresh=1E-4
op = N.nnsc_optimize(Mp,d,lambda ;
  step_zero=stepsize, nstepmax=nsteps,
  opt_threshold=optthresh,verbose=true)

om = N.nnsc_optimize(Mm,d,lambda ;
  step_zero=stepsize, nstepmax=nsteps,
  opt_threshold=optthresh,verbose=true)

##

scatty(X,N.revert_plus_minus(op.AS,om.AS,mytransf))

extrema( X .- N.revert_plus_minus(op.AS,om.AS,mytransf))

###
mytransf,Mp,Mm = N.preprocess_plus_minus(X,N.PreprocessWhiten)

##

lambda=0.1
nsteps=10_000
stepsize=0.2
optthresh=1E-4
op = N.nnsc_optimize(Mp,d,lambda ;
  step_zero=stepsize, nstepmax=nsteps,
  opt_threshold=optthresh,verbose=true)

om = N.nnsc_optimize(Mm,d,lambda ;
  step_zero=stepsize, nstepmax=nsteps,
  opt_threshold=optthresh,verbose=true)

##

scatty(X,N.revert_plus_minus(op.AS,om.AS,mytransf))

extrema( X .- N.revert_plus_minus(op.AS,om.AS,mytransf))

boh = N.revert_plus_minus(op.S,zero(op.S),mytransf)
