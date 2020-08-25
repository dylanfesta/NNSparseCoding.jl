using Pkg
pkg"activate ."
using NNSparseCoding; const N = NNSparseCoding;
using Test
using Plots;
theme(:dark);
using BenchmarkTools
using StatsBase,Statistics

function scatty(M1,M2)
  scatter(M1[:], M2[:]; leg = false, ratio = 1)
  return plot!([extrema(M1)...].* 1.1, identity; linewidth = 3)
end
# function to plot one or more black/white images
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

const dirimg = abspath(@__DIR__,"..","data")
const sizesq=315
all_images = N.read_natural_images(dirimg,sizesq ; verbose=true)

const npatches = 1000
const patch_size = 40

patches = N.get_random_patches(npatches,all_images,patch_size)


patchmat = N.merge_mats(patches)

mean(patchmat;dims=2) |> extrema
std(patchmat;dims=2) |> extrema
cov(patchmat;dims=2) |> heatmap


##
showbw_many("done!",sample(patches,5)...)
