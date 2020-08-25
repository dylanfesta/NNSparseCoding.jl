module NNSparseCoding
using Random,Distributions
using LinearAlgebra
using Images, FileIO
using MultivariateStats

struct NNSC_optimizer{T}
  X::Matrix{T}
  AS::Matrix{T}
  A::Matrix{T}
  S::Matrix{T}
  lambda::T
end

function NNSC_optimizer(X::Matrix{<:Real},d::Integer,
    lambda::Real)
  nfull,nsampl=size(X)
  A,S = initialize_mats(X,d)
  project!(A)
  NNSC_optimizer(X,A*S,A,S,lambda)
end
function nnsc_objective(o::NNSC_optimizer)
  return 0.5 * mapreduce( (x,x̃) -> (x-x̃)^2,
    +,o.AS,o.X) + o.lambda * sum(o.S)
end

function nnsc_objective(X::M,A::M,S::M,lambda::Real) where M <: Matrix{<:Real}
  X̃ = A*S
  @. X̃ = (X̃ - X)^2
  return 0.5*sum(X̃) + lambda * sum(S)
end

function project!(A::Matrix{<:Real})
  # no negative values
  for i in eachindex(A)
    @inbounds begin
      if A[i] < 0.0
        A[i] = 0.0
  end end end
  # normalize columns
  for c in 1:size(A,2)
    @inbounds A[:,c] ./= norm(A[:,c])
  end
  return nothing
end
function projectA!(o::NNSC_optimizer)
  project!(o.A)
  return nothing
end


"""
    initialize_mats(n,b,k)
`n` : number of dimensions
`b` : number of basis vectors
`k` : number of samples
"""
function initialize_mats(n::I,b::I,k::I) where I<:Integer
  A = rand(distr_initA,(n,b))
  project!(A)
  B = rand(distr_initB,(b,k))
  return A,B
end
const distr_initA = Exponential(1.0)
const distr_initB = Exponential(1.0)

function initialize_mats(X,d)
  (nfull,ndat) = size(X)
  Xv = X[:]
  A = sqrt.(sample(Xv,(nfull,d))./ d)
  B = sqrt.(sample(Xv,(d,ndat)) ./ d)
  return A,B
end

function updateS!(S::M,A::M,X::M,lambda::Real) where M<:Matrix{<:Real}
  return S .*= (A' * X) ./ (A' * A * S .+ lambda)
end
function updateS!(o::NNSC_optimizer)
  o.S .*= o.A'*o.X
  o.S ./= (o.A'*o.AS .+ o.lambda)
  updateAS!(o)
  return nothing
end
@inline function updateAS!(o::NNSC_optimizer)
  mul!(o.AS,o.A,o.S)
  return nothing
end

function updateA!(A::M,S::M,X::M,lambda::Real,
    stepsize::Real,
    obj_pre::Union{Nothing,<:Real}=nothing) where M<:Matrix{<:Real}
  obj_pre = something(obj_pre, nnsc_objective(X,A,S,lambda))
  Apost = A .- stepsize .* ((A*S-X)*S')
  project!(Apost)
  obj_post = nnsc_objective(X,Apost,S,lambda)
  if (obj_post < obj_pre || stepsize < 1E-9)
    copy!(A,Apost)
    return stepsize
  else
    return updateA!(A,S,X,lambda,0.5*stepsize,obj_pre)
  end
end

function updateA!(o::NNSC_optimizer,stepsize::Real,obj_pre::Real)
  Apre = copy(o.A)
  ASpre = copy(o.AS)
  o.A .-= stepsize .* ((o.AS-o.X)*o.S')
  projectA!(o)
  updateAS!(o)
  obj_post = nnsc_objective(o)
  delta_obj = obj_pre - obj_post
  if (delta_obj > 0.0 || stepsize < 1E-9)
    return (stepsize,obj_post,delta_obj)
  else
    copy!(o.A,Apre)
    copy!(o.AS,ASpre)
    return updateA!(o,0.5*stepsize,obj_pre)
  end
end

function nnsc_optimize(X::Matrix{<:Real},d::Integer,
  lambda::Real ; step_zero::Real=0.05, nstepmax::Integer=10_000,
    opt_threshold::Real=1E-4,verbose::Bool=false)
  o = NNSC_optimizer(X,d,lambda)
  stepsize=step_zero
  obj_temp = nnsc_objective(o)
  delta_obj = Inf
  nsteps=0
  while ( (nsteps < nstepmax) && (delta_obj > opt_threshold) )
    nsteps +=1
    (stepsize,obj_temp,delta_obj) = updateA!(o, stepsize,obj_temp)
    updateS!(o)
    if verbose && (nsteps == nstepmax)
      @info """NNSC completed:
      threshold NOT reached: $opt_threshold
      after : $nsteps (max) iterations
      objective : $obj_temp
      """
    elseif verbose && (delta_obj <= opt_threshold)
      @info """NNSC completed:
      threshold reached: $opt_threshold
      after : $nsteps iterations
      objective : $obj_temp
      """
    end
  end
  o.S[o.S .< 1E-7] .= 0.0
  return o
end

#################
# What follows is manipulaton of image patches and files


# regularization types
abstract type ReguType end
# thou shall not broadcast!
Base.Broadcast.broadcastable(reg::ReguType)=Ref(reg)

struct NoRegu <: ReguType end
struct StandardRegu <: ReguType end

function regularize_image(img,rt::Type{NoRegu})::Matrix{Float64}
    return convert(Array{Float64},Gray.(img))
end
function regularize_image(img,rt::Type{StandardRegu})::Matrix{Float64}
    im=regularize_image(img,NoRegu)
    _mu,_std = mean(im),std(im)
    @. im = (im -_mu)/_std
    return im
end



"""

"""
function read_natural_images(
    dir,
    sz,
    (regu::Type{<:ReguType}) = StandardRegu;
    rotate::Bool = true,
    verbose::Bool = false,
    format::String = ".jpg")::Vector{Matrix{Float64}}
  @assert isdir(dir) "directory $dir not found!"
  filenames = filter(f-> occursin(Regex("$format\\b"),f),  readdir(dir))
  filenames = joinpath.(dir,filenames)
  good_size(img, sz) = min(size(img)...) >= sz
  @assert !isempty(filenames) "no files with extension $format found in $dir"
  ret = Matrix{Any}[]
  _rot_fun = rotate ? rotation_functions : [identity]
  nfiles = length(filenames)
  for (k, file) in enumerate(filenames)
    verbose && println("reading file $k/$(nfiles)")
    img = load(file)
    if good_size(img, sz)
      for rot in _rot_fun
        push!(ret, rot(img))
      end
    else
      @warn "image $file has size smaller than $sz , ignored"
    end
  end
  return cut_patch.(regularize_image.(ret,regu),Ref(sz))
end


# rotate images exactly
function rot_90(img)
    r,c=size(img)
    out=similar(img,c,r)
    for cc in 1:c
        out[cc,:] = img[end:-1:1,cc]
    end
    out
end
function rot_m90(img)
    r,c=size(img)
    out=similar(img,c,r)
    for (cc,mcc) in zip(1:c,c:-1:1)
        out[cc,:] = img[:,mcc]
    end
    out
end
function rot_180(img)
    r,c=size(img)
    out=similar(img)
    for (cc,mcc) in zip(1:c,c:-1:1)
        out[:,cc] = img[r:-1:1,mcc]
    end
    out
end
const rotation_functions = [identity , rot_90, rot_m90, rot_180 ]

"""
        function cut_patch(img, sz)

cuts a random square section of size `sz` from the matrix `img`
"""
function cut_patch(img::AbstractArray{T}, sz) where T
  _rows, _cols = size(img)
  _crows,_ccols =_rows-sz+1, _cols-sz+1
  i,j= rand(1:_crows), rand(1:_ccols)
  return img[i:i+sz-1,j:j+sz-1]
end

"""
    sampling_tiles(n_samples::Integer, images::Vector{Matrix{Float64}},
        patch_size::Integer)
Generates `n_samples` patches of size `patch_size`,
"""
function get_random_patches(n_samples::Integer, images::Vector{Matrix{Float64}},
    patch_size::Integer)
  n_img=length(images)
  @assert !isempty(images) "images are missing!"
  #check how many patches per image
  n_full,n_rem = divrem(n_samples,n_img)
  n_imgsampl = fill(n_full,n_img)
  # add the rem, picking them randomly!
  if n_rem>0
    idx_rem=sample(1:n_img , n_rem ;replace=false)
    n_imgsampl[idx_rem] .+= 1
  end
  @assert  sum(n_imgsampl) == n_samples "wrong selection?"
  ret = Vector{Matrix{Float64}}(undef,n_samples)
  k=0
  for (im,ns) in zip(images,n_imgsampl) , s in 1:ns
    k+=1
    ret[k]=cut_patch(im,patch_size)
  end
  return ret
end

function regularize01!(mat)
    l,u = extrema(mat)
    @. mat = (mat-l)/(u-l)
end

function merge_mats(mats::Vector{Matrix{R}};
   zeromean::Bool=true) where R<:Real
  sz = size(mats[1],1)
  nmats = length(mats)
  ret = Matrix{R}(undef,sz*sz,nmats)
  for (k,mat) in enumerate(mats)
    ret[:,k] .= mat[:]
  end
  if zeromean
    broadcast!(-,ret,ret,mean(ret;dims=2))
  end
  return ret
end

# pretty much wrappers to MultivariateStats

function plus_minus_parts(M::Array{T}) where T<:Real
  plus,minus = zero(M),zero(M)
  for i in eachindex(M)
    @inbounds begin
    if M[i] > 0
      plus[i] = M[i]
    else
      minus[i] = -M[i]
  end end end
  return plus,minus
end

abstract type Preprocess end

struct PreprocessWhiten <: Preprocess end
function preprocess_plus_minus(X,::Type{PreprocessWhiten})
  M = fit(Whitening,X)
  @assert all(x-> isapprox(x,0.0 ; atol=1E-6),mean(M)) "The input should have zero mean!"
  return (M, plus_minus_parts(transform(M,X))...)
end
function revert_plus_minus(Mp,Mm,W::Whitening)
  return  W.W' \ (Mp - Mm)
end


struct PreprocessPCA <: Preprocess
  d::Integer
end

function preprocess_plus_minus(X,p::PreprocessPCA)
  M = fit(PCA,X;maxoutdim=p.d)
  @assert all(x-> isapprox(x,0.0 ; atol=1E-6),mean(M)) "The input should have zero mean!"
  return (M, plus_minus_parts(transform(M,X))...)
end
function revert_plus_minus(Mp,Mm,p::PCA)
  return reconstruct(p, Mp - Mm)
end

struct PreprocessNot <: Preprocess end
function preprocess_plus_minus(X,::Type{PreprocessNot})
  @assert all(x-> isapprox(x,0.0 ; atol=1E-6),mean(X;dims=2)) "The input should have zero mean!"
  return (PreprocessNot, plus_minus_parts(X)...)
end
function revert_plus_minus(Mp,Mm,::Type{PreprocessNot})
  return (Mp - Mm)
end




end #module
