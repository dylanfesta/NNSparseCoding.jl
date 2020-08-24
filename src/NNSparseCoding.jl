module NNSparseCoding
using Random,Distributions
using LinearAlgebra

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




end #module
