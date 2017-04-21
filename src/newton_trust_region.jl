type NewtonTrustRegionState{T,N}
    x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    s::Array{T,N}
    z::Vector{T}
    d::Vector{T}
    δ::T
    η::T
    ρ::T
end

function svdminvec!(out::Vector, L::LowerTriangular)
    n = size(L, 1)
    LL = L.data
    @assert n == length(out)
    @inbounds begin
        out[1] = inv(LL[1,1])
        for i in 2:n
            ltx = zero(eltype(L))
            for j in 1:i - 1
                ltx += LL[i,j]*out[j]
            end
            out[i] = (-sign(ltx) - ltx)/LL[i,i]
        end
    end
    Ac_ldiv_B!(L, out)

    nrm = norm(out)
    scale!(out, inv(nrm))

    # Calculate norm of L'x
    nrm = zero(nrm)
    for j in 1:n
        acc = zero(nrm)
        for i in j:n
            @inbounds acc += LL[i,j]'*out[i]
        end
        nrm += acc*acc
    end

    return sqrt(nrm)
end

@inline function diagmin(A::StridedMatrix)
    n = LinAlg.checksquare(A)
    mn = typemax(eltype(A))
    @inbounds for i in 1:n
        mn = min(mn, A[i,i])
    end
    return mn
end

function quadform(C::Cholesky2, x::StridedVector)
    n = length(x)
    @assert size(C, 1) == n
    nrm = zero(promote_type(eltype(C), eltype(x)))
    if C.uplo == 'L'
        LL = C.factors
        for j in 1:n
            acc = zero(nrm)
            for i in j:n
                @inbounds acc += LL[i,j]'*x[i]
            end
            nrm += acc*acc
        end
    else
        error("not implemented yet")
    end
    return nrm
end

function restore!(H, d)
    LinAlg.copytri!(H, 'U')
    for i in 1:length(d)
        @inbounds H[i,i] = d[i]
    end
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of Nocedal and Wright.
# This is appropriate for Hessians that you factorize quickly.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  δ:  The trust region size, ||s|| <= δ
#  s: Memory allocated for the step size, updated in place
#  tolerance: The convergence tolerance for root finding
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  λ - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W
#  reached_solution - Whether or not a solution was reached (as opposed to
#      terminating early due to max_iters)
function solve_tr_subproblem!{T}(gr::Vector{T},
                                 H::Matrix{T},
                                 s::Vector{T},
                                 d::Vector{T},
                                 z::Vector{T},
                                 δ::T;
                                 max_iters::Int=20,
                                 σ₁ = 0.1,
                                 σ₂ = 0.0,
                                 debug = false)

    tol0      = sqrt(eps(T))

    # See p. 566 for discussion of values for σ₁ and σ₂
    n     = LinAlg.checksquare(H)
    Hsym  = Symmetric(H)
    F     = Cholesky2{eltype(H)}(H, 'L', 0)
    L     = getL(F)
    δ²    = δ^2
    nrmgr = norm(gr)
    nrmH1 = norm(H, 1)

    ## store diagonal of H in d
    for i in 1:n
        @inbounds d[i] = H[i,i]
    end

    # Initial safeguard values
    λs = -diagmin(H)
    λl = max(0, λs, nrmgr/δ - nrmH1)
    λu = nrmgr/δ + nrmH1
    λ  = zero(λs)

    # Algorithm 3.14 of Moré and Sorensen
    ## 1. Safeguard λ
    for k in 1:max_iters

        λ = max(λ, λl)
        λ = min(λ, λu)
        if λ <= λs
            λ = max(λu/1000, sqrt(λl*λu))
        end

        debug && @show λ
        ## 2. Is positive definite
        # copy!(HλI.data, H)
        ### Set diagonal
        for i in 1:n
            @inbounds H[i,i] = d[i] + λ
        end
        cholfact2!(F, Hsym)
        if isposdef(F)

            ## 3. Solve LLtp = -g
            scale!(s, gr, -one(T))
            A_ldiv_B!(L, s)
            nrmLs  = norm(s)
            Ac_ldiv_B!(L, s)
            nrms  = norm(s)
            nrms² = nrms^2

            debug && @show nrms

            # 4. Done or hard case?
            if nrms < (1 - σ₁)*δ && λ > tol0
                debug && println("HARD CASE")
                # z[1] = 1
                # z = F\z
                # scale!(z, inv(norm(z)))
                # z = eig(Symmetric(L*L'))[2][:,1]
                # nrmLz = norm(L'z)
                nrmLz = svdminvec!(z, L)
                # @show nrmLz
                # @show norm(L'z)
                # nrmLz = norm(L'z)
                sz = dot(s, z)
                τ  = (δ² - nrms²)/(sz + sign(sz)*sqrt(sz^2 + δ² - nrms²))
                debug && @show τ

                ## 5. Update safeguard parameters (hard case)
                λu = min(λu, λ)
                λs = max(λs, λ - nrmLz^2)
                λl = max(λl, λs)
# @show norm(L'*z); @show norm(L'*p); @show τ; @show nrms; @show norm(p + τ*z)
                ## 6. Check convergence
                debug && println("abs2(nrmLz*τ) = $(abs2(nrmLz*τ))")
                debug && println("σ₁*(2 - σ₁)*max(σ₂, nrmLs^2 + λ*δ²) = $(σ₁*(2 - σ₁)*max(σ₂, nrmLs^2 + λ*δ²))")
                if abs2(nrmLz*τ) <= σ₁*(2 - σ₁)*max(σ₂, nrmLs^2 + λ*δ²)
                    debug && println("DONE!")
                    LinAlg.axpy!(τ, z, s)
                    m = dot(gr, s) + (quadform(F, s) - λ*nrms²)/2
                    ### restore H
                    restore!(H, d)
                    # return dot(gr, s) + dot(s, H * s)/2, false
                    return m, false, λ, true
                end
            else
                debug && println("@show PD CASE")
                ## 5. Update safeguard parameters (normal positive definite case)
                λl = max(λl, λ)

                ## 6. Check convergence
                if abs(δ - nrms) < σ₁*δ || (λ <= tol0 && nrms <= δ)
                    debug && println("DONE")
                    debug && println("abs(δ - nrms) = $(abs(δ - nrms))")
                    debug && println("λ = $λ, nrms = $nrms")
                    m = dot(gr, s) + (quadform(F, s) - λ*nrms²)/2
                    restore!(H, d)
                    # return dot(gr, s) + dot(s, H*s)/2, nrms < δ
                    return m, !(abs(δ - nrms) < σ₁*δ), λ, false
                end
            end

            ## 7. Update λ
            copy!(z, s)
            A_ldiv_B!(L, z)
            λ += abs2(nrms/norm(z))*((nrms - δ)/δ)
        else
            debug && println("ID CASE")
            ## 5. Update safeguard parameters (normal indefinite case)
            λl = max(λl, λ)
            fill!(z, 0)
            z[F.info] = 1
            μ = L[F.info,F.info] # Gay's notation
            L[F.info,F.info] = 1
            # z[1:F.info] = LowerTriangular(L[1:F.info,1:F.info])'\z[1:F.info]
            Ac_ldiv_B!(LowerTriangular(view(L.data, 1:F.info, 1:F.info)), view(z, 1:F.info))
            λs = max(λs, λ - μ/dot(z,z))
            λl = max(λl, λs)

            ## 6. Check convergence
            ## Nothing to do since H + λI not PD

            ## 7. Update λ
            λ = λs
        end
    end
    restore!(H, d)
    error("algorithm didn't converge in $max_iters iterations")
end

immutable NewtonTrustRegion{T <: Real} <: Optimizer
    initial_δ::T
    δ_hat::T
    η::T
    ρ_lower::T
    ρ_upper::T
end

NewtonTrustRegion(; initial_delta::Real = 1.0,
                    delta_hat::Real = 100.0,
                    eta::Real = 0.1,
                    rho_lower::Real = 0.25,
                    rho_upper::Real = 0.75) =
                    NewtonTrustRegion(initial_delta, delta_hat, eta, rho_lower, rho_upper)

Base.summary(::NewtonTrustRegion) = "Newton's Method (Trust Region)"

function initial_state{T}(method::NewtonTrustRegion, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    @assert(method.δ_hat > 0, "δ_hat must be strictly positive")
    @assert(0 < method.initial_δ < method.δ_hat, "δ must be in (0, δ_hat)")
    @assert(0 <= method.η < method.ρ_lower, "η must be in [0, ρ_lower)")
    @assert(method.ρ_lower < method.ρ_upper, "must have ρ_lower < ρ_upper")
    @assert(method.ρ_lower >= 0.)
    # Keep track of trust region sizes
    δ = method.initial_δ

    # Record attributes of the subproblem in the trace.
    λ = NaN
    value_gradient!(d, initial_x)
    hessian!(d, initial_x)
    NewtonTrustRegionState(copy(initial_x), # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         T(NaN),
                         similar(initial_x), # Maintain current search direction in state.s
                         similar(initial_x), # buffer of same type and size as stats.s
                         similar(initial_x), # buffer for diagonal
                         T(δ),
                         method.η, # η
                         zero(T)) # ρ
end


function update_state!{T}(d, state::NewtonTrustRegionState{T}, method::NewtonTrustRegion)
    n = length(state.x)

    # hoist
    x          = state.x
    x_previous = state.x_previous
    s          = state.s

    # Find the next step direction.
    m, interior = solve_tr_subproblem!(gradient(d), NLSolversBase.hessian(d),
                                       s, state.d, state.z, state.δ, debug = false)

    # Maintain a record of previous position
    copy!(x_previous, state.x)

    # Update current position
    @simd for i in 1:n
        @inbounds x[i] += s[i]
    end

    # Update the function value and gradient
    state.f_x_previous = value(d)
    value!(d, state.x)


    # Update the trust region size based on the discrepancy between
    # the predicted and actual function values.  (Algorithm 4.1 in N&W)
    f_x_diff = state.f_x_previous - value(d)
    if abs(m) <= eps(T)
        # This should only happen when the step is very small, in which case
        # we should accept the step and assess_convergence().
        state.ρ = 1.0
    elseif m > 0
        # This can happen if the trust region radius is too large and the
        # Hessian is not positive definite.  We should shrink the trust
        # region.
        state.ρ = method.ρ_lower - 1.0
    else
        state.ρ = f_x_diff / (0 - m)
    end

    if state.ρ < method.ρ_lower
        state.δ *= 0.25
    elseif (state.ρ > method.ρ_upper) && !interior
        state.δ = min(2 * state.δ, method.δ_hat)
    else
        # else leave δ unchanged.
    end

    if state.ρ <= state.η
        # The improvement is too small and we won't take it.

        # If you reject an interior solution, make sure that the next
        # δ is smaller than the current step.  Otherwise you waste
        # steps reducing δ by constant factors while each solution
        # will be the same.
        δ = sqrt(zero(T))
        @inbounds @simd for i in eachindex(x)
            δ += abs2(x[i] - x_previous[i])
        end
        state.δ = sqrt(δ)/4

        d.f_x = state.f_x_previous
        copy!(x, x_previous)
        copy!(d.last_x_f, x)
    end

    false
end
