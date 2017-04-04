using Base.LinAlg: BlasInt, BlasReal, BlasFloat, checksquare, chkstride1
using Base.LinAlg.BLAS: @blasfunc
using Base.LinAlg.LAPACK: chklapackerror

# for (geev, elty) in
#     ((:dgeev_,:Float64),
#      (:sgeev_,:Float32))
#     @eval begin
#         #      SUBROUTINE DGEEV( JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR,
#         #      $                  LDVR, WORK, LWORK, INFO )
#         # *     .. Scalar Arguments ..
#         #       CHARACTER          JOBVL, JOBVR
#         #       INTEGER            INFO, LDA, LDVL, LDVR, LWORK, N
#         # *     .. Array Arguments ..
#         #       DOUBLE PRECISION   A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ),
#         #      $                   WI( * ), WORK( * ), WR( * )
#         function geev!(jobvl::Char, jobvr::Char, A::StridedMatrix{$elty}, VL::StridedMatrix{$elty}, VR::StridedMatrix{$elty}, WR::StridedVector{$elty}, WR::StridedVector{$elty}, work::Vector{$elty}, query::Bool = false)

#             chkstride1(A)
#             n = checksquare(A)
#             chkfinite(A) # balancing routines don't support NaNs and Infs
#             lwork = query ? -1 : length(work) # is it a query for work space?

#             @assert length(WR) == length(WI) == n
#             @assert jobvl == 'N' || (jobvl == 'V' && size(VL, 1) == n && size(VL, 2) == n)
#             @assert jobvr == 'N' || (jobvr == 'V' && size(VR, 1) == n && size(VR, 2) == n)
#             @assert lwork >= query ? -1 : max(1, 3*n)

#             info  = Ref{BlasInt}()

#             ccall((@blasfunc($geev), liblapack), Void,
#                 (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
#                  Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
#                  Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
#                  Ptr{BlasInt}, Ptr{BlasInt}),
#                  &jobvl, &jobvr, &n, A,
#                  &max(1,stride(A,2)), WR, WI, VL,
#                  &n, VR, &n, work,
#                  &lwork, info)
#             end
#             chklapackerror(info[])
#             return WR, WI, VL, VR
#         end

#         function geev!(jobvl::Char, jobvr::Char, A::StridedMatrix{$elty})

#             n = checksquare(A)

#             @assert jobvl == 'N' || jobvl == 'V'
#             @assert jobvr == 'N' || jobvr == 'V'

#             WR = Vector{$elty}(n)
#             WI = Vector{$elty}(n)
#             if jobvl == 'V'
#                 VL = Matrix{$elty}(n,n)
#             else
#                 VL = Matrix{$elty}(0,0)
#             end
#             if jobvr == 'V'
#                 VR = Matrix{$elty}(n,n)
#             else
#                 VR = Matrix{$elty}(0,0)
#             end
#             work = Vector{BlasInt}(1)

#             # Query optimal size
#             geev!(jobvl, jobvr, A, VL, VR, WR, WR, work, true)
#             resize!(work, Int(work[1]))

#             return geev!(jobvl, jobvr, A, VL, VR, WR, WR, work)
#         end
#     end
# end

syev_worksize(::Type{T}, n::Integer) where {T<:BlasFloat} = ccall((@blasfunc(ilaenv_), Base.liblapack_name), Int,
    (Ptr{Int}, Ptr{UInt8}, Ptr{UInt8}, Ptr{Int}, Ptr{Int}, Ptr{Int}, Ptr{Int}),
        &1, (T === Float32 ? "s" : (T === Float64 ? "d" : (T === Complex{Float64} ? "c" : "z")))*"sytrd",
            "V", &n, &(-1), &(-1), &(-1))

for (syev, elty) in
    ((:dsyev_,:Float64),
     (:ssyev_,:Float32))
    @eval begin
        #       SUBROUTINE DSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LWORK, N
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
        function syev!(jobz::Char, uplo::Char, A::StridedMatrix{$elty}, W::Vector{$elty}, work::Vector{$elty}, query::Bool = false)
            chkstride1(A)
            n = checksquare(A)
            lwork = query ? -1 : length(work)

            @assert jobz == 'N' || jobz == 'V'
            @assert uplo == 'U' || uplo == 'L'
            @assert length(W) == n
            @assert query || lwork >= max(1, 3*n - 1)

            info  = Ref{BlasInt}()
            ccall((@blasfunc($syev), Base.liblapack_name), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                 &jobz, &uplo, &n, A, &max(1,stride(A,2)),
                 W, work, &lwork, info)
            chklapackerror(info[])
            return W, A, work
        end

        function syev!(jobz::Char, uplo::Char, A::StridedMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)

            W    = Vector{$elty}(n)
            work = Vector{$elty}(3*n - 1)

            syev!(jobz, uplo, A, W, work, true)
            resize!(work, Int(work[1]))
            return syev!(jobz, uplo, A, W, work)
        end
    end
end

immutable SymmetricEigen{T<:Real}
    values::Vector{T}
    vectors::Matrix{T}
    workbuffer::Vector{T}
end

eigfact2!(A::Symmetric{<:BlasReal,<:StridedMatrix}) = SymmetricEigen(syev!('V', A.uplo, A.data)...)
function eigfact2!(F::SymmetricEigen{T}, A::Symmetric{T}) where T<:BlasReal
    copy!(F.vectors, A.data)
    syev!('V', A.uplo, F.vectors, F.values, F.workbuffer)
    return F
end
eigfact2(A::Symmetric{<:Real,<:StridedMatrix}) = eigfact2!(copy(A))

type NewtonTrustRegionState{T,N,G}
    @add_generic_fields()
    x_previous::Array{T,N}
    g_previous::G
    f_x_previous::T
    s::Array{T,N}
    q_l::Array{T,N}
    H_ridged::Matrix{T}
    qg::Vector{T}
    H_eig::SymmetricEigen{T}
    F::LinAlg.Cholesky{T,Matrix{T}}
    hard_case::Bool
    reached_subproblem_solution::Bool
    interior::Bool
    δ::T
    λ::T
    η::T
    ρ::T
end

#
# Check whether we are in the "hard case".
#
# Args:
#  H_eigv: The eigenvalues of H, low to high
#  qg: The inner product of the eigenvalues and the gradient in the same order
#
# Returns:
#  hard_case: Whether it is a candidate for the hard case
#  λ_1_multiplicity: The number of times the lowest eigenvalue is repeated,
#                         which is only correct if hard_case is true.
function check_hard_case_candidate(H_eigv, qg)
    @assert length(H_eigv) == length(qg)
    if H_eigv[1] >= 0
        # The hard case is only when the smallest eigenvalue is negative.
        return false, 1
    end
    hard_case = true
    λ_index = 1
    hard_case_check_done = false
    while !hard_case_check_done
        if λ_index > length(H_eigv)
            hard_case_check_done = true
        elseif abs(H_eigv[1] - H_eigv[λ_index]) > 1e-10
            # The eigenvalues are reported in order.
            hard_case_check_done = true
        else
            if abs(qg[λ_index]) > 1e-10
                hard_case_check_done = true
                hard_case = false
            end
            λ_index += 1
        end
    end

    hard_case, λ_index - 1
end

# Function 4.39 in N&W
function p_sq_norm{T}(λ::T, min_i, n, qg, H_eig)
    p_sum = zero(T)
    for i = min_i:n
        p_sum += qg[i]^2 / (λ + H_eig.values[i])^2
    end
    p_sum
end

function Base.LinAlg.cholfact!(C::LinAlg.Cholesky{T,S}, A::Hermitian{T,S}) where {T,S}
    if C.uplo != A.uplo
        throw(ArgumentError("uplo fields are not the same"))
    end
    copy!(C.factors, A.data)
    cholfact!(Hermitian(C.factors, Symbol(A.uplo)))
    return C
end
function getR(C::LinAlg.Cholesky)
    if C.uplo == :L
        throw(ArgumentError("only possible to extract R factor when factorization is stored in upper triangle"))
    end
    return UpperTriangular(C.factors)
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
                                 state::NewtonTrustRegionState;
                                 tolerance::T=1e-10,
                                 max_iters::Int=5)

    δ = state.δ
    s = state.s

    n = length(gr)
    δ_sq = δ^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    # H_eig = eigfact(Symmetric(H))
    H_eig = state.H_eig
    eigfact2!(H_eig, Symmetric(H))
    min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
    H_ridged           = state.H_ridged
    copy!(H_ridged, H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = Ac_mul_B!(state.qg, H_eig.vectors, gr)

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior         = true
    hard_case        = false
    reached_solution = true

    if min_H_ev >= 1e-8 && p_sq_norm(zero(T), 1, n, qg, H_eig) <= δ_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        BLAS.gemv!('N', -one(T), H_eig.vectors, qg ./ H_eig.values, zero(T), s)
        λ = zero(T)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        hard_case_candidate, min_H_ev_multiplicity =
            check_hard_case_candidate(H_eig.values, qg)

        # Solutions smaller than this lower bound on λ are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        λ_lb = -min_H_ev + max(1e-8, 1e-8 * (max_H_ev - min_H_ev))
        λ = λ_lb

        hard_case = false
        if hard_case_candidate
            # The "hard case". λ is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W
            p_λ2 = p_sq_norm(λ, min_H_ev_multiplicity + 1, n, qg, H_eig)
            if p_λ2 > δ_sq
                # Then we can simply solve using root finding.
                # Set a starting point greater than the minimum based on the
                # range between the largest and smallest eigenvalues.
                λ = λ_lb + 0.01 * (max_H_ev - min_H_ev)
            else
                hard_case = true
                reached_solution = true

                tau = sqrt(δ_sq - p_λ2)

                # I don't think it matters which eigenvector we pick so take
                # the first.
                for i=1:n
                    s[i] = tau * H_eig.vectors[i, 1]
                    for k = (min_H_ev_multiplicity + 1):n
                        s[i] = s[i] +
                               qg[k] * H_eig.vectors[i, k] / (H_eig.values[k] + λ)
                    end
                end
            end
        end

        if !hard_case
            # Algorithim 4.3 of N&W, with s insted of p_l for consistency with
            # Optim.jl

            for i in 1:n
                H_ridged[i, i] = H[i, i] + λ
            end

            reached_solution = false
            q_l              = state.q_l
            H_ridgedHerm     = Hermitian(H_ridged, :U)
            F                = state.F
            for iter in 1:max_iters
                λ_previous = λ

                cholfact!(F, H_ridgedHerm)
                R = getR(F)
                @inbounds for i in eachindex(gr)
                    s[i] = -gr[i]
                end
                A_ldiv_B!(R, Ac_ldiv_B!(R, s))
                copy!(q_l, s)
                Ac_ldiv_B!(R, q_l)
                norm2_s = dot(s, s)
                λ_update = norm2_s * (sqrt(norm2_s) - δ) / (δ * dot(q_l, q_l))
                λ += λ_update

                # Check that λ is not less than λ_lb, and if so, go
                # half the way to λ_lb.
                if λ < (λ_lb + 1e-8)
                    λ = (λ_previous - λ_lb)/2 + λ_lb
                end

                for i in 1:n
                    H_ridged[i, i] = H[i, i] + λ
                end

                if abs(λ - λ_previous) < tolerance
                    reached_solution = true
                    break
                end
            end
        end
    end

    m = dot(gr, s) + dot(s, H * s)/2

    return m, interior, λ, hard_case, reached_solution
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


function initial_state{T}(method::NewtonTrustRegion, options, d, initial_x::Array{T})
    n = length(initial_x)
    # Maintain current gradient in gr
    @assert(method.δ_hat > 0, "δ_hat must be strictly positive")
    @assert(0 < method.initial_δ < method.δ_hat, "δ must be in (0, δ_hat)")
    @assert(0 <= method.η < method.ρ_lower, "η must be in [0, ρ_lower)")
    @assert(method.ρ_lower < method.ρ_upper, "must have ρ_lower < ρ_upper")
    @assert(method.ρ_lower >= 0.)
    # Keep track of trust region sizes
    δ = copy(method.initial_δ)

    # Record attributes of the subproblem in the trace.
    hard_case = false
    reached_subproblem_solution = true
    interior = true
    λ = NaN
    value_gradient!(d, initial_x)
    hessian!(d, initial_x)
    NewtonTrustRegionState("Newton's Method (Trust Region)", # Store string with model name in state.method
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         similar(gradient(d)), # Store previous gradient in state.g_previous
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         similar(initial_x), # buffer of same type and size as stats.s
                         Matrix{T}(n,n),     # buffer of H_ridged
                         similar(initial_x), # buffer qg
                         SymmetricEigen{T}(Vector{T}(n), Matrix{T}(n,n), Vector{BlasInt}((syev_worksize(T, n) + 2)*n)), # worksize only correct for syev solver
                         LinAlg.Cholesky{T,Matrix{T}}(Matrix{T}(n,n), 'U'), # buffer for Cholesky
                         hard_case,
                         reached_subproblem_solution,
                         interior,
                         T(δ),
                         λ,
                         method.η, # η
                         zero(T)) # ρ
end


function update_state!{T}(d, state::NewtonTrustRegionState{T}, method::NewtonTrustRegion)


    # Find the next step direction.
    m, state.interior, state.λ, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(gradient(d), NLSolversBase.hessian(d), state)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    @simd for i in 1:state.n
        @inbounds state.x[i] = state.x[i] + state.s[i]
    end

    # Update the function value and gradient
    copy!(state.g_previous, gradient(d))
    state.f_x_previous = value(d)
    value_gradient!(d, state.x)


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
    elseif (state.ρ > method.ρ_upper) && (!state.interior)
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
        # δ = sqrt(mapreduce(t -> abs2(t[1] - t[2]), +, zip(state.x, state.x_previous)))/4
        x, x_previous = state.x, state.x_previous
        δ = sqrt(zero(T))
        @inbounds @simd for i in eachindex(x)
            δ += abs2(x[i] - x_previous[i])
        end
        state.δ = sqrt(δ)/4
        # x_diff = state.x - state.x_previous
        # δ = 0.25 * sqrt(vecdot(x_diff, x_diff))

        d.f_x = state.f_x_previous
        copy!(state.x, state.x_previous)
        copy!(gradient(d), state.g_previous)
    end

    false
end
