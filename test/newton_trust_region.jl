@testset "Newton Trust Region" begin

model(s, gr, H) = dot(gr, s) + dot(s, H * s)/2

@testset "Subproblems I" begin
    # verify that solve_tr_subproblem! finds the minimum
    n = 2
    gr = [-0.74637,0.52388]
    H = [0.945787 -3.07884; -3.07884 -1.27762]

    s = zeros(n)
    d = similar(s)
    z = similar(s)
    m, interior = Optim.solve_tr_subproblem!(gr, H, s, d, z, 1.0, max_iters=100)

    for j in 1:10
        bad_s = rand(n)
        bad_s ./= norm(bad_s)  # boundary
        @test model(s, gr, H) <= model(bad_s, gr, H) + 1e-8
    end
end

@testset "Subproblems II" for n in 2:10
    # random Hessians--verify that solve_tr_subproblem! finds the minimum
    s = zeros(n)
    d = similar(s)
    z = similar(s)

    for i in 1:1000
        gr = randn(n)
        H  = randn(n, n)
        H += H'
        σ₁ = rand()/10

        m, interior = Optim.solve_tr_subproblem!(gr, H, s, d, z, 1., max_iters=100, σ₁ = σ₁)

        ψstar = model(zeros(n), gr, H)
        @test model(s, gr, H) - ψstar  <= σ₁*(2 - σ₁)*abs(ψstar) # origin

        for j in 1:10
            bad_s = rand(n)
            bad_s ./= norm(bad_s)  # boundary
            ψstar = model(zeros(n), gr, H)
            @test model(s, gr, H) - ψstar  <= σ₁*(2 - σ₁)*abs(ψstar)
            bad_s .*= rand()  # interior
            ψstar = model(zeros(n), gr, H)
            @test model(s, gr, H) - ψstar  <= σ₁*(2 - σ₁)*abs(ψstar)
        end
    end
end

@testset "Test problems" for σ₁ in exp10.((-1, -5, -10))
    #######################################
    # First test the subproblem.
    srand(42)
    n = 5
    H = rand(n, n)
    H = H' * H + 4 * eye(n)
    H_eig = eigfact(H)
    U = H_eig[:vectors]

    gr = zeros(n)
    gr[1] = 1
    s = zeros(Float64, n)
    d = similar(s)
    z = similar(s)

    true_s = -H \ gr
    s_norm2 = dot(true_s, true_s)
    true_m = dot(true_s, gr) + dot(true_s, H * true_s)/2

    # An interior solution
    delta = sqrt(s_norm2) + 1
    m, interior, lambda, hard_case =
        Optim.solve_tr_subproblem!(gr, H, s, d, z, delta, σ₁ = σ₁)
    @test interior
    @test !hard_case
    @test m - true_m <= abs(true_m)*σ₁*(2 - σ₁)
    @test norm(s - true_s) < 1e-12
    @test abs(lambda) < 1e-12

    # A boundary solution
    delta = 0.5 * sqrt(s_norm2)
    m, interior, lambda, hard_case =
        Optim.solve_tr_subproblem!(gr, H, s, d, z, delta, σ₁ = σ₁)
    @test !interior
    @test !hard_case
    @test m > true_m
    @test abs(norm(s) - delta) < σ₁
    @test lambda > 0

    # Now check an actual hard case problem
    L = zeros(Float64, n) + 0.1
    L[1] = -1.
    H = U * diagm(L) * U'
    H = 0.5 * (H' + H)
    @test issymmetric(H)
    gr = U[:,2][:]
    @test abs(dot(gr, U[:,1][:])) < 1e-12
    true_s = -H \ gr
    s_norm2 = dot(true_s, true_s)
    true_m = dot(true_s, gr) + 0.5 * dot(true_s, H * true_s)

    delta = 0.5 * sqrt(s_norm2)
    m, interior, lambda, hard_case =
        Optim.solve_tr_subproblem!(gr, H, s, d, z, delta)
    @test !interior
    @test hard_case
    @test abs(norm(s) - delta) < 1e-12

    #######################################
    # Next, test on actual optimization problems.

    function f(x::Vector)
        (x[1] - 5.0)^4
    end

    function g!(storage::Vector, x::Vector)
        storage[1] = 4.0 * (x[1] - 5.0)^3
    end

    function h!(storage::Matrix, x::Vector)
        storage[1, 1] = 12.0 * (x[1] - 5.0)^2
    end

    d = TwiceDifferentiable(f, g!, h!, [0.0])

    results = Optim.optimize(d, [0.0], NewtonTrustRegion())
    @test length(results.trace) == 0
    @test results.g_converged
    @test norm(Optim.minimizer(results) - [5.0]) < 0.01
    @test summary(results) == "Newton's Method (Trust Region)"

    eta = 0.9

    function f_2(x::Vector)
        0.5 * (x[1]^2 + eta * x[2]^2)
    end

    function g!_2(storage::Vector, x::Vector)
        storage[1] = x[1]
        storage[2] = eta * x[2]
    end

    function h!_2(storage::Matrix, x::Vector)
        storage[1, 1] = 1.0
        storage[1, 2] = 0.0
        storage[2, 1] = 0.0
        storage[2, 2] = eta
    end

    d = TwiceDifferentiable(f_2, g!_2, h!_2, Float64[127, 921])

    results = Optim.optimize(d, Float64[127, 921], NewtonTrustRegion())
    @test results.g_converged
    @test norm(Optim.minimizer(results) - [0.0, 0.0]) < 0.01

    # Test Optim.newton for all twice differentiable functions in
    # Optim.UnconstrainedProblems.examples
    @testset "Optim problems" begin
        run_optim_tests(NewtonTrustRegion())
    end
end


@testset "PR #341" begin
    # verify that no PosDef exception is thrown
    Optim.solve_tr_subproblem!([0, 1.], [-1000 0; 0. -999], ones(2), zeros(2), zeros(2), 1e-2)
end

end
