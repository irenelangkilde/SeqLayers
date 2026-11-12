# Learned product weights (user 2026-11-12): fit the weights of the product score's terms -- log P(attach) and each head
# property's log term -- on held-out attachment decisions dumped by ceiling.jl proddump=<prefix>, as a conditional
# logit over each decision's surviving candidates (Newton; L2 pull toward 1, the plain product).  Prints the prodw
# string for ceiling.jl and the share of decisions whose argmax is gold, plain vs weighted, on the fit data and on an
# optional second dump (e.g. dev).
#
#   julia --project=experiments experiments/fit_prodw.jl fit=<prefix> [eval=<prefix>] [lambda=1.0]

using LinearAlgebra

const NAMES = [:attach, :head_dir, :head_dist, :head_kinleft, :head_kinright, :head_sdist, :head_sibin, :head_sibleft,
  :head_sibout, :head_sibright, :head_trank, :head_upos, :head_wc30, :head_wc300, :head_xpos]   # (ceiling_lib PRODDUMP_NAMES)

args = Dict(split(a, '=', limit = 2)[1] => split(a, '=', limit = 2)[2] for a in ARGS)

"The decisions in <prefix>-w*.tsv: each a (term matrix candidates x NAMES, NaN as 0; index of the gold candidate or 0)."
function load(prefix)
  dir, base = dirname(abspath(prefix)), basename(prefix)
  out = Tuple{Matrix{Float64},Int}[]
  for f in sort(filter(f -> startswith(f, base * "-w") && endswith(f, ".tsv"), readdir(dir)))
    rows = Dict{Int,Vector{Tuple{Bool,Vector{Float64}}}}()
    for line in eachline(joinpath(dir, f))
      p = split(line, '\t')
      push!(get!(rows, parse(Int, p[1]), Tuple{Bool,Vector{Float64}}[]), (p[2] == "1", [isnan(x) ? 0.0 : x for x in parse.(Float64, p[3:end])]))
    end
    for (_, r) in sort(collect(rows); by = first)
      X = reduce(vcat, [permutedims(v) for (_, v) in r])
      push!(out, (X, something(findfirst(first, r), 0)))
    end
  end
  out
end

"Share of decisions (two or more candidates, gold among them) whose top weighted score is the gold candidate."
accuracy(D, w) = (ok = [argmax(X * w) == g for (X, g) in D if g > 0 && size(X, 1) > 1]; (sum(ok), length(ok)))

function fit(D, k; lambda = 1.0, iters = 100)
  used = [(X, g) for (X, g) in D if g > 0 && size(X, 1) > 1]
  lse(s) = (mx = maximum(s); mx + log(sum(exp.(s .- mx))))
  objective(w) = sum(X * w |> s -> s[g] - lse(s) for (X, g) in used) - lambda / 2 * sum(abs2, w .- 1)
  w = ones(k)
  f = objective(w)
  for it in 1:iters
    grad, H = -lambda .* (w .- 1), -lambda * Matrix(1.0I, k, k)
    for (X, g) in used
      s = X * w
      p = exp.(s .- maximum(s)); p ./= sum(p)
      mu = X' * p
      grad .+= X[g, :] .- mu
      H .-= X' * (p .* X) .- mu * mu'
    end
    step = -(H \ grad)
    a = 1.0                                          # backtracking: the step must raise the objective
    while a > 1e-8
      f2 = objective(w .+ a .* step)
      f2 > f && break
      a /= 2
    end
    a <= 1e-8 && break
    w .+= a .* step
    f2 = objective(w)
    abs(f2 - f) < 1e-6 * abs(f) && (f = f2; break)
    f = f2
  end
  w
end

D = load(args["fit"])
k = length(NAMES)
live = [j for j in 1:k if any(X -> any(!iszero, X[:, j]), first.(D))]     # terms that ever enter the product
Dl = [(X[:, live], g) for (X, g) in D]
println("fit data: ", length(D), " decisions, ", count(((X, g),) -> g > 0 && size(X, 1) > 1, D), " with >= 2 candidates and gold among them; terms: ",
  join(NAMES[live], ", "))
w = fit(Dl, length(live); lambda = parse(Float64, get(args, "lambda", "1.0")))
for (n, x) in zip(NAMES[live], w)
  println("  ", rpad(n, 14), round(x; digits = 3))
end
println("prodw=", join(["$(n)=$(round(x; digits = 3))" for (n, x) in zip(NAMES[live], w)], ","))
show_acc(lbl, D) = (a = accuracy(D, ones(length(live))); b = accuracy(D, w);
  println(lbl, ": plain ", a[1], " / ", a[2], " = ", round(a[1] / a[2]; digits = 4), ";  weighted ", b[1], " / ", b[2], " = ", round(b[1] / b[2]; digits = 4),
    "  (", b[1] - a[1] >= 0 ? "+" : "", b[1] - a[1], ")"))
show_acc("fit", Dl)
if haskey(args, "eval")
  E = [(X[:, live], g) for (X, g) in load(args["eval"])]
  show_acc("eval", E)
end
