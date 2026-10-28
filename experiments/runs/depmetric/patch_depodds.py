# accumby=depodds: the dependence walk's move cost |corr(E|∅)| * (dependence lost) + |log o(context') - T|
# (both in log odds).  Applied after runs/depmetric finished (the sweep reads src/ per run).
p='src/chain.jl'
s=open(p).read()
def rep(old,new):
    global s
    assert s.count(old)==1, old[:90]
    s=s.replace(old,new)
rep("""      xpath = m.opts.accumby == :dep ? Any[path[1]] : path
      xstep! = m.opts.accumby == :dep ? (p -> depstep!(p, X)) : step!""",
"""      xpath = m.opts.accumby == :odds ? path : Any[path[1]]
      imp = m.opts.accumby == :depodds ? margimpact(X) : 0.0
      xstep! = m.opts.accumby == :odds ? step! : (p -> depstep!(p, X, imp))""")
rep("""    # accumby :dep: the move (feature i one rung up, or dropped) giving up the least of X's dependence on the context,
    # ρ(X, item now) - ρ(X, item after); ties by the running-estimate rule
    function depstep!(p, X)""","""    # accumby :dep: the move (feature i one rung up, or dropped) giving up the least of X's dependence on the context,
    # ρ(X, item now) - ρ(X, item after); ties by the running-estimate rule.  :depodds: the move with the smallest
    # imp * (dependence lost) + |log o(context after) - T|, imp = |X's marginal correction| (both in log odds)
    function depstep!(p, X, imp = 0.0)""")
rep("""        sc = (loss, abs(log(c[1].odds[end]) - T))""","""        d = abs(log(c[1].odds[end]) - T)
        sc = m.opts.accumby == :depodds ? (imp * loss + d, 0.0) : (loss, d)""")
rep("""    bestX, bestc, bestk, bestr = 0, 0.0, 0, Int[]""","""    bestX, bestc, bestk, bestr = 0, 0.0, 0, Int[]
    # accumby :depodds: X's marginal correction |log o(x | X) - log o(x)| (0 if unsupported)
    function margimpact(X)
      eX = only(extend_all(m, ChainState(prior, base), 0, [X]))
      y, ok = update_odds(m, eX, label, prior)
      ok ? abs(log(y) - log(prior)) : 0.0
    end""")
rep("""  accumby in (:odds, :dep) || error("unknown accumby $accumby (use :odds or :dep)")""",
"""  accumby in (:odds, :dep, :depodds) || error("unknown accumby $accumby (use :odds, :dep or :depodds)")""")
rep("""  accumby::Symbol                                 # accum walk step: :odds (closest to the running estimate, shared) | :dep (per leftover: least dependence lost)""",
"""  accumby::Symbol                                 # accum walk step: :odds (closest to the running estimate, shared) | :dep (per leftover: least dependence lost) | :depodds (both, in log odds)""")
open(p,'w').write(s)
p='experiments/dev_eval.jl'
s=open(p).read()
rep("""#                the least of its dependence ρ = |Yule's Q| on the context's features)""","""#                the least of its dependence ρ = |Yule's Q| on the context's features) | depodds (each leftover
#                its own path: the move with the smallest |marginal correction| * dependence lost + |log odds
#                change from the running estimate|)""")
open(p,'w').write(s)
