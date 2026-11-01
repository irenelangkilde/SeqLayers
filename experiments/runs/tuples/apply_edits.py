import re, sys
p = "src/chain.jl"; s = open(p).read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)
rep("  wselweight::Bool                                # accumby=:wsel: the leftover's correction times w of its chosen context (min, or sum / 2)\n",
    "  wselweight::Bool                                # accumby=:wsel: the leftover's correction times w of its chosen context (min, or sum / 2)\n"
    "  tuplek::Int                                     # > 0: a :maxdev chain's first step adds the best tuple (triple, else pair, else single) of its tuplek largest single steps (see tuple_step!)\n"
    "  tuplemin::Float64                               # > 0: the tuple pool is the features whose single step reaches tuplemin (and tau; with tuplek too: the tuplek largest of them)\n")
rep("wselcomb = :sum, wselmove = :weights, wselweight = false)", "wselcomb = :sum, wselmove = :weights, wselweight = false, tuplek = 0, tuplemin = 0.0)")
rep("leftwcomb, wselcomb, wselmove, wselweight, posnegonly,", "leftwcomb, wselcomb, wselmove, wselweight, tuplek, tuplemin, posnegonly,")
old = """  order = m.opts.poolm > 0 && tk !== nothing ? get(m.pool, tk, nothing) : nothing
  if order === nothing
    nosingle"""
new = """  order = m.opts.poolm > 0 && tk !== nothing ? get(m.pool, tk, nothing) : nothing
  (m.opts.tuplek > 0 || m.opts.tuplemin > 0) && isempty(seed) && tuple_step!(m, st, Ic, label, remaining, tk)
  if order === nothing
    nosingle"""
rep(old, new)
old = """\"\"\"
:maxdev chains in rounds (one step per still-running candidate per round)"""
new = '''"""
Tuple first step (tuplek / tuplemin; user's design 2026-10-31): the pool is the features whose single step
|Δ log odds| reaches tau (and tuplemin), largest first, cut to the tuplek largest; of its supported triples (else
pairs, else singles) the one with the largest max_x l(X = x | tuple) -- each value's log odds after inserting the
members in pool order, as the chain would (unlabeled decisions: |l|, the larger of attached / not) -- joins the chain,
member by member.  Returns whether a tuple was added.
"""
function tuple_step!(m::ChainModel, st::ChainState, Ic, label, remaining, tk)
  depth(st) == 0 || return false
  x = st.odds[end]
  slots = [s for s in remaining if Ic[s] != ABSENT]
  E1 = extend_all(m, st, 0, present_items(Ic, slots))
  pool = Tuple{Float64,Int,Any,Float64}[]           # (single step, slot, entry, own odds)
  for (s, e) in zip(slots, E1)
    y, ok = update_odds(m, e, label, x)
    ok || continue
    d = abs(log(y) - log(x))
    d >= max(m.opts.tau, m.opts.tuplemin) && push!(pool, (d, s, e, y))
  end
  sort!(pool; by = p -> -p[1])
  m.opts.tuplek > 0 && length(pool) > m.opts.tuplek && resize!(pool, m.opts.tuplek)
  isempty(pool) && return false
  # each value's prior: the chain's own odds for the candidate's label, the type's prior for the others
  prior(l) = l == label || tk === nothing ? x : priorodds(m, lpriorkey(tk, l))
  # max_x l(X = x | members), inserting the members' entries in order, each smoothed toward the last
  function tscore(es)
    ls = label == 0 ? (UInt32(0),) : keys(es[end][2])
    best = -Inf
    for l in ls
      o = prior(l)
      for e in es
        o = first(update_odds(m, e, l, o))
      end
      best = max(best, label == 0 ? abs(log(o)) : log(o))
    end
    best
  end
  branch(s) = ChainState(copy(s.keys), copy(s.views), copy(s.sizes), copy(s.members), copy(s.odds),
    Dict{UInt64,Float64}[], copy(s.cnt))
  n = length(pool)
  best, bi = -Inf, Int[]                            # bi: pool indices of the winner, its entries in pairs / triples
  bes = Any[]
  ytrip = Dict{Tuple{Int,Int},Tuple{Any,Float64,ChainState}}()   # supported pairs: (entry, own odds, state)
  for i in 1:n-1
    sa = branch(st)
    advance!(m, sa, Ic[pool[i][2]], pool[i][3], pool[i][4])
    js = collect(i+1:n)
    for (j, e) in zip(js, extend_all(m, sa, 1, UInt64[Ic[pool[j][2]] for j in js]))
      y, ok = update_odds(m, e, label, pool[i][4])
      ok || continue
      sab = branch(sa)
      advance!(m, sab, Ic[pool[j][2]], e, y)
      ytrip[(i, j)] = (e, y, sab)
    end
  end
  for ((i, j), (eab, yab, sab)) in sort!(collect(ytrip); by = first), size in (3,)
    ls = collect(j+1:n)
    isempty(ls) && continue
    for (l, e) in zip(ls, extend_all(m, sab, 2, UInt64[Ic[pool[l][2]] for l in ls]))
      _, ok = update_odds(m, e, label, yab)
      ok || continue
      sc = tscore(Any[pool[i][3], eab, e])
      sc > best && ((best, bi, bes) = (sc, [i, j, l], Any[pool[i][3], eab, e]))
    end
  end
  if isempty(bi)                                    # no supported triple: the best pair
    for ((i, j), (eab, _, _)) in sort!(collect(ytrip); by = first)
      sc = tscore(Any[pool[i][3], eab])
      sc > best && ((best, bi, bes) = (sc, [i, j], Any[pool[i][3], eab]))
    end
  end
  if isempty(bi)                                    # no supported pair: the best single
    for i in 1:n
      sc = tscore(Any[pool[i][3]])
      sc > best && ((best, bi, bes) = (sc, [i], Any[pool[i][3]]))
    end
  end
  rec = m.opts.poolsigma > 0 && m.opts.poolmode == :local
  o = x
  for (i, e) in zip(bi, bes)
    o = first(update_odds(m, e, label, o))
    advance!(m, st, Ic[pool[i][2]], e, o)
    rec && push!(st.alts, Dict{UInt64,Float64}())
    deleteat!(remaining, findfirst(==(pool[i][2]), remaining))
  end
  (d = DC_STATS[]) === nothing || (d[Symbol(:tuple, length(bi))] = get(d, Symbol(:tuple, length(bi)), 0.0) + 1)
  true
end

"""
:maxdev chains in rounds (one step per still-running candidate per round)'''
rep(old, new)
open(p, "w").write(s)

p = "experiments/dev_eval.jl"; s = open(p).read()
rep('#   dualchain    two chains per candidate',
    '#   tuplek       > 0: each :maxdev chain\'s first step adds the best tuple of its tuplek largest single steps (those\n'
    '#                reaching tau, and tuplemin if > 0): a supported triple, else pair, else single, by max_x l(X = x | tuple)\n'
    '#   tuplemin     > 0: the tuple pool is every feature whose single step reaches tuplemin (with tuplek: its tuplek largest)\n'
    '#   dualchain    two chains per candidate')
rep('"wselweight" => "false",', '"wselweight" => "false", "tuplek" => 0, "tuplemin" => 0.0,')
rep('''        wselmove = Symbol(cfg["wselmove"]), wselweight = cfg["wselweight"] == "true")))
''', '''        wselmove = Symbol(cfg["wselmove"]), wselweight = cfg["wselweight"] == "true")))
      (cfg["tuplek"] > 0 || cfg["tuplemin"] > 0) && (chainopts = merge(chainopts, (tuplek = cfg["tuplek"], tuplemin = cfg["tuplemin"])))
''')
open(p, "w").write(s)
print("ok")
