p = "src/chain.jl"; s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old)); s = s.replace(old, new)

# --- option
rep("  tuplemin::Float64                               # > 0: the tuple pool",
    "  interleave::Bool                                # ctxmode :accum: chain steps and leftovers in one maxdev loop (see accum_leftovers)\n"
    "  tuplemin::Float64                               # > 0: the tuple pool")
rep("wselcomb = :sum, wselmove = :weights, wselweight = false, tuplek = 0, tuplemin = 0.0)",
    "wselcomb = :sum, wselmove = :weights, wselweight = false, tuplek = 0, tuplemin = 0.0, interleave = false)")
rep("wselcomb, wselmove, wselweight, tuplek, tuplemin, posnegonly,", "wselcomb, wselmove, wselweight, tuplek, interleave, tuplemin, posnegonly,")
rep('''  wselmove in (:weights, :wcorr) || error''', '''  interleave && (leftovers != :greedy || ctxmode != :accum || select != :maxdev || pairsteps || pickfewest > 0 || target || lockstep != :none || sample > 0 || dualchain != :off) &&
    error("interleave: needs select = :maxdev, leftovers = :greedy, ctxmode = :accum, and no pairsteps, pickfewest, target, lockstep, sample or dualchain")
  interleave && !(stepscore in (:abs, :posneg)) && error("interleave: stepscore :abs or :posneg")
  wselmove in (:weights, :wcorr) || error''')

# --- step_single!: peek mode
rep('''function step_single!(m::ChainModel, st::ChainState, Ic, label, remaining; target = nothing, exhaust = false,
  cap = EXACT, nosingle = NOSINGLE)''', '''function step_single!(m::ChainModel, st::ChainState, Ic, label, remaining; target = nothing, exhaust = false,
  cap = EXACT, nosingle = NOSINGLE, peek = false)''')
rep('''  for (s, e) in zip(slots, extend_all(m, st, depth(st), present_items(Ic, slots); cap))
    y, ok = update_odds(m, e, label, x; pn)
    tr === nothing || push!(tr, (s, ok ? abs(log(y) - log(x)) : NaN, total(e[2])))
    ok || continue''', '''  unsup = Int[]                                     # (peek) the present slots the prefix doesn't support with
  for (s, e) in zip(slots, extend_all(m, st, depth(st), present_items(Ic, slots); cap))
    y, ok = update_odds(m, e, label, x; pn)
    tr === nothing || peek || push!(tr, (s, ok ? abs(log(y) - log(x)) : NaN, total(e[2])))
    ok || (push!(unsup, s); continue)''')
rep('''  pair = nothing                                    # (slot a, slot b, pair entry, odds) if a pair wins
  if usepairs''', '''  # peek (interleave): the best step and, for posneg, the fallback step lowering the odds most, without taking either
  peek && return (x = x, best = best, bs = bs, by = by, be = be, nbest = negfall ? nbest : -Inf, nbs = nbs, nby = nby, nbe = nbe,
    unsup = unsup, alts = alts)
  pair = nothing                                    # (slot a, slot b, pair entry, odds) if a pair wins
  if usepairs''')

# --- run_single: interleaved chains step inside the leftovers
rep('''    nosingle = m.opts.pairsteps && m.opts.pairsfirst ? pair_covered(m, Ic, remaining) : NOSINGLE
    while step_single!(m, st, Ic, label, remaining; nosingle) end''', '''    nosingle = m.opts.pairsteps && m.opts.pairsfirst ? pair_covered(m, Ic, remaining) : NOSINGLE
    interleaves(m, tk) || while step_single!(m, st, Ic, label, remaining; nosingle) end   # (interleave: in chain_logodds)''')
rep('''"Per-candidate chain (:maxdev): each candidate picks its own features."''', '''"The leftover context mode for decision type tk (ctxmode, unless ctxattach / ctxtypes keep it greedy)."
leftmode(m::ChainModel, tk) = (m.opts.ctxattach && tk != typekey(:token, 1)) || (!isempty(m.ctxtypes) && !(tk in m.ctxtypes)) ? :greedy : m.opts.ctxmode
"Does decision type tk build its chain interleaved with the leftovers (interleave, :accum contexts)?"
interleaves(m::ChainModel, tk) = m.opts.interleave && tk !== nothing && leftmode(m, tk) == :accum

"Per-candidate chain (:maxdev): each candidate picks its own features."''')

# --- chain_logodds: with interleave the chain grows during the leftovers, so read its odds after them
rep('''  if m.opts.leftovers in (:none, :greedy)
    corr = m.opts.leftovers == :none ? 0.0 : greedy_leftovers(m, st, Ic, label, prior, remaining)''', '''  if m.opts.leftovers in (:none, :greedy)
    corr = m.opts.leftovers == :none ? 0.0 : greedy_leftovers(m, st, Ic, label, prior, remaining)
    m.opts.interleave && (lo = log(st.odds[end]))     # (interleave: the chain grew with the leftovers)''')
# the noleft filter must not hide chain slots: interleave passes the full list plus the slots kept out
rep('''  if !isempty(m.noleft) && !isempty(st.views[1].base)        # slots kept out of the leftovers
    skip = get(m.noleft, m.recs[st.views[1].base[1]].typekey, nothing)
    skip === nothing || (remaining = [s for s in remaining if !(s in skip)])
  end''', '''  if !isempty(m.noleft) && !isempty(st.views[1].base) && !m.opts.interleave   # slots kept out of the leftovers
    skip = get(m.noleft, m.recs[st.views[1].base[1]].typekey, nothing)
    skip === nothing || (remaining = [s for s in remaining if !(s in skip)])
  end''')

# --- _greedy_leftovers: interleave skips the end-of-chain filtering (done when the chain stops)
rep('''  nred = 0
  poolw = Dict{UInt64,Float64}()                     # poolsigma: pool feature -> weight of conditioning on it
  if m.opts.redundant                                # redundant features''', '''  nred = 0
  poolw = Dict{UInt64,Float64}()                     # poolsigma: pool feature -> weight of conditioning on it
  tk0 = isempty(base) ? nothing : m.recs[base[1]].typekey
  if interleaves(m, tk0)                             # the chain hasn't run yet: it runs with the leftovers
    G0 = get(m.generalize, tk0, Dict{Int,Vector{Int}}())
    skip = isempty(m.noleft) ? Set{Int}() : Set{Int}(get(m.noleft, tk0, Int[]))
    return accum_leftovers(m, st, Ic, G0, Int[], label, prior, base; chainrem = remaining, skip)
  end
  if m.opts.redundant                                # redundant features''')
rep('''  mode = (m.opts.ctxattach && tk != typekey(:token, 1)) || (!isempty(m.ctxtypes) && !(tk in m.ctxtypes)) ? :greedy : m.opts.ctxmode
  G = mode == :greedy''', '''  mode = leftmode(m, tk)
  G = mode == :greedy''')

# --- accum_leftovers: the interleaved loop
rep('''function accum_leftovers(m::ChainModel, st::ChainState, Ic, G, rel::Vector{Int}, label, prior, base)''',
'''function accum_leftovers(m::ChainModel, st::ChainState, Ic, G, rel::Vector{Int}, label, prior, base;
  chainrem = nothing, skip = Set{Int}())''')
rep('''  taken = Set{UInt64}()
  # the anchor X is conditioned on''', '''  taken = Set{UInt64}()
  # interleave (chainrem: the chain's remaining slots; user's design 2026-11-01): the chain grows inside this loop.  Each
  # round the best chain step (as step_single! would take it) competes with the best leftover -- the slots the current
  # prefix doesn't support, past the relevance gate -- scored by its (discounted) correction: by size (stepscore :abs),
  # or (:posneg) raises first, chain step vs positive correction, then, if neither reaches tau, the largest lowerings.
  # A chain winner joins the prefix and the context (its own ladder, rung 1) and moves T by its step; a leftover is
  # taken as usual.  When nothing reaches tau the chain stops: the end-of-chain filtering (redundant, weak) runs and
  # the remaining leftovers are taken as usual.
  chainon = chainrem !== nothing
  ss = STEPSCORE_OVERRIDE[] === nothing ? m.opts.stepscore : STEPSCORE_OVERRIDE[]
  pk = nothing                                       # the chain's peeked step (stale once the chain moves)
  function relevant(sx)
    m.opts.relevance <= 0 || leftover_relevance(m, st, Ic[sx], label, prior, base) >= m.opts.relevance
  end
  # the unsupported slots join the leftovers while the chain runs (relevance against the chain so far)
  function refresh!()
    pk === nothing || return
    pk = step_single!(m, st, Ic, label, chainrem; peek = true)
    for sx in pk.unsup
      (sx in pending || sx in skip || Ic[sx] in taken || Ic[sx] in items(rung)) && continue
      relevant(sx) && push!(pending, sx)
    end
  end
  # the chain stops: its remaining supported slots go through the usual end-of-chain filtering; the rest are leftovers
  function finish!()
    chainon = false
    gone = Int[]
    if m.opts.redundant
      weak = Int[]
      red = first.(redundant_divergence(m, st, Ic, label, chainrem; weak))
      gone = m.opts.poolmarginal == :drop ? union(red, weak) : red
    end
    for sx in chainrem
      (Ic[sx] == ABSENT || sx in gone || sx in pending || sx in skip || Ic[sx] in taken || Ic[sx] in items(rung)) && continue
      relevant(sx) && push!(pending, sx)
    end
  end
  function chainstep!(neg::Bool)
    s_, y_, e_ = neg ? (pk.nbs, pk.nby, pk.nbe) : (pk.bs, pk.by, pk.be)
    advance!(m, st, Ic[s_], e_, y_)
    m.opts.poolsigma > 0 && m.opts.poolmode == :local && push!(st.alts, pk.alts === nothing ? Dict{UInt64,Float64}() : pk.alts)
    deleteat!(chainrem, findfirst(==(s_), chainrem))
    T += log(y_) - log(pk.x)
    (li, ls) = ladder(Ic[s_], s_)
    push!(ladders, li); push!(lslots, ls); push!(rung, 1)
    filter!(sx -> !(Ic[sx] in items(rung)), pending)
    (d = DC_STATS[]) === nothing || (d[:il_chain] = get(d, :il_chain, 0.0) + 1)
    pk = nothing
  end
  # the anchor X is conditioned on''')
rep('''  while !isempty(pending)
    # the generalization path from the current rungs (the same for every X), built as far as needed''',
'''  while true
    chainon && refresh!()
    isempty(pending) && !chainon && break
    # the generalization path from the current rungs (the same for every X), built as far as needed''')
rep('''      (abs(xc) > abs(bestc) || bestX == 0) && ((bestX, bestc, bestk, bestr) = (pi_, xc, xk, xr))
    end''', '''      (abs(xc) > abs(bestc) || bestX == 0) && ((bestX, bestc, bestk, bestr) = (pi_, xc, xk, xr))
      chainon && push!(lcands, (pi_, xc, xk, xr))
    end''')
rep('''    bestX, bestc, bestk, bestr = 0, 0.0, 0, Int[]
''', '''    bestX, bestc, bestk, bestr = 0, 0.0, 0, Int[]
    lcands = Tuple{Int,Float64,Int,Vector{Int}}[]     # (interleave) every scored leftover this round
''')
rep('''    if bestX == 0
      isempty(explained) || (poolexplained!(); continue)''', '''    if chainon                                        # interleave: the chain's step against the best leftover
      tau = m.opts.tau
      cs = pk.bs == 0 ? -Inf : pk.best
      if ss == :posneg
        lp = isempty(lcands) ? nothing : lcands[argmax([c[2] for c in lcands])]
        lsc = lp === nothing ? -Inf : lp[2]
        if max(cs, lsc) < tau                         # nothing raises the odds by tau: the largest lowering
          ln = isempty(lcands) ? nothing : lcands[argmin([c[2] for c in lcands])]
          ncs, nlsc = pk.nbs == 0 ? -Inf : pk.nbest, ln === nothing ? -Inf : -ln[2]
          if max(ncs, nlsc) < tau
            finish!(); continue
          end
          ncs >= nlsc ? (chainstep!(true); continue) : ((bestX, bestc, bestk, bestr) = ln)
        else
          cs >= lsc ? (chainstep!(false); continue) : ((bestX, bestc, bestk, bestr) = lp)
        end
      else
        lsc = bestX == 0 ? -Inf : abs(bestc)
        max(cs, lsc) < tau && (finish!(); continue)
        cs >= lsc && (chainstep!(false); continue)
      end
      (d = DC_STATS[]) === nothing || (d[:il_left] = get(d, :il_left, 0.0) + 1)
    end
    if bestX == 0
      isempty(explained) || (poolexplained!(); continue)''')
rep('''    deleteat!(pending, bestX)
    filter!(s -> s != sx, explained)''', '''    deleteat!(pending, bestX)
    chainrem === nothing || filter!(!=(sx), chainrem)
    filter!(s -> s != sx, explained)''')
open(p, "w").write(s); print("ok")
