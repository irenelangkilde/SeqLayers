# domains.jl  --  domain variables for a token's attachment and head properties (prototype)
#
# Instead of committing a head property only when it is nearly certain (and otherwise leaving it
# entirely open), each of the token's attachment and head-property decisions keeps a domain: the
# values whose probability is at least `eps`.  The selection constraint head_X(t) = X(attach(t))
# then propagates between them to a fixpoint: an attachment candidate survives only if each of its
# head-property values is still in that property's domain (values not known yet don't prune), and
# a property keeps only values some surviving candidate has.  If propagation empties the
# attachment domain, nothing is pruned.
#
# The attachment is then chosen over its surviving domain, either by its own probabilities alone
# (`:prune`: the pruning is the only change) or by the product of its probability and each head
# property's probability (renormalized over the property's surviving domain) of the value the
# candidate implies (`:product`: log odds over the whole set of options, before any is committed).

"""
Narrow the attachment candidates `cands` (probabilities `pa`) and head properties `props`
(name => (values, probabilities)) for token t to their consistent domains (see the top of the
file); returns the surviving candidates' indices and, per property, the surviving values.
"""
function propagate_domains(st, t, cands, pa, props; eps = 1e-3)
  hv = Dict(name => [head_value(st, t, h, name) for h in cands] for name in keys(props))
  keepvals(vals, ps) = (k = Set(v for (v, p) in zip(vals, ps) if p >= eps); isempty(k) ? Set([vals[argmax(ps)]]) : k)
  D = Dict(name => keepvals(vals, ps) for (name, (vals, ps)) in props)
  alive = [j for j in eachindex(cands) if pa[j] >= eps]
  isempty(alive) && (alive = [argmax(pa)])
  while true
    alive2 = [j for j in alive if all(name -> hv[name][j] == PENDING || hv[name][j] in D[name], keys(props))]
    isempty(alive2) && return collect(eachindex(cands)), Dict(name => Set(vals) for (name, (vals, _)) in props)
    D2 = Dict(name => (any(j -> hv[name][j] == PENDING, alive2) ? D[name] : intersect(D[name], Set(hv[name][j] for j in alive2)))
      for name in keys(props))
    (alive2 == alive && D2 == D) && return alive, D
    alive, D = alive2, D2
  end
end

"""
The attachment of t chosen over its domain: `mode` :prune (its own probabilities) or :product
(times each head property's renormalized probability of the candidate's value).  Returns
(choice, number of surviving candidates).
"""
function domain_choice(st, t, cands, pa, props; eps = 1e-3, mode = :product)
  alive, D = propagate_domains(st, t, cands, pa, props; eps)
  mode == :prune && return cands[alive[argmax(pa[alive])]], length(alive)
  score = Float64[]
  for j in alive
    s = log(pa[j])
    for (name, (vals, ps)) in props
      v = head_value(st, t, cands[j], name)
      v == PENDING && continue
      z = sum(p for (u, p) in zip(vals, ps) if u in D[name]; init = 0.0)
      k = findfirst(==(v), vals)
      s += k === nothing || z <= 0 ? log(1e-12) : log(max(ps[k], 1e-12) / z)
    end
    push!(score, s)
  end
  cands[alive[argmax(score)]], length(alive)
end
