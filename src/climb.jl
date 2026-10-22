# climb.jl  --  climbing rings for word-form slots (see Climbing in chain.jl), from word rings
# built outside the package (juliaNLP/clusters/rings.jl)

"""
Climbing table for the attachment decision's form slots, from word rings: `targets` (the
clustered words) and `R` (rings.jl's `load_rings` output: order, inner, outer, tree.nodes).  A
word climbs through itself, its nearest class-mates (the `R.inner` sizes), its whole class, then
its outer rings.  Every clustered word is interned in `lex` first, so a word unseen in training
still gets rings.  `families`: :cand (the candidate head's word), :dep (the dependent's word).
"""
function climb_table(targets, R, lex::Lexicon; families = (:cand,))
  codes = UInt32[intern!(lex, :form, w) for w in targets]
  a = findfirst(==(1), LINK_PATHS)                  # attachment slot of context path 1, the form
  a === nothing && error("LINK_PATHS has no form path")
  out = Dict{UInt64,Climb}()
  for fam in families
    side = fam == :cand ? 2 : fam == :dep ? 1 : error("unknown family $fam (use :cand or :dep)")
    item(i) = hash((ATTACH_TAG, side, a, codes[i]))      # as link_items makes it
    nodeitems = Dict{Int,Vector{UInt64}}()
    for w in eachindex(targets)
      order = UInt64[item(i) for i in R.order[w]]
      cuts = sort!(unique!([1; [min(k + 1, length(order)) for k in R.inner]; length(order)]))
      outer = [get!(() -> UInt64[item(i) for i in R.tree.nodes[x]], nodeitems, Int(x)) for x in R.outer[w]]
      out[item(w)] = Climb(order, cuts, outer)
    end
  end
  out
end
