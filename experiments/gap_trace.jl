# gap_trace.jl  --  where does the full decode lose to the ceiling?  (user's request 2026-11-03)
#
#   julia --project=experiments experiments/gap_trace.jl [dev_eval.jl / ceiling.jl options] devtb=EWT,CHILDES examples=12
#
# For each chosen dev doc, scores every attachment twice with the same model: in the perfect-context ceiling (as
# ceiling.jl: everything else gold) and in a full decode (with the decode options: softheads, candmemo, attachwait,
# rescoreevery, fcommit ...).  At each attachment the decode commits, notes the state around the token: its own
# UPOS / XPOS right or not, its neighbours' tags right, its neighbours' attachments still pending, the gold head
# among the candidates, and (kinitems) the gold head's kinship items unknown; attachments settled by inference are
# marked as such.  Prints right / wrong in each, then for the tokens the ceiling gets right and the decode wrong the
# share showing each condition (and for comparison the same shares over the decode's right ones), then examples.

include(joinpath(@__DIR__, "dev_eval.jl"))
include(joinpath(@__DIR__, "ceiling_lib.jl"))
const S = SeqLayers
import SeqLayers: observe!, segment_done!, segment_discard!

DEFAULTS["examples"] = 12
DEFAULTS["headconf"] = 1.0
DEFAULTS["domains"] = "product"
DEFAULTS["domeps"] = 0.0
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
idx = s.idx
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true",
  relbounded = cfg["relbounded"] == "true", relnone = cfg["relnone"] == "true", treerank = cfg["treerank"] == "true",
  headtrank = cfg["headtrank"] == "true", headwc = Int[], headsibs = headsib_names(cfg), idxeq = cfg["idxeq"] == "true",
  kinitems = cfg["kinitems"] == "true", headkin = headkin_names(cfg), neighbor2 = cfg["neighbor2"] == "true",
  oclass = cfg["oclass"] == "true", bioes = cfg["bioes"] == "true", fwdsets = cfg["fwdsets"] == "true", shapeitems = cfg["shapeitems"] != "false", wc3 = s.w3)
eventsdir, opts = s.recipe
pc = S.load_counts(eventsdir, eventsig(eventsdir))
m = ChainModel(eventsdir; opts...)
pc !== nothing && (m.pcounts = pc)
m.generalize = S.generalization_slots(specs)
lex = s.lex
docs = [d for (tb, d) in s.devdocs if isempty(cfg["devtb"]) || tb in split(cfg["devtb"], ',')]
spanof(st, t) = t == NONE ? (0:-1) : S.tokspan(st, Int(t))

# 1. the ceiling, per token: (gold head span, predicted head span)
ceil = Dict{UnitRange{Int},Tuple{UnitRange{Int},UnitRange{Int}}}()
TOKEN_HOOK[] = (st, t, a, pa, rel, pr) -> (ceil[S.tokspan(st, t)] = (spanof(st, a), spanof(st, pa)))
t0 = time()
for d in docs
  ceiling_counts(d, specs, m, lex; scoretags = false, headconf = cfg["headconf"], domains = Symbol(cfg["domains"]), domeps = cfg["domeps"])
end
TOKEN_HOOK[] = nothing
println("ceiling: $(length(ceil)) attachments in $(round(time() - t0; digits = 1)) s")

# 2. the decode, with its options, noting the state at each attachment commit
cfg["softheads"] == "true" && (S.SOFTHEADS[] = true)
cfg["attachwait"] == "true" && (S.ATTACH_WAIT[] = true)
cfg["fcommit"] == "true" && (S.FCOMMIT[] = true)
S.RESCORE_EVERY[] = cfg["rescoreevery"]
S.DISPLACE[] = cfg["displace"]; S.DISPLACE_CAP[] = cfg["displacecap"]
S.PROD_NORM[] = Symbol(cfg["prodnorm"])
S.SOFTTAGS[] = cfg["softtags"]
empty!(m.memo); empty!(m.cmemo)
struct GapRec end
const NOTES = Dict{UnitRange{Int},NamedTuple}()
function observe!(::GapRec, st, key, choice)
  key[1] == :token && key[2] == 1 || return
  t = key[3]
  toks = st.layers[:token]
  gold = st.doc.gold
  sp = S.tokspan(st, t)
  tagok(u) = (u == NONE || u == PENDING) ? missing :
    get(gold.tags[:upos], S.tokspan(st, Int(u)), 0) == toks[:upos, Int(u)] && get(gold.tags[:xpos], S.tokspan(st, Int(u)), 0) == toks[:xpos, Int(u)]
  ln, rn = S.adjacent_token(st, t, -1), S.adjacent_token(st, t, 1)
  pend(u) = u == PENDING ? true : u == NONE ? false : toks[:attach, Int(u)] == PENDING
  gh = get(gold.head, sp, nothing)
  gtok = gh === nothing ? nothing : isempty(gh) ? NONE : something(findfirst(u -> S.tokspan(st, u) == gh, 1:length(toks)), PENDING)
  cands = S.decisiondef(st, key).candidates(st, t)
  ingold = gtok !== nothing && gtok != PENDING && UInt32(gtok) in cands
  # which filter (attach_candidates' order) removed the gold head, if it isn't a candidate
  prune = ingold ? :none : gtok === nothing ? :misaligned : gtok == PENDING ? :tokenization : pruned_by(st, t, UInt32(gtok))
  kinunk = cfg["kinitems"] == "true" && ingold ? any(dir -> S.kin_value(st, t, UInt32(gtok), dir) == PENDING, (-1, 1)) : missing
  NOTES[sp] = (owntag = tagok(t), nbtag = all(skipmissing([tagok(ln), tagok(rn)])), nbpending = pend(ln) || pend(rn),
    goldincands = ingold, prune = prune, ncands = length(cands), kinunknown = kinunk, ncommitted = count(u -> toks[:attach, u] != PENDING, 1:length(toks)),
    words = [S.tokstr(st, u) for u in sort(1:length(toks); by = u -> toks[:first_char, u])])
end
"The first of attach_candidates' filters that excludes head h for t (:other if none does: e.g. changed since)."
function pruned_by(st, t, h)
  toks = st.layers[:token]
  s_ = toks[:sentence, t]
  if h == NONE
    hasroot = s_ != PENDING && any(u -> u != t && toks[:sentence, u] == s_ && toks[:attach, u] == NONE, 1:length(toks))
    return hasroot ? :root_taken : :other
  end
  hh = Int(h)
  toks[:sentence, hh] == s_ || return :sentence
  abs(Int(toks[:first_char, hh]) - Int(toks[:first_char, t])) <= S.MAX_LINK_CHARS || return :distance
  S.creates_cycle(st, t, h) && return :cycle
  S.crosses_E(st, t, h) && return :sentence_end
  S.head_consistent(st, t, h) || return :headprops
  S.crosses_arc(st, t, h) && return :projectivity
  :other
end
segment_done!(::GapRec, st) = nothing
segment_discard!(::GapRec) = nothing
dec = Dict{UnitRange{Int},UnitRange{Int}}()
t0 = time()
for d in docs
  for ps in process(d, specs, m, lex; observer = GapRec()), (k, sp) in enumerate(ps.spans)
    dec[sp] = ps.heads[k] == 0 ? (0:-1) : ps.spans[ps.heads[k]]
  end
end
println("decode: $(length(dec)) attachments in $(round(time() - t0; digits = 1)) s")

# 3. join
cats = Dict{Symbol,Vector{UnitRange{Int}}}(k => UnitRange{Int}[] for k in (:both, :ceilonly, :deconly, :neither))
for (sp, (g, cp)) in ceil
  haskey(dec, sp) || continue
  c, dd = cp == g, dec[sp] == g
  push!(cats[c && dd ? :both : c ? :ceilonly : dd ? :deconly : :neither], sp)
end
n = sum(length, values(cats))
println("\n$n tokens in both: both right $(length(cats[:both])), ceiling only $(length(cats[:ceilonly])), decode only $(length(cats[:deconly])), neither $(length(cats[:neither]))")
share(sps, f) = (v = [f(NOTES[sp]) for sp in sps if haskey(NOTES, sp)]; v = collect(skipmissing(v)); isempty(v) ? "-" : string(round(count(v) / length(v); digits = 3)))
inferred(sps) = round(count(sp -> !haskey(NOTES, sp), sps) / max(length(sps), 1); digits = 3)
println("\nconditions at the decode's commit\t\tceiling right, decode wrong\tdecode right")
for (nm, f) in (("own UPOS / XPOS wrong", x -> x.owntag === missing ? missing : !x.owntag), ("a neighbour's tags wrong", x -> !x.nbtag),
                ("a neighbour's attachment pending", x -> x.nbpending), ("gold head not a candidate", x -> !x.goldincands),
                ("gold head's kinship unknown", x -> x.kinunknown))
  println(rpad(nm, 40), "\t", share(cats[:ceilonly], f), "\t\t\t\t", share(vcat(cats[:both], cats[:deconly]), f))
end
println(rpad("inferred, not decided", 40), "\t", inferred(cats[:ceilonly]), "\t\t\t\t", inferred(vcat(cats[:both], cats[:deconly])))
for (nm, grp) in (("ceiling right, decode wrong", cats[:ceilonly]), ("both wrong", cats[:neither]))
  rs = [NOTES[sp].prune for sp in grp if haskey(NOTES, sp) && NOTES[sp].prune != :none]
  isempty(rs) && continue
  println("\ngold head pruned ($nm): $(length(rs)) of $(length(grp)) -- ",
    join(["$k $(count(==(k), rs))" for k in sort(unique(rs); by = k -> -count(==(k), rs))], ", "))
end
med(sps, f) = (v = [f(NOTES[sp]) for sp in sps if haskey(NOTES, sp)]; isempty(v) ? "-" : string(sort(v)[cld(length(v), 2)]))
println(rpad("candidates (median)", 40), "\t", med(cats[:ceilonly], x -> x.ncands), "\t\t\t\t", med(vcat(cats[:both], cats[:deconly]), x -> x.ncands))
println(rpad("attachments already committed (median)", 40), "\t", med(cats[:ceilonly], x -> x.ncommitted), "\t\t\t\t", med(vcat(cats[:both], cats[:deconly]), x -> x.ncommitted))

# 4. examples
alltext = Dict(sp => d for d in docs for sp in keys(d.gold.head))
word(sp) = isempty(sp) ? "ROOT" : (d = get(alltext, sp, nothing); d === nothing ? "?" : String(d.text[sp]))
println("\nexamples (ceiling right, decode wrong):")
for sp in first(sort(cats[:ceilonly]; by = first), cfg["examples"])
  g, _ = ceil[sp]
  x = get(NOTES, sp, nothing)
  println("  '$(word(sp))' gold head '$(word(g))', decode '$(word(dec[sp]))'",
    x === nothing ? "  (inferred)" : "  [own tags $(x.owntag === missing ? "?" : x.owntag ? "ok" : "WRONG"), nb tags $(x.nbtag ? "ok" : "WRONG"), nb pending $(x.nbpending), gold in cands $(x.goldincands) of $(x.ncands)$(x.prune == :none ? "" : " (pruned: $(x.prune))"), kin unknown $(x.kinunknown), committed $(x.ncommitted)]")
end
