# lookup_counts.jl  --  what do the treebanks and the chain model's training records say about a word?
#
#   julia --project=experiments experiments/lookup_counts.jl form=ran [upos=VERB] [headupos=VERB] [train_sents=500] [events=gold]
#
# 1. Treebank counts (full train and dev files of the active treebanks): the word's UPOS, its head's
#    UPOS, and the pair.
# 2. The recorded events (experiments/work/train<N>-...-<events>/events, as dev_eval.jl records
#    them, before grouping): for decisions of tokens with this form,
#      upos       records per candidate label, chosen / not chosen
#      head_upos  the same
#      attach     candidate records with candidate UPOS = `headupos`: chosen / not chosen
#    i.e. the counts behind feature sets {dep form = form} and {dep form = form, cand upos = headupos}.
#      as head    attach records whose candidate is the word with UPOS `upos` (cand form = form,
#                 cand upos = upos): chosen (it is the head) / not chosen, by the dependent's UPOS.
#    With the treebanks' dependents of the word (by dependent UPOS and relation) for comparison.

include(joinpath(@__DIR__, "dev_eval.jl"))
const S = SeqLayers

DEFAULTS["form"] = "ran"
DEFAULTS["upos"] = "VERB"
DEFAULTS["headupos"] = "VERB"
cfg = parse_cfg(ARGS)
form, upos, hupos = lowercase(cfg["form"]), cfg["upos"], cfg["headupos"]

trainfiles = filter(l -> !startswith(l, '#') && !isempty(strip(l)), readlines(CORPUS_LIST))
println("== treebanks (full files): \"$form\"")
for part in ("train", "dev")
  tag, htag, pair, pertb = Dict{String,Int}(), Dict{String,Int}(), Dict{Tuple{String,String},Int}(), Dict{String,Int}()
  deps, rels = Dict{String,Int}(), Dict{String,Int}()
  for f in trainfiles
    g = part == "train" ? f : replace(f, "-ud-train" => "-ud-dev")
    sent = Vector{SubString{String}}[]
    flush() = begin
      byid = Dict(c[1] => c for c in sent)
      for c in sent
        lowercase(c[2]) == form || continue
        ht = c[7] == "0" ? "ROOT" : haskey(byid, c[7]) ? byid[c[7]][4] : "?"
        tag[c[4]] = get(tag, c[4], 0) + 1
        htag[ht] = get(htag, ht, 0) + 1
        pair[(c[4], ht)] = get(pair, (c[4], ht), 0) + 1
        (c[4] == upos && ht == hupos) && (pertb[tbname(f)] = get(pertb, tbname(f), 0) + 1)
      end
      for c in sent                                    # dependents of the word (as `upos`)
        h = get(byid, c[7], nothing)
        (h !== nothing && lowercase(h[2]) == form && h[4] == upos) || continue
        deps[c[4]] = get(deps, c[4], 0) + 1
        rels[String(first(split(c[8], ':')))] = get(rels, String(first(split(c[8], ':'))), 0) + 1
      end
      empty!(sent)
    end
    for l in eachline(g)
      isempty(l) && (flush(); continue)
      startswith(l, '#') && continue
      c = split(l, '\t')
      occursin(r"^\d+$", c[1]) && push!(sent, c)
    end
    flush()
  end
  srt(d) = join(["$k $v" for (k, v) in sort(collect(d); by = p -> -p[2])], ", ")
  println("$part: $(sum(values(tag); init = 0)) occurrences; UPOS: $(srt(tag)); head UPOS: $(srt(htag))")
  println("  UPOS = $upos and head UPOS = $hupos: $(get(pair, (upos, hupos), 0))  (by treebank: $(srt(pertb)))")
  println("  as the head ($upos) of other words: $(sum(values(deps); init = 0)) dependents; their UPOS: $(srt(deps)); relations: $(srt(rels))")
end

# the recorded events
tbs = tbname.(trainfiles)
workdir = joinpath(HERE, "work", "train$(cfg["train_sents"])-" * join(tbs, "+") * "-" * cfg["events"])
lex = deserialize(joinpath(workdir, "lex.jls"))
recs, items = S.load_events(joinpath(workdir, "events"))
code = get(get(lex.vals, :form, Dict{String,UInt32}()), form, nothing)
println("\n== recorded events in $workdir: \"$form\"")
code === nothing && (println("not in the lexicon: never seen in the training sample"); exit())
specs = default_specs()
toks = specs[2]
k_of(name) = findfirst(d -> d.name == name, toks.decisions)
formslot = 1                                           # context path 1 (form) is the first label item
# label decisions: items are the context values, so slot 1 = (tag, 1, form)
for name in (:upos, :head_upos)
  k = k_of(name)
  tk, tag = S.typekey(:token, k), hash("token." * String(name))
  x = hash((tag, formslot, code))
  labels = name == :upos ? S.TAGSETS.upos : S.HEADSETS.head_upos
  c = Dict{String,Vector{Int}}()
  for r in recs
    r.typekey == tk && r.n >= formslot && items[r.start+formslot-1] == x || continue
    v = get!(() -> [0, 0], c, labels[r.label])
    v[r.chosen == 1 ? 1 : 2] += 1
  end
  println("$name decisions of \"$form\" tokens: ", isempty(c) ? "none" :
    join(["$l: chosen $(v[1]), not chosen $(v[2])" for (l, v) in sort(collect(c); by = p -> -p[2][1] - p[2][2])], "; "))
end
# attachment: slots 1 = distance, 2 = direction, then dependent/candidate value per link path
a_form = findfirst(==(1), S.LINK_PATHS)
a_upos = findfirst(==(7), S.LINK_PATHS)                # context path 7 = upos
xd = hash((S.ATTACH_TAG, 1, a_form, code))
ucode = S.tagcode(:upos, hupos)
xc = hash((S.ATTACH_TAG, 2, a_upos, ucode))
let tk = S.typekey(:token, 1), n = 0, cand = 0, chosen = 0, chosen_any = 0
for r in recs
  r.typekey == tk || continue
  items[r.start+2a_form] == xd || continue            # slot 2 + 2a - 1 = the dependent's form
  n += 1
  chosen_any += r.chosen == 1
  items[r.start+2a_upos+1] == xc || continue          # slot 2 + 2a = the candidate's upos
  cand += 1
  chosen += r.chosen == 1
end
println("attach decisions of \"$form\" tokens: $n candidate records ($chosen_any chosen, i.e. decisions with the gold head among them)")
println("  with candidate UPOS = $hupos: $cand records -- chosen $chosen, not chosen $(cand - chosen)",
  cand > 0 ? "  (odds $(round(chosen / max(cand - chosen, 1); digits = 3)))" : "")
end
# the word as a candidate head
let tk = S.typekey(:token, 1), a_dupos = a_upos
  xf = hash((S.ATTACH_TAG, 2, a_form, code))
  xu = hash((S.ATTACH_TAG, 2, a_upos, S.tagcode(:upos, upos)))
  depname = Dict(hash((S.ATTACH_TAG, 1, a_dupos, UInt32(i))) => l for (i, l) in enumerate(S.TAGSETS.upos))
  by = Dict{String,Vector{Int}}()
  for r in recs
    r.typekey == tk || continue
    (items[r.start+2a_form+1] == xf && items[r.start+2a_upos+1] == xu) || continue
    v = get!(() -> [0, 0], by, get(depname, items[r.start+2a_dupos], "?"))
    v[r.chosen == 1 ? 1 : 2] += 1
  end
  c1, c2 = sum(first, values(by); init = 0), sum(last, values(by); init = 0)
  println("as a candidate head (cand form = $form, cand upos = $upos): $(c1 + c2) records -- chosen $c1, not chosen $c2",
    c1 + c2 > 0 ? "  (odds $(round(c1 / max(c2, 1); digits = 3)))" : "")
  for (l, v) in sort(collect(by); by = p -> -p[2][1] - p[2][2])
    println("  dep upos = ", rpad(l, 6), " chosen $(v[1]), not chosen $(v[2])")
  end
end
