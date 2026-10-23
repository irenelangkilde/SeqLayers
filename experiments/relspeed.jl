# relspeed.jl  --  time recording training events with and without relations=true, on a few treebank
# documents (for optimizing the relation rows).
#
#   julia --project=experiments experiments/relspeed.jl [nsents=60]
using SeqLayers
nsents = length(ARGS) >= 1 ? parse(Int, split(ARGS[1], '=')[2]) : 60
f = "/Users/irenelangkildegeary2015/Projects/data/unidep/unideps-English/UD_English-GUM/en_gum-ud-train.conllu"
docs = read_conllu(f)
pick, k = Doc[], 0
for d in docs
  global k
  k >= nsents && break
  push!(pick, d); k += length(d.gold.sentends)
end
for rel in (false, true)
  specs = default_specs(relations = rel)
  dir = mktempdir()
  record_events(pick[1:1], specs, Lexicon(), dir)                     # compile
  t = @elapsed record_events(pick, specs, Lexicon(), dir)
  println("relations=$rel: $(k) sentences recorded in $(round(t; digits = 2)) s")
end
