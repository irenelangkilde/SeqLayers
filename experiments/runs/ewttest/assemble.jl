# Assemble the EWT test predictions for the official scorer (2026-11-14): every test doc's prediction in the test file's
# doc order -- shard 1 from first-pred.conllu (its docs in file order, each starting at "# sent_id = 1"), shard 2 from the
# per-doc files second-doc<hash>.conllu / help-doc<hash>.conllu -- and the test file itself as the gold.  (The runs' own
# -gold.conllu files are wrong for shards: dev_eval writes the first n sentences of the test file.)
#
#   julia --project=experiments experiments/runs/ewttest/assemble.jl
# With "list": instead, print shard 2's doc hashes (one per line), for the helper's coverage check.

using SeqLayers

const OUT = joinpath(@__DIR__)
const TEST = expanduser("~/Projects/data/unidep/ud-2.2/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu")

ds = read_conllu(TEST)
L = [length(d.text) for d in ds]
o = sortperm(L, rev = true)
shard1, shard2 = sort(o[1:2:end]), sort(o[2:2:end])           # (as dev_eval shard=k/2: every 2nd doc by size, file order)
hx(d) = string(hash(String(d.text)); base = 16)

if !isempty(ARGS) && ARGS[1] == "list"
  foreach(i -> println(hx(ds[i])), shard2)
  exit()
end

# each predicted doc matched to its test doc by content: its tokens' characters without spaces (2026-11-14: the helper's
# skips depended on timing, so the order of its merged help-pred.conllu isn't known in advance)
bykey = Dict(replace(String(d.text), r"\s" => "") => i for (i, d) in enumerate(ds))
docblocks(text) = (st = [first(r) for r in findall(r"^# sent_id = 1\n"m, text)];
  [text[st[k]:(k < length(st) ? prevind(text, st[k+1]) : lastindex(text))] for k in eachindex(st)])
key(block) = join(split(l, '\t')[2] for l in split(block, '\n') if !isempty(l) && !startswith(l, '#'))
pred, src = Dict{Int,String}(), Dict{String,Int}()
function add!(block, from)
  i = get(bykey, key(block), 0)
  i == 0 && error("a predicted doc from $from matches no test doc")
  haskey(pred, i) && return
  pred[i] = block; src[from] = get(src, from, 0) + 1
end
foreach(b -> add!(b, "first-pred"), docblocks(read(joinpath(OUT, "first-pred.conllu"), String)))
for i in shard2
  f = joinpath(OUT, "second-doc$(hx(ds[i])).conllu")
  isfile(f) && add!(read(f, String), "second")
end
isfile(joinpath(OUT, "help-pred.conllu")) && foreach(b -> add!(b, "help-pred"), docblocks(read(joinpath(OUT, "help-pred.conllu"), String)))
println("docs by source: ", src)
missing_ = [i for i in eachindex(ds) if !haskey(pred, i)]
isempty(missing_) || error("no prediction for $(length(missing_)) docs: $(missing_[1:min(end, 10)])")
open(io -> foreach(i -> write(io, pred[i]), eachindex(ds)), joinpath(OUT, "ewttest-pred.conllu"), "w")
cp(TEST, joinpath(OUT, "ewttest-gold.conllu"); force = true)
println("assembled $(length(ds)) docs: ewttest-pred.conllu, ewttest-gold.conllu")
