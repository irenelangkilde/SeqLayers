# One root per sentence in an existing CoNLL-U file (2026-11-13), with the same repair write_conllu now makes
# (SeqLayers.root_groups): the extra roots of a sentence attach to its main root (the first VERB / AUX root, else the
# largest subtree) as parataxis; cycles are broken.  For prediction files written before
# the fix, so the CoNLL shared-task scorers accept them.
#
#   julia --project=experiments experiments/conllu_fix.jl in.conllu out.conllu

using SeqLayers
const S = SeqLayers

function fix(inpath, outpath)
  sents, cur = Vector{Vector{Vector{String}}}(), Vector{Vector{String}}()
  for line in eachline(inpath)
    if isempty(line)
      isempty(cur) || (push!(sents, cur); cur = Vector{String}[])
    elseif !startswith(line, '#')
      c = split(line, '\t')
      occursin(r"^\d+$", c[1]) || continue          # (no multiword / empty nodes in our output)
      push!(cur, String.(c))
    end
  end
  isempty(cur) || push!(sents, cur)
  sid = 0
  open(outpath, "w") do io
    for s in sents
      for (rg, hs, ds) in S.root_groups([parse(Int, c[7]) for c in s], [c[8] for c in s], [c[4] for c in s])
        sid += 1
        rows = s[rg]
        text = join([c[2] * (k < length(rows) && !occursin("SpaceAfter=No", c[10]) ? " " : "") for (k, c) in enumerate(rows)])
        println(io, "# sent_id = ", sid)
        println(io, "# text = ", text)
        for (k, c) in enumerate(rows)
          println(io, join((k, c[2], c[3], c[4], c[5], c[6], hs[k], ds[k], c[9], k == length(rows) ? "_" : c[10]), '\t'))
        end
        println(io)
      end
    end
  end
  println("$(length(sents)) sentences in, $sid out; repairs: ", S.CONLLU_FIXES)
end

fix(ARGS[1], ARGS[2])
