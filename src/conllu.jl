# conllu.jl  --  CoNLL-U -> documents with gold gap labels, heads, relations and tags
#
# Syntactic words are aligned to the "# text =" line.  Multiword tokens (don't -> do n't)
# align because their words concatenate to the surface string; sentences that don't align
# are dropped (and counted).

struct UDWord
  form::String
  upos::String
  xpos::String
  head::Int
  deprel::String
end

function parse_conllu(io::IO)
  sents = Tuple{String,Vector{UDWord},Bool}[]     # (text, words, starts new doc)
  text, words, newdoc = "", UDWord[], false
  for line in eachline(io)
    if isempty(strip(line))
      isempty(words) || push!(sents, (text, words, newdoc))
      text, words, newdoc = "", UDWord[], false
    elseif startswith(line, "# text =")
      text = strip(line[length("# text =")+1:end])
    elseif startswith(line, "# newdoc")
      newdoc = true
    elseif !startswith(line, '#')
      cols = split(line, '\t')
      occursin(r"[-.]", cols[1]) && continue      # multiword-token range or empty node
      push!(words, UDWord(cols[2], cols[4], cols[5], parse(Int, cols[7]), cols[8]))
    end
  end
  isempty(words) || push!(sents, (text, words, newdoc))
  sents
end

"Char spans of the words within `chars`, or nothing if they don't line up with the text."
function align(chars::Vector{Char}, words::Vector{UDWord})
  spans = UnitRange{Int}[]
  p = 1
  for w in words
    while p <= length(chars) && isspace(chars[p])
      p += 1
    end
    f = collect(w.form)
    (p + length(f) - 1 <= length(chars) && chars[p:p+length(f)-1] == f) || return nothing
    push!(spans, p:p+length(f)-1)
    p += length(f)
  end
  all(isspace, chars[p:end]) ? spans : nothing
end

function build_doc(sents)
  text, gap = Char[], UInt32[]
  head, deprel, ends = Dict{UnitRange{Int},UnitRange{Int}}(), Dict{UnitRange{Int},UInt32}(), Set{Int}()
  tags = Dict(name => Dict{UnitRange{Int},UInt32}() for name in [collect(keys(TAGSETS)); collect(keys(ALLHEADSETS))])
  dropped = 0
  for (stext, words, _) in sents
    chars = collect(stext)
    spans = align(chars, words)
    spans === nothing && (dropped += 1; continue)
    if !isempty(text)
      push!(text, ' ')
      push!(gap, NONE)
    end
    offset = length(text)
    append!(text, chars)
    append!(gap, fill(NONE, length(chars)))
    dspans = [s .+ offset for s in spans]
    for (wi, (w, s)) in enumerate(zip(words, dspans))
      gap[first(s):last(s)-1] .= GAP_I
      gap[last(s)] = (last(s) == length(text) || isspace(text[last(s)+1])) ? GAP_S : GAP_B
      head[s] = w.head == 0 ? (0:-1) : dspans[w.head]
      deprel[s] = deprelcode(w.deprel)
      for (name, code) in pairs(gold_tags(w.form, w.upos, w.xpos))
        tags[name][s] = code
      end
      for (name, code) in pairs(gold_head(wi, w.head, words))
        tags[name][s] = code
      end
    end
    push!(ends, last(dspans[end]))
    gap[last(dspans[end])] = GAP_E
  end
  Doc(text, Gold(gap, head, deprel, ends, tags)), dropped
end

"""
Read a CoNLL-U file into documents with gold annotation.  A document starts at each
`# newdoc` and holds at most `sents_per_doc` sentences.
"""
function read_conllu(path::AbstractString; sents_per_doc = 50, maxsents = typemax(Int))
  sents = open(parse_conllu, path)
  sents = sents[1:min(end, maxsents)]
  docs, dropped, group = Doc[], 0, eltype(sents)[]
  flush() = isempty(group) || begin
    d, n = build_doc(group)
    isempty(d.text) || push!(docs, d)
    dropped += n
    empty!(group)
  end
  for s in sents
    (s[3] || length(group) >= sents_per_doc) && flush()
    push!(group, s)
  end
  flush()
  dropped > 0 && @info "read_conllu: dropped $dropped of $(length(sents)) sentences that don't align with their text" path
  docs
end

"A document from raw text, without gold annotation."
doc_from_text(s::AbstractString) = Doc(collect(s), nothing)
