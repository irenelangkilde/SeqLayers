# driver.jl  --  segments, carryover, training and evaluation

struct ParsedSentence
  span::UnitRange{Int}             # doc char range
  tokens::Vector{String}
  spans::Vector{UnitRange{Int}}    # doc char range of each token
  heads::Vector{Int}               # 0 = root, else 1-based index within the sentence
  deprels::Vector{String}
end

nextnonspace(text, p) = (while p <= length(text) && isspace(text[p]); p += 1; end; p)

"End of the segment starting at s: the last char of its `words`-th whitespace-separated chunk."
function segment_end(text, s, words)
  p, chunks = s, 0
  while p <= length(text)
    while p <= length(text) && !isspace(text[p])
      p += 1
    end
    chunks += 1
    chunks == words && return p - 1
    p = nextnonspace(text, p)
  end
  length(text)
end

function sentences(st::State)
  toks, sents = st.layers[:token], st.layers[:sentence]
  out = ParsedSentence[]
  for s in 1:length(sents)
    ts = sort!([t for t in 1:length(toks) if toks[:sentence, t] == s]; by = t -> toks[:first_char, t])
    idx = Dict(t => k for (k, t) in enumerate(ts))
    spans = [tokspan(st, t) for t in ts]
    push!(out, ParsedSentence(first(spans[1]):last(spans[end]), [tokstr(st, t) for t in ts], spans,
      [toks[:attach, t] == NONE ? 0 : idx[Int(toks[:attach, t])] for t in ts],
      [DEPRELS[toks[:deprel, t]] for t in ts]))
  end
  sort!(out; by = s -> first(s.span))
end

const MAX_GROW = 8

"""
Tokenize and parse a document segment by segment.  Material after a segment's last closed
sentence is carried into the next segment and re-processed.  A segment that closes no
sentence is retried at double size, up to MAX_GROW times, then force-closed.
"""
function process(doc::Doc, specs, model, lex::Lexicon; words = 128, edge = 16, observer = nothing)
  out = ParsedSentence[]
  n = length(doc.text)
  s, grow = nextnonspace(doc.text, 1), 1
  while s <= n
    e = segment_end(doc.text, s, words * grow)
    final = e == n
    st = State(doc, s:e, final, final ? 0 : edge, specs, lex)
    st.infer = !(model isa OracleModel)          # the oracle commits gold itself
    solve!(st, model; observer)
    if final || grow >= MAX_GROW
      force_close!(st)
    elseif st.closed_through == 0
      segment_discard!(observer)
      grow *= 2
      continue
    end
    segment_done!(observer, st)
    settle_closed!(st)
    append!(out, sentences(st))
    s, grow = nextnonspace(doc.text, s + st.closed_through), 1
  end
  out
end

"""
Round 1 of training: replay gold decisions in left-to-right order through the same
engine, counting every candidate's features as they looked when the decision was made.
TODO round 2: decode with the trained model, replay gold in the order the model chooses.
"""
function train!(model::CountModel, docs::Vector{Doc}, specs, lex::Lexicon; kw...)
  rec = Recorder(model)
  for d in docs
    process(d, specs, OracleModel(), lex; observer = rec, kw...)
  end
  model
end

const EVAL_COUNTS = (:tok_gold, :tok_pred, :tok_ok, :uas, :las, :sent_gold, :sent_pred, :sent_ok)

"Token, attachment (UAS/LAS over correctly tokenized words) and sentence-boundary scores."
evaluate(docs::Vector{Doc}, specs, model, lex::Lexicon; kw...) = eval_scores(evaluate_counts(docs, specs, model, lex; kw...))

"Scores from counts (which can be summed across documents, treebanks or worker processes)."
function eval_scores(c::Dict{Symbol,Int})
  f1(ok, p, g) = ok == 0 ? 0.0 : 2ok / (p + g)
  (token_f1 = f1(c[:tok_ok], c[:tok_pred], c[:tok_gold]),
    uas = c[:uas] / max(c[:tok_ok], 1), las = c[:las] / max(c[:tok_ok], 1),
    sentence_f1 = f1(c[:sent_ok], c[:sent_pred], c[:sent_gold]))
end

"The raw counts behind `evaluate`."
function evaluate_counts(docs::Vector{Doc}, specs, model, lex::Lexicon; kw...)
  c = Dict(k => 0 for k in EVAL_COUNTS)
  for d in docs
    gold = d.gold
    pred = process(d, specs, model, lex; kw...)
    predhead = Dict{UnitRange{Int},Tuple{UnitRange{Int},String}}()
    for ps in pred, (k, sp) in enumerate(ps.spans)
      predhead[sp] = (ps.heads[k] == 0 ? (0:-1) : ps.spans[ps.heads[k]], ps.deprels[k])
    end
    c[:tok_gold] += length(gold.head)
    c[:tok_pred] += length(predhead)
    for (sp, (h, rel)) in predhead
      haskey(gold.head, sp) || continue
      c[:tok_ok] += 1
      h == gold.head[sp] && (c[:uas] += 1; rel == DEPRELS[gold.deprel[sp]] && (c[:las] += 1))
    end
    predends = Set(last(ps.span) for ps in pred)
    c[:sent_gold] += length(gold.sentends)
    c[:sent_pred] += length(predends)
    c[:sent_ok] += length(intersect(predends, gold.sentends))
  end
  c
end
