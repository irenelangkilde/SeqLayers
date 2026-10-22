# suffixarray.jl  --  long char contexts for gap decisions
#
# CharIndex holds two suffix arrays over the training text: one over the reversed text
# (left contexts, read backwards from the char before a gap) and one over the forward
# text (right contexts, starting at the char after a gap).  Every row also records the
# gold label of the gap that context belongs to, with running totals per label, so the
# label counts for any context string -- all of whose occurrences form one contiguous
# range of rows -- take two lookups per label.
#
# A query extends the context one char at a time while it still occurs at least
# `mincount` times (up to `maxlen`), smoothing each longer context's label distribution
# toward the shorter one's (Dirichlet backoff with weight `beta`).
#
# Contexts from the training docs themselves are scored leave-one-out: the occurrence
# being labeled is subtracted, so long contexts don't look perfectly predictive.

const NLABELS = 4      # I, B, S, E

"Suffix array of s by prefix doubling: sort by (rank of first k chars, rank of next k chars)."
function suffixarray(s::Vector{UInt32})
  n = length(s)
  sa = collect(Int32(1):Int32(n))
  n == 0 && return sa
  rank = Int32.(s)
  tmp = similar(rank)
  k = 1
  while true
    sortround!(sa, rank, k)
    tmp[sa[1]] = 0
    for j in 2:n
      tmp[sa[j]] = tmp[sa[j-1]] + (pairkey(rank, sa[j], k) != pairkey(rank, sa[j-1], k))
    end
    rank, tmp = tmp, rank
    (n == 0 || rank[sa[n]] == n - 1 || k >= n) && break
    k *= 2
  end
  sa
end

pairkey(rank, i, k) = (rank[i], i + k <= length(rank) ? rank[i+k] : Int32(-1))   # past the end sorts first
sortround!(sa, rank, k) = sort!(sa; by = i -> pairkey(rank, i, k))

"Running label totals down the rows: cum[l, j+1] - cum[l, i] = count of label l in rows i:j."
function labelcounts(labels::Vector{UInt32})
  cum = zeros(Int32, NLABELS, length(labels) + 1)
  for (j, l) in enumerate(labels)
    cum[:, j+1] .= cum[:, j]
    1 <= l <= NLABELS && (cum[l, j+1] += 1)
  end
  cum
end

struct CharIndex
  fwd::Vector{UInt32}            # training text (codepoints); 0 separates documents
  rev::Vector{UInt32}            # reversed
  gap::Vector{UInt32}            # gold gap label per position of fwd
  saL::Vector{Int32}             # suffix array of rev (left contexts)
  saR::Vector{Int32}             # suffix array of fwd (right contexts)
  cumL::Matrix{Int32}
  cumR::Matrix{Int32}
  prior::Vector{Float64}         # overall label distribution
  offsets::IdDict{Doc,Int}       # training doc -> its offset in fwd (for leave-one-out)
  maxlen::Int
  mincount::Int
  beta::Float64
end

function CharIndex(docs::Vector{Doc}; maxlen = 12, mincount = 3, beta = 2.0)
  fwd, gap, offsets = UInt32[], UInt32[], IdDict{Doc,Int}()
  for d in docs
    d.gold === nothing && continue
    offsets[d] = length(fwd)
    append!(fwd, UInt32.(codepoint.(d.text)))
    append!(gap, d.gold.gap)
    push!(fwd, 0)
    push!(gap, NONE)
  end
  n = length(fwd)
  rev = reverse(fwd)
  saL, saR = suffixarray(rev), suffixarray(fwd)
  # left row: rev suffix starting at q covers fwd chars p, p-1, ... with p = n-q+1; its gap is p's
  cumL = labelcounts([gap[n-q+1] for q in saL])
  # right row: fwd suffix starting at s covers chars s, s+1, ...; its gap is the one before s
  cumR = labelcounts([s > 1 ? gap[s-1] : NONE for s in saR])
  tot = Float64.(cumL[:, end])
  prior = sum(tot) > 0 ? tot ./ sum(tot) : fill(1 / NLABELS, NLABELS)
  CharIndex(fwd, rev, gap, saL, saR, cumL, cumR, prior, offsets, maxlen, mincount, beta)
end

"Narrow rows lo:hi (all sharing their first d chars) to those whose char at offset d is c."
function narrow(sa, txt, lo, hi, d, c)
  key(j) = (t = sa[j] + d; t <= length(txt) ? txt[t] : UInt32(0))
  a, b = lo, hi + 1                      # first row with key >= c
  while a < b
    m = (a + b) >>> 1
    key(m) < c ? (a = m + 1) : (b = m)
  end
  start = a
  b = hi + 1                             # first row with key > c
  while a < b
    m = (a + b) >>> 1
    key(m) <= c ? (a = m + 1) : (b = m)
  end
  start, a - 1
end

"""
Label distribution for the gap after doc position p, from its left (side = :left) or right
context, and the context length reached.
"""
function context_probs(idx::CharIndex, doc::Doc, p::Int, side::Symbol)
  text = doc.text
  sa, txt, cum = side == :left ? (idx.saL, idx.rev, idx.cumL) : (idx.saR, idx.fwd, idx.cumR)
  qchar(d) = side == :left ? (p - d >= 1 ? UInt32(codepoint(text[p-d])) : nothing) :
             (p + 1 + d <= length(text) ? UInt32(codepoint(text[p+1+d])) : nothing)
  off = get(idx.offsets, doc, nothing)
  own = off === nothing ? NONE : idx.gap[off+p]          # leave-one-out label, if doc is in the index
  probs = copy(idx.prior)
  lo, hi, depth = 1, length(sa), 0
  c = zeros(Float64, NLABELS)
  for d in 0:idx.maxlen-1
    q = qchar(d)
    q === nothing && break
    lo, hi = narrow(sa, txt, lo, hi, d, q)
    lo > hi && break
    for l in 1:NLABELS
      c[l] = cum[l, hi+1] - cum[l, lo]
    end
    1 <= own <= NLABELS && (c[own] -= 1)
    N = sum(c)
    N < idx.mincount && break
    probs = (c .+ idx.beta .* probs) ./ (N + idx.beta)
    depth = d + 1
  end
  probs, depth
end

problevel(x) = x < 0.02 ? 0 : x < 0.2 ? 1 : x < 0.5 ? 2 : x < 0.8 ? 3 : 4

"Feature code for a label distribution: each label's probability in 5 levels (1..625)."
distcode(probs) = UInt32(1 + sum(problevel(probs[l]) * 5^(l - 1) for l in 1:NLABELS))
