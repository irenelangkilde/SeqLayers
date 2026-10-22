# lattice.jl  --  calculated (not learned) corrections for redundant features
#
# For a candidate with active items S, the log-odds that it is chosen are expanded over the
# subsets of S (Möbius expansion):
#
#     log o(S) = log prior + Σ_{∅ ≠ T ⊆ S} λ_T
#
# Each λ_T is calculated from counts, order by order:
#
#     e_T = prior × exp(Σ_{∅ ≠ U ⊊ T} λ_U)              what the lower orders predict for T
#     λ_T = log smoothOdds(c1_T, c2_T, e_T) − log e_T    how far T's own counts disagree
#
# so a set with no data gets λ = 0, and redundant features get negative higher-order terms
# that cancel the evidence they duplicate.  Up to the full order the sum reproduces the
# smoothed joint odds of S exactly; truncated at `maxorder` it approximates them.
#
# Pruning, as in Phase N: a k-set is counted only if all of its (k−1)-subsets have enough
# support (Apriori), and its λ is kept only if |λ| ≥ diffthreshold or p·|λ| ≥ wmothreshold
# (p = smoothed probability of being chosen).  A dropped λ counts as exactly 0, both when
# calculating higher orders and when scoring.
#
# Training decisions are recorded once (by replaying gold) into two mmapped files: candidate
# records and their sorted items.  Each order is then one streaming pass over those files.

using Mmap

"One candidate of one training decision.  Items are `n` UInt64s starting at `start` in items.bin, in slot order."
struct CandRec
  typekey::UInt64      # decision type
  label::UInt32        # candidate label for labeled decisions, else 0
  chosen::UInt32       # 1 if this was the gold choice
  start::UInt64
  n::UInt64
end

const CANDS_FILE = "cands.bin"
const ITEMS_FILE = "items.bin"

##======== recording events

mutable struct EventRecorder
  candio::IOStream
  itemio::IOStream
  nitems::Int
  buffer::Vector{Tuple{Int,Vector{Tuple{UInt64,UInt32,Bool}},Vector{Vector{UInt64}}}}
end
function EventRecorder(dir::AbstractString)
  mkpath(dir)
  EventRecorder(open(joinpath(dir, CANDS_FILE), "w"), open(joinpath(dir, ITEMS_FILE), "w"), 0, [])
end

function observe!(r::EventRecorder, st::State, key::AgendaKey, choice::UInt32)
  d, i = decisiondef(st, key), key[3]
  cands = copy(d.candidates(st, i))
  choice in cands || push!(cands, choice)
  tk = typekey(key[1], key[2])
  # items are kept in slot order (one per context path etc.); LatticeModel sorts them itself
  push!(r.buffer, (st.layers[key[1]].spec.anchor(st, i),
    [(tk, d.labeled ? c : UInt32(0), c == choice) for c in cands],
    [d.items(st, i, c) for c in cands]))
end

"Write out decisions inside closed sentences; the rest is re-processed with the next segment."
function segment_done!(r::EventRecorder, st::State)
  for (anchor, cands, items) in r.buffer
    anchor <= st.closed_through || continue
    for ((tk, label, chosen), I) in zip(cands, items)
      write(r.candio, [CandRec(tk, label, chosen, r.nitems + 1, length(I))])
      write(r.itemio, I)
      r.nitems += length(I)
    end
  end
  empty!(r.buffer)
end
segment_discard!(r::EventRecorder) = empty!(r.buffer)
Base.close(r::EventRecorder) = (close(r.candio); close(r.itemio))

"""
Records every decision from the finished gold state of a segment (omnidirectional): each one
sees all other decisions at their gold values, with only itself and what's derived from it
masked, and is recorded against all its candidates before constraints (`allcandidates`).
"""
mutable struct GoldRecorder
  candio::IOStream
  itemio::IOStream
  nitems::Int
end
function GoldRecorder(dir::AbstractString)
  mkpath(dir)
  GoldRecorder(open(joinpath(dir, CANDS_FILE), "w"), open(joinpath(dir, ITEMS_FILE), "w"), 0)
end
observe!(::GoldRecorder, st::State, key::AgendaKey, choice::UInt32) = nothing
segment_discard!(::GoldRecorder) = nothing
Base.close(r::GoldRecorder) = (close(r.candio); close(r.itemio))

function segment_done!(r::GoldRecorder, st::State)
  for lname in st.order
    seq = st.layers[lname]
    for (k, d) in enumerate(seq.spec.decisions), i in 1:seq.ncols
      choice = seq[d.name, i]
      choice == PENDING && continue
      seq.spec.anchor(st, i) <= st.closed_through || continue   # the rest is redone with the next segment
      cands = copy(d.allcandidates(st, i))
      isempty(cands) && continue
      choice in cands || push!(cands, choice)
      key = (lname, k, i)
      restore = d.mask(st, key)
      items = [d.items(st, i, c) for c in cands]
      restore()
      tk = typekey(lname, k)
      for (c, I) in zip(cands, items)
        write(r.candio, [CandRec(tk, d.labeled ? c : UInt32(0), c == choice, r.nitems + 1, length(I))])
        write(r.itemio, I)
        r.nitems += length(I)
      end
    end
  end
end

"""
Record training events for `docs` into `dir`.  mode = :gold (default) records every decision
from the finished gold state (omnidirectional); mode = :replay records decisions as a
left-to-right gold replay commits them.
"""
function record_events(docs::Vector{Doc}, specs, lex::Lexicon, dir::AbstractString; mode = :gold, kw...)
  r = mode == :gold ? GoldRecorder(dir) : mode == :replay ? EventRecorder(dir) : error("unknown mode $mode")
  try
    for d in docs
      process(d, specs, OracleModel(), lex; observer = r, kw...)
    end
  finally
    close(r)
  end
  dir
end

function load_events(dir::AbstractString)
  cp, ip = joinpath(dir, CANDS_FILE), joinpath(dir, ITEMS_FILE)
  nc, ni = filesize(cp) ÷ sizeof(CandRec), filesize(ip) ÷ sizeof(UInt64)
  recs = nc == 0 ? CandRec[] : Mmap.mmap(cp, Vector{CandRec}, nc)
  items = ni == 0 ? UInt64[] : Mmap.mmap(ip, Vector{UInt64}, ni)
  recs, items
end

##======== the model

struct LatticeModel <: AbstractModel
  lambda::Dict{UInt64,Float64}                 # set key -> λ (significant sets only)
  supported::Set{UInt64}                       # set keys with enough support (for Apriori)
  prior::Dict{UInt64,Tuple{Float64,Float64}}   # prior key -> (chosen, unchosen)
  maxorder::Int
end

setkey(tk::UInt64, label::UInt32, T::Vector{UInt64}) = hash((tk, label, T))
lpriorkey(tk::UInt64, label::UInt32) = hash((tk, label))      # label 0: the whole decision type

priorodds(m::LatticeModel, pk::UInt64) = (p = get(m.prior, pk, (0.0, 0.0)); (p[1] + 0.5) / (p[2] + 0.5))

"Do all (k−1)-subsets of T have support?"
function subsets_supported(m::LatticeModel, tk, label, T)
  length(T) == 1 && return true
  for drop in eachindex(T)
    setkey(tk, label, T[setdiff(eachindex(T), drop)]) in m.supported || return false
  end
  true
end

"Call f(T, key) for each k-subset T of I (sorted) whose (k−1)-subsets all have support."
function foreach_kset(f, m::LatticeModel, tk, label, I, k, T = UInt64[], start = 1)
  for j in start:length(I)
    push!(T, I[j])
    key = setkey(tk, label, T)
    if length(T) == k
      subsets_supported(m, tk, label, T) && f(T, key)
    elseif key in m.supported           # only supported prefixes can lead to counted k-sets
      foreach_kset(f, m, tk, label, I, k, T, j + 1)
    end
    pop!(T)
  end
end

"Σ λ over all kept subsets of I (sizes 1..maxorder)."
function lambda_sum(m::LatticeModel, tk, label, I, T = UInt64[], start = 1)
  s = 0.0
  for j in start:length(I)
    push!(T, I[j])
    key = setkey(tk, label, T)
    s += get(m.lambda, key, 0.0)
    length(T) < m.maxorder && key in m.supported && (s += lambda_sum(m, tk, label, I, T, j + 1))
    pop!(T)
  end
  s
end

"Σ λ over the proper, nonempty subsets of T."
function lower_sum(m::LatticeModel, tk, label, T)
  k = length(T)
  s = 0.0
  for mask in 1:(1<<k)-2
    U = T[[(mask >> (j - 1)) & 1 == 1 for j in 1:k]]
    s += get(m.lambda, setkey(tk, label, U), 0.0)
  end
  s
end

"Log-odds that a candidate with these (sorted, unique) items is chosen."
lattice_logodds(m::LatticeModel, tk::UInt64, label::UInt32, I::Vector{UInt64}) =
  log(priorodds(m, lpriorkey(tk, label))) + lambda_sum(m, tk, label, I)

"""
Build a LatticeModel from events recorded in `dir`.  Defaults follow Phase N:
minsupport = TUPLE_COUNT_THRESHOLD (2), diffthreshold = DIFF_THRESHOLD (3.0),
wmothreshold = WMO_THRESHOLD (1e-3).
"""
function LatticeModel(dir::AbstractString; maxorder = 3, minsupport = 2, diffthreshold = 3.0, wmothreshold = 1e-3,
  verbose = true)
  recs, items = load_events(dir)
  m = LatticeModel(Dict{UInt64,Float64}(), Set{UInt64}(), Dict{UInt64,Tuple{Float64,Float64}}(), maxorder)
  for r in recs
    for pk in unique((lpriorkey(r.typekey, UInt32(0)), lpriorkey(r.typekey, r.label)))
      (c, oc) = get(m.prior, pk, (0.0, 0.0))
      m.prior[pk] = r.chosen == 1 ? (c + 1, oc) : (c, oc + 1)
    end
  end
  for k in 1:maxorder
    counts = Dict{UInt64,Tuple{Float64,Float64}}()
    members = Dict{UInt64,Tuple{UInt64,UInt32,Vector{UInt64}}}()
    for r in recs
      I = sort!(unique(filter(!=(ABSENT), items[r.start:r.start+r.n-1])))
      foreach_kset(m, r.typekey, r.label, I, k) do T, key
        (c1, c2) = get(counts, key, (0.0, 0.0))
        counts[key] = r.chosen == 1 ? (c1 + 1, c2) : (c1, c2 + 1)
        haskey(members, key) || (members[key] = (r.typekey, r.label, copy(T)))
      end
    end
    nsupported = nkept = 0
    for (key, (c1, c2)) in counts
      c1 + c2 >= minsupport || continue
      push!(m.supported, key)
      nsupported += 1
      tk, label, T = members[key]
      e = priorodds(m, lpriorkey(tk, label)) * exp(lower_sum(m, tk, label, T))
      o = smoothOdds(c1, c2, e)
      λ = log(o) - log(e)
      p = o / (1 + o)
      if abs(λ) >= diffthreshold || p * abs(λ) >= wmothreshold
        m.lambda[key] = λ
        nkept += 1
      end
    end
    verbose && @info "order $k: $(length(counts)) sets counted, $nsupported supported, $nkept kept"
  end
  m
end

function score(m::LatticeModel, st::State, key::AgendaKey, c::UInt32)
  d = decisiondef(st, key)
  lattice_logodds(m, typekey(key[1], key[2]), d.labeled ? c : UInt32(0),
    sort!(unique(filter(!=(ABSENT), d.items(st, key[3], c)))))
end

waitscore(m::LatticeModel, st::State, key::AgendaKey) = log(priorodds(m, lpriorkey(typekey(key[1], key[2]), UInt32(0))))
