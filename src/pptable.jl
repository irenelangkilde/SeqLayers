# pptable.jl  --  prepositional-attachment preferences from a large parsed corpus, as attachment features
#
# experiments/build_pptable.py counts, in DepCC (collapsed Stanford dependencies), how often each head
# word, and each finest-level word class (clusters/classes5), takes a PP with each preposition
# (head --prep_P--> noun), per occurrence.  For the attachment of a token t that has a `case` child P
# (e.g. "internet" with "in") to a candidate head c, two items: the candidate's class rate for P and
# its word's rate for P, per 100 occurrences, bucketed in doublings.  Absent when t has no case child
# (yet), P or the word isn't in the table, or c is the root.  The model learns from the training
# records how much these buckets tell.
#
# Relative mode (pprel): the same two items, but each says how the candidate's rate compares with the
# best rate among the token's other candidates (log2 ratio, bucketed), so the value is directly
# about which head prefers the preposition here rather than how often the candidate takes it at all.

"Preposition attachment counts by head class and head word (see build_pptable.py)."
struct PPTable
  base::Dict{String,Float64}                     # P -> PPs per token, over the corpus
  classocc::Dict{Int32,Float64}
  classpp::Dict{Tuple{Int32,String},Float64}
  formocc::Dict{String,Float64}
  formpp::Dict{Tuple{String,String},Float64}
  formclass::Dict{String,Int32}                  # word -> its finest-level class
end

"Read the table (build_pptable.py) and the word classes it was aggregated over (wordclasses.tsv)."
function load_pptable(file::AbstractString, classes::AbstractString)
  isdir(classes) && (classes = joinpath(classes, "wordclasses.tsv"))
  formclass = Dict{String,Int32}()
  for (i, l) in enumerate(eachline(classes))
    i == 1 && continue
    f = split(l, '\t')
    formclass[String(f[1])] = parse(Int32, f[3])
  end
  total, base = 0.0, Dict{String,Float64}()
  classocc, classpp = Dict{Int32,Float64}(), Dict{Tuple{Int32,String},Float64}()
  formocc, formpp = Dict{String,Float64}(), Dict{Tuple{String,String},Float64}()
  for l in eachline(file)
    f = split(l, '\t')
    if f[1] == "#total"
      total = parse(Float64, f[2])
    elseif f[1] == "base"
      base[String(f[2])] = parse(Float64, f[3])
    elseif f[1] == "class" || f[1] == "form"
      n = parse(Float64, f[3])
      for j in 4:2:length(f)-1
        p, k = String(f[j]), parse(Float64, f[j+1])
        f[1] == "class" ? (classpp[(parse(Int32, f[2]), p)] = k) : (formpp[(String(f[2]), p)] = k)
      end
      f[1] == "class" ? (classocc[parse(Int32, f[2])] = n) : (formocc[String(f[2])] = n)
    end
  end
  total > 0 || error("$file: no #total line")
  foreach(p -> (base[p] /= total), collect(keys(base)))
  PPTable(base, classocc, classpp, formocc, formpp, formclass)
end

const PP_TAG = hash("token.attach.pp")

"""
Bucket 1..8 of the PP rate per 100 occurrences, in doublings: below 1, 1-2, 2-4, 4-8, 8-16, 16-32,
32-64, 64 and up; the rate smoothed toward the corpus-wide rate `base` with `k` pseudo-occurrences.
"""
function pp_bucket(npp, nocc, base; k = 20.0)
  r = 100 * (npp + k * base) / (nocc + k)
  UInt32(clamp(floor(Int, log2(max(r, 1e-9))) + 2, 1, 8))
end

"The preposition of token t: the form of its (first) committed `case` child, or nothing."
function case_prep(st, t)
  toks = st.layers[:token]
  cc = deprelcode("case")
  for u in children_index(st)[t]
    toks[:deprel, u] == cc && return lowercase(tokstr(st, u))
  end
  nothing
end

"Smoothed PPs with p per 100 occurrences (as in pp_bucket) of word w's class (level 1) or w (level 2); nothing if unknown."
function pp_rate(tab::PPTable, w, p, level; k = 20.0)
  b = tab.base[p]
  if level == 1
    cl = get(tab.formclass, w, Int32(0))
    (cl == 0 || !haskey(tab.classocc, cl)) && return nothing
    npp, nocc = get(tab.classpp, (cl, p), 0.0), tab.classocc[cl]
  else
    haskey(tab.formocc, w) || return nothing
    npp, nocc = get(tab.formpp, (w, p), 0.0), tab.formocc[w]
  end
  100 * (npp + k * b) / (nocc + k)
end

"""
Bucket 1..9 of log2(r / best), r a candidate's rate and best the highest among the other candidates:
1 below 1/8 of it, 2..4 the halvings up to 1/1, 5..8 one to sixteen times (5: 1-2x, ..., 8: 8x and
up), 9 when no other candidate has a rate.
"""
pp_relbucket(r, best) = best === nothing ? UInt32(9) : UInt32(clamp(floor(Int, log2(r / best)) + 5, 1, 8))

"""
The two PP items for attaching t to candidate c (class rate, word rate), absent when unknown.
`cands` (relative mode): the token's candidates; each item then compares c's rate with the best of
the other candidates' (pp_relbucket) instead of bucketing it alone.
"""
function pp_items(tab::PPTable, st, t, c; cands = nothing)
  p = c == NONE ? nothing : case_prep(st, t)
  (p === nothing || !haskey(tab.base, p)) && return UInt64[ABSENT, ABSENT]
  w = lowercase(tokstr(st, Int(c)))
  out = UInt64[]
  for level in 1:2
    r = pp_rate(tab, w, p, level)
    v = if r === nothing
      PENDING
    elseif cands === nothing
      pp_bucket(r / 100, 1.0, 0.0; k = 0.0)
    else
      best = nothing
      for h in cands
        (h == c || h == NONE) && continue
        rh = pp_rate(tab, lowercase(tokstr(st, Int(h))), p, level)
        rh === nothing || (best = best === nothing ? rh : max(best, rh))
      end
      pp_relbucket(r, best)
    end
    push!(out, item(v, (PP_TAG, level + (cands === nothing ? 0 : 2), v)))
  end
  out
end
