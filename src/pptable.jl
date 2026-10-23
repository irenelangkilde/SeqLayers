# pptable.jl  --  prepositional-attachment preferences from a large parsed corpus, as attachment features
#
# experiments/build_pptable.py counts, in DepCC (collapsed Stanford dependencies), how often each head
# word, and each finest-level word class (clusters/classes5), takes a PP with each preposition
# (head --prep_P--> noun), per occurrence.  For the attachment of a token t that has a `case` child P
# (e.g. "internet" with "in") to a candidate head c, two items: the candidate's class rate for P and
# its word's rate for P, per 100 occurrences, bucketed in doublings.  Absent when t has no case child
# (yet), P or the word isn't in the table, or c is the root.  The model learns from the training
# records how much these buckets tell.

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

"The two PP items for attaching t to candidate c (class rate, word rate), absent when unknown."
function pp_items(tab::PPTable, st, t, c)
  p = c == NONE ? nothing : case_prep(st, t)
  (p === nothing || !haskey(tab.base, p)) && return UInt64[ABSENT, ABSENT]
  b = tab.base[p]
  w = lowercase(tokstr(st, Int(c)))
  cl = get(tab.formclass, w, Int32(0))
  vc = cl == 0 || !haskey(tab.classocc, cl) ? PENDING : pp_bucket(get(tab.classpp, (cl, p), 0.0), tab.classocc[cl], b)
  vf = haskey(tab.formocc, w) ? pp_bucket(get(tab.formpp, (w, p), 0.0), tab.formocc[w], b) : PENDING
  UInt64[item(vc, (PP_TAG, 1, vc)), item(vf, (PP_TAG, 2, vf))]
end
