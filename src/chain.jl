# chain.jl  --  greedy chains of conditioning features, calculated from joint counts
#
# For a decision, start with no conditioning features and each candidate at its prior odds.
# Repeatedly add the feature slot (context path, head form, distance, ...) chosen by the
# selection criterion, re-estimating each candidate's odds from the *joint* counts of the
# whole conditioning set:
#
#     o_c(C ∪ {s}) = smoothOdds(c1, c2, o_c(C))        c1/c2: chosen/unchosen training records
#                                                      containing all of C ∪ {s} for candidate c
#
# Because each feature is judged given the ones already chosen, a redundant feature barely
# changes the counts and adds ~nothing -- no evidence is counted twice.  Stop when the best
# feature's criterion falls below `tau` (or counts fall below `mincount`).
#
# Selection criteria (x = candidates' odds before adding s, y = after; K = number of candidates):
#   :maxdev         per candidate: its own chain, maximizing |log y_c − log x_c|
#   :avglogodds     mean_c |log y_c − log x_c|
#   :clr_mean_abs   mean absolute clr change, (1/K) Σ |Δ_c|,    Δ = clr(log y) − clr(log x)
#   :clr_rms        RMS-normalized Aitchison distance, √((1/K) Σ Δ_c²)
# (clr of the log odds vector: subtract its mean.)  With K = 2 the two clr criteria coincide.
#
# Target mode: if a step changes the top two candidates or their order, the chain restarts
# from nothing with a target odds profile, now picking the feature that moves the odds closest
# to the target (same distance family), stopping when no feature gets at least `tau` closer.
# A later change of the top two to an order different from the target's restarts again, up to
# `maxrestarts`.  The target is (`restart`):
#   :immediate   the odds profile right after the step that changed the order
#   :exhaust     the profile reached by continuing to add features -- ignoring `tau` -- until
#                none has `mincount` joint records left
# With :maxdev the per-candidate chains run in step (each candidate adds its own best feature
# per round) so the ranking can be checked after each round, and each candidate approaches its
# own target odds.
#
# Leftovers (features never added to the chain):
#   :none       ignore them
#   :marginal   add each one's single-feature correction relative to the prior
#   :backoff    add each one's correction conditioned on the longest prefix of the chain that
#               still has `mincount` joint records with it (empty prefix = :marginal)
#   :greedy     each leftover X gets its own conditioning context, grown greedily from the chain's
#               features and the leftovers already included: at each step the available feature
#               f that most changes X's correction (`greedyby` = :correction: |corr(S∪f) − corr(S)|,
#               corr(S) = log o(S∪X) − log o(S)) -- or most moves the context's own odds (:odds) --
#               among those keeping S∪f∪X supported; until none does.  X is included (its
#               correction added, and itself available as context for later leftovers) only if
#               its context has at least `minctx` features.  Leftovers are taken in slot order.
#               Speed-ups (approximate): a context stops growing at `maxctx` features; a leftover
#               whose relevance -- the largest of |λ_X| and |λ(a, X)| over the chain's features a,
#               for the candidate's label -- is below `relevance` is skipped outright.
#   :backoff1   the same as :backoff, but never back off to the empty prefix: a leftover without joint
#               support with at least the chain's first feature is dropped, since a correction
#               relative to the prior would re-count evidence the chain already used.  (With an
#               empty chain it behaves like :marginal.)
#
# Joint tables (`jointmin` > 0): at build time, counts per label are precomputed for every
# "frequent" item (at least `jointmin` records) and every co-occurring pair of frequent items.
# A chain step whose set is such an item or pair takes its counts from the tables without
# intersecting record lists; the set's list is built only if the chain goes deeper.  Exact:
# scores are the same as without the tables.  Intersections with a rare item stay on the fly,
# since they're cheap (binary search of the long list).
#
# Screening (`screen` > 0), treating items as absent at decode time so chains never pay for them:
#   - an item with fewer than `mincount` records (exact: no set containing it can be supported);
#   - a frequent item whose own correction and every stored pair's correction are below `screen`.
#     As in the lattice, a pair's correction is λ = log smoothOdds(c1, c2, e) − log e with e what
#     its two items predict (prior × exp(λ_a + λ_b)); so an item that only matters together with
#     another (a dependent's own tag, for attachment) survives through its pairs.  Labeled
#     decisions combine per-label λ by `screenby`: :maxdev (largest |λ|) or :clr_rms (Aitchison).
# Items that are neither rare nor frequent are kept.  Needs the joint tables.
#
# Records: a feature value's sorted list of records (its posting) is stored only if it holds fewer
# than `keepfrac` of its decision type's records.  A chain prefix's records are a view: a stored
# base list (a posting, a list built earlier, or all records of the type) plus filter items its
# records must also contain.  A prefix gets a list of its own once its count falls below
# `keepfrac` of its base's length; so a near-universal value (tense=0) never stores one, and a
# view never scans more than 1/keepfrac times the records it holds.  All of a chain step's trial
# items are counted in one pass over the view's base, checking the filter in the same pass -- or,
# when every trial item is rare, by intersecting their postings.  For speed, each item gets a
# small integer id at build time and records are scanned as ids against marker arrays.  Since a
# decision type's records all have the same layout, each item sits at a fixed position (its slot);
# a pass then reads only the trial and filter items' positions of each record, not all of its
# items (`usepos`; an item seen at more than one position falls back to reading everything).  Entries
# for sets already seen are cached.  None of this changes any count, so scores are the same for
# any `keepfrac`.
#
# Climbing (`m.climb`): an item x given rings -- nested sets of items of the same slot, innermost
# first (e.g. a word, then its nearest neighbours, its class, groups of classes) -- doesn't count
# "slot = x" in a chain: at each step (and leftover depth) it counts "slot ∈ ring L of x" for the
# finest L whose records support the candidate's label (≥ mincount), so the chain uses the most
# specific ring the data allow there.  Every ring level is counted in the same scan (a record's
# value lies in one innermost ring), and the chosen level's records are kept from that scan.
# The rings themselves come from outside (see climb.jl).  As a leftover (`climbleft`), a
# climbing slot takes :deepest -- the finest ring supported at the deepest prefix that supports
# any (the usual leftover rule) -- or :specific -- the finest ring supported at any depth, from
# the deepest prefix supporting that ring (specificity before depth) -- or :none (no correction).
#
# Grouped decisions (`group`): a labeled decision is recorded as one record per candidate label,
# all with the same items (the context).  With grouping, consecutive records of one decision become
# one record holding its gold label and its label set; a set's "not chosen" count for label l is
# then the number of matching decisions whose label set holds l, less those whose gold label is l.
# Exact (every count is unchanged); attachment's records differ in their items, so they stay one
# per candidate.  Record numbers then refer to the grouped records.
#
# Relevance gates (gate > 0): which slots a decision considers at all, learned once from the
# training records, so an irrelevant feature costs nothing when deciding.  For each unlabeled
# decision type with a fixed record layout: a slot's average effect (mean |log-odds change| from
# knowing its value, over the records holding it, supported values only) and its effect among the
# records holding each value v of a "gate" slot (one with at most MAXGATEVALS values: tags,
# distance, direction, ...).  A slot is admitted when its average effect reaches `gate`, or when a
# gate value present in the record makes its effect reach `gate`.  The admitted slots are all the
# chain and the leftovers ever look at (one rule for both).  Labeled types: every slot admitted.
#
# Persistent counts: `pcounts` holds counts for feature sets computed in earlier runs (they
# depend only on the training events), consulted before any scan.  On disk they are packed:
# sorted keys and flat label/count arrays (save_counts / load_counts), memory-mapped so every
# worker shares one copy, and looked up by binary search; a Dict works too (tests).  With
# `keepcounts`, every exactly counted set whose counting read at least `keepmin` records (cheaper
# ones are not worth storing; not climbing ones, which depend on rings and label) is also added
# to `newcounts`, for the caller to save.
#
# Candidate pool (`poolm` = m > 0; :maxdev chains): at build time each decision type gets an
# ordering of entries -- slots, and slot pairs whose mean total effect exceeds their stronger
# slot's by at least `poolgain` (so redundant pairs, which add nothing over their stronger value,
# get no entry; optionally also mean |λ| ≥ `poollam`) -- by their mean total effect on the outcome,
# |log o − log prior|, over a sample of `poolsample` records.  With `pairscore` = :min, a pair is
# scored -- in the ordering and in each step -- by the least of its own effect and its two slots'
# effects, so it ranks no higher than its weaker part; with :increment, by its excess over its
# stronger slot, effect(AB) − max(effect(A), effect(B)) (= the least of effect(AB) and its increments
# over each slot), so only interactions score well.  Ties go to the lower order (single first).
# `pairfilter`: a supported pair removes its slots from the rest of the step (else they compete).  A chain step walks that ordering, skipping entries with a slot already in the chain
# (or an absent value, or found unsupported earlier in this chain: support only shrinks), and
# evaluates entries until m of them are supported; a supported pair removes its two slots from
# the rest of this step's walk.  The best of those m is added if it moves the odds by ≥ tau; a
# pair adds both values in one step.  Leftovers are unaffected.
#
# Sampling (`sample` = c > 0; approximate): a scan of a view reads at most n = K·c / p₂ of its
# base's records, evenly spaced, and scales the counts up; p₂ is the probability (o/(1+o)) of the
# decision's second-ranked candidate at that point, and K the number of labels (1 for unlabeled
# decisions: labeled ones have a record per label, so only 1/K of the records are any one label's).
# So the top two candidates see about c chosen records or more (relative error ≈ 1/√c), and
# lower-ranked ones, which barely affect the decision, get less precision.  A trial item whose
# estimated count is under 10·mincount is counted exactly, so small counts (support, smoothing)
# stay exact.  With sampling, :maxdev chains run in rounds so
# the ranking is known at each step.  Cached entries remember how many records they read, and are
# reused only where that's enough.

struct ChainOpts
  select::Symbol
  tau::Float64
  mincount::Int
  leftovers::Symbol
  target::Bool
  maxrestarts::Int
  restart::Symbol
  screen::Float64
  screenby::Symbol
  sample::Float64
  climbleft::Symbol
  minctx::Int
  greedyby::Symbol
  maxctx::Int
  relevance::Float64
  poolm::Int
  pairscore::Symbol
  pairfilter::Bool
  gate::Float64
  redundant::Bool                                 # greedy: redundant features join the leftovers' conditioning pool
  ctxchain::Bool                                  # greedy: a leftover's context has exactly as many features as the chain
  stepscore::Symbol                               # :maxdev step score: :abs |Δ|, :signed Δ (raise the odds most), :maxmin, :maxmin2, :posneg / :posnegprior (raise the odds most; when no step raises them by tau, the one lowering them most below the chain / the prior); :bipolar (first step :abs, then the previous step's direction first: :posneg after a raise, :negpos after a lowering); :pmi, :mi, :oddspmi, :moi, :logor -- among the steps moving the odds by tau, the most associated with the candidate's choice (see step_crit)
  top2::Bool                                      # :maxdev: only the two leading candidates' chains advance (run_top2)
  contest::Bool                                   # re-decide the top two by a contest chain on their odds ratio (contest!)
  pickfewest::Int                                 # >0: of the k best-scoring features passing tau, take the one with fewest records
  adaptctx::Bool                                  # greedy: minimum context max(1, min(minctx, chain + redundant pool))
  pairsteps::Bool                                 # :maxdev chains may add a precomputed pair (joint tables) in one step
  pairsfirst::Bool                                # with pairsteps: a feature in any precomputed pair joins only via a pair
  poolsigma::Float64                              # soft pool: generalization g of a chain form weighted exp(-d_g/sigma) (0: off)
  poolmode::Symbol                                # :whole (rebuild the chain with g for w; blend) | :local (at w's step; scale)
  poolmarginal::Symbol                            # pool features weak on their own too (:off) | send them to the leftovers (:leftover) | drop them (:drop)
  poolkeepgen::Bool                               # poolmarginal with poolsigma: exempt generalizations of chain forms
  poolscale::Symbol                               # soft pool distance: :logodds (d_g) | :z (d_g / SE of the chain's final log odds)
  ctxfromchain::Bool                              # greedy: a leftover's context starts as the chain (as far as supported), then extends
  ctxsupport::Int                                 # greedy: a context step needs context + leftover with at least this many records (0: mincount)
  leftsmooth::Symbol                              # greedy: o(x|S∪E) smoothed toward o(x|S) (:odds) | by E's rates within x and ¬x (:evidence)
  ctxmode::Symbol                                 # greedy contexts: :greedy | :backoff / :limited (generalize the chain) | :cover (greedy, must cover the chain) | :accum (leftover chain)
  ctxattach::Bool                                 # ctxmode for the attachment decision only (labels keep :greedy)
  cidrop::Float64                                 # accum: drop a context feature B only if some A kept has D(A→B) < cidrop (0: always may)
  pairorder::Symbol                               # a chosen pair's members enter the chain in slot order (:slot) or larger single effect first (:maxdev)
  accumwin::Int                                   # accum: a leftover's context minimizes |correction| over this many further generalization steps (0: the first supported)
  accumreset::Bool                                # accum: each leftover generalizes from the most specific context (generalizations not carried forward)
  greedymin::Bool                                 # greedy: a leftover's context is the growth prefix (>= minctx) with the smallest |correction|
  greedysel::Bool                                 # greedy: leftovers taken in rounds, largest |correction| first (else in slot order)
  driftsigma::Float64                             # accum: a correction weighted exp(-d/sigma), d = |log o(x|context) - running estimate| (0: off)
  drifttau::Float64                               # accum: a leftover whose context has d > tau is left out (0: off)
  driftstep::Float64                              # accum: a correction weighted driftstep^(generalization / drop steps of its context) (0: off)
  distback::Bool                                  # leftover contexts: the attachment's distance generalizes to its direction
  shrink::Float64                                 # > 0: estimates shrunk toward the parent's with a Beta prior of this many pseudo-records (else one-count smoothOdds)
  shrinkleft::Float64                             # > 0: the same, for the leftovers' corrections only (chain steps keep their rule)
  leftmin2::Int                                   # > 0: a leftover counts only with this many chosen and this many not-chosen records (with its context)
  ciorder::Symbol                                 # accum: the walk generalizes the context feature least dependent (ci table) first: :off | :slot | :value
  telescope::Bool                                 # accum: a leftover is conditioned on its ladder's nearest counted coarser rung; coarser rungs go first
  branches::Int                                   # accum: a leftover needing generalization gets contexts along this many first steps (1: the best only)
  combine::Symbol                                 # accum branches: :avg (inverse variance) | :ie (inclusion-exclusion) | :mix
  smoothd::Float64                                # smoothing: the estimate moves d = smoothd + smootha * n counts toward its parent (1, 0: one count)
  smootha::Float64
  accumby::Symbol                                 # accum walk step: :odds (closest to the running estimate, shared) | :dep (per leftover: least dependence lost) | :depodds (both, in log odds)
  depweight::Bool                                 # leftovers: correction times 1 - max ρ(X, a) over counted features a outside X's context
  smoothn::Symbol                                 # n in d: the set's own records (:set) or its parent's (:parent: chain steps and accum leftovers)
  depmeasure::Symbol                              # ρ for greedyby=dep / accumby=dep: :yule (|Yule's Q|) | :pmi (log p(e,a) / p(e)p(a)) | :pmipos (max(pmi, 0))
  mindevtau::Float64                              # greedyby=mindev: a leftover whose diff falls below this is explained away (dropped)
  mindevstop::Symbol                              # greedyby=mindev: :localmin | :exhaust | :pathmin (see greedy_context)
  explainpool::Bool                               # mindev (greedy / accum): an explained-away leftover joins the context pool (correction 0) instead of being discarded
  chainshrink::Float64                            # > 0: the chain's log odds from Beta-shrunk step sizes (k pseudo-records); selection unchanged
  chainshrinkto::Symbol                           # chainshrink target: :prev (previous step's odds) | :gen (the step with the feature's generalization) | :prior (the final prefix toward the prior)
  lookw::Symbol                                   # accumby=look1*: supported moves by :dw (ΔW) | :int (interaction) | :min | :max of the two
  dualchain::Symbol                               # two chains per candidate, one for positive evidence, one for negative: :off | :marginal (eligibility by a feature's marginal sign) | :inchain (by its sign in the chain)
  dualcombine::Symbol                             # dualchain score: :sum | :maxabs | :weighted (negative chain times dualw) | :sumdedup (sum without the duplicate copies)
  dualw::Float64                                  # dualcombine=:weighted: the negative chain's weight
  dualturn::Symbol                                # dualchain, a feature whose sign turns: :both (it joins both chains) | :refine (turned in the picking chain only: it goes to the other chain only; turned in both: both)
  dualleft::Bool                                  # dualchain: leftovers (context: both chains' features) on top of the combined score
  leftwback::Float64                              # > 0: accum leftover corrections times exp(-(l_i - l_0)^2 / leftwback), l_i the chosen context's log odds, l_0 the starting context's
  leftwovl::Bool                                  # accum leftover corrections times 1 - their largest overlap with a counted feature outside the context, (Δe + Δa - Δea) / min(|Δe|, |Δa|) over all records
  leftwcomb::Symbol                               # both weights on: :prod | :min | :max
  wselcomb::Symbol                                # accumby=:wsel: each leftover's backoff move by the largest :min or :sum of w_back (N = leftwback, 1 if 0) and w_ovl
  wselmove::Symbol                                # accumby=:wsel: moves by :weights alone, or :wcorr (among moves supporting the leftover the largest |w * corr|; else the weights)
  wselweight::Bool                                # accumby=:wsel: the leftover's correction times w of its chosen context (min, or sum / 2)
  tuplek::Int                                     # > 0: a :maxdev chain's first step adds the best tuple (triple, else pair, else single) of its tuplek largest single steps (see tuple_step!)
  interleave::Bool                                # ctxmode :accum: chain steps and leftovers in one maxdev loop (see accum_leftovers)
  candmemo::Bool                                  # per-candidate score memo (see cand_scores): only candidates whose own items changed are re-scored; exact
  accumpick::Symbol                               # ctxmode :accum: each round take the leftover with the largest :abs correction, or the largest :signed one (the biggest raise)
  accumfloor::Bool                               # ctxmode :accum: the backoff never drops a context feature once only minctx are left (user 2026-11-04)
  forkcap::Int                                    # > 0 (:maxdev): when the largest |step| lowers the odds and a different feature raises them most (by tau), the chain forks; at most this many chains per candidate (see fork_chains)
  forkcomb::Symbol                                # forkcap: a candidate's score from its chains, :max or :avg
  forkleft::Symbol                                # forkcap: :each (every chain gets its leftovers; compared on chain + leftovers) or :chosen (compared on the chain alone; the best gets leftovers; :max only)
  accumstop::Float64                              # > 0 (ctxmode :accum): stop taking leftovers once the best one's backed-off context drifts more than this from T (the rest are discarded)
  accumgreedy::Float64                            # > 0 (ctxmode :accum): a leftover whose backed-off context's log odds drift more than this from T gets a greedy (greedyby :dep) context instead
  tuplemin::Float64                               # > 0: the tuple pool is the features whose single step reaches tuplemin (and tau; with tuplek too: the tuplek largest of them)
  posnegonly::Symbol                              # posneg / posnegrestart for :all decisions, :attach (unlabeled decisions) only, or :heads (attachment and head properties) -- the others then use :abs chains
  posnegafter::Bool                               # posnegrestart: pick the candidate to restart by its full score (chain + leftovers), recomputing its leftovers after each restart
  posnegleftneg::Bool                             # posnegafter: also seed with the leftovers whose correction was negative for it (after the chain negatives)
  posnegseed::Symbol                              # posnegrestart's seed: :own (the top candidate's own negative steps) | :oldtop (the negatives of the candidate ranked first after the positive stage, then of the previous top)
  posnegrestart::Bool                             # stepscore=posneg: restart the top candidate's chain seeded with its negative steps, until rank 1 is stable (up to 3)
  stickbreak::Symbol                              # candidates' probabilities: :off (softmax of the log odds) | :renorm (stick-breaking in rank order) | :cond (labeled decisions: each value's odds conditioned on not the higher-ranked values, then stick-breaking)
  lookwback::Float64                              # > 0: the interaction backs off to the pairwise one, |corr(W | a) - corr(W | a')| over all records, where W has fewer than this many records in the thinner context
  lookwsoft::Bool                                 # lookwback: blend by support (w = n / (n + lookwback)) instead of switching
  lookwsmooth::Float64                            # > 0: W's estimates for ΔW / interaction from its likelihood ratio, each rate smoothed by this many records toward W's pooled rate
  chainsel::Float64                               # > 0: selective shrinkage of chain steps toward their generalization, by z = deviation / SE (z0 = this)
end

const LabelCounts = Dict{UInt32,Tuple{Float64,Float64}}   # label -> (chosen, not chosen); estimates when sampled
const NOLIST = Int32[]
const NOCOUNTS = LabelCounts()

"The records of `base` that contain every item of `filt`."
struct RecView
  base::Vector{Int32}
  filt::Vector{UInt64}
end
RecView(base::Vector{Int32}) = RecView(base, UInt64[])
const Entry = Tuple{RecView,LabelCounts}
const NOENTRY = (RecView(NOLIST), NOCOUNTS)

"""
Relevance gates for one decision type (built when `gate` > 0; see Relevance gates in the header):
`effect[s]` is slot s's average effect, `always[s]` whether it passes the gate on that alone, and
`byvalue[id]` the slots that pass when the item with that id (a value of a gate slot) is present.
"""
struct Gates
  nslots::Int
  effect::Vector{Float32}
  always::BitVector
  gateslots::Vector{Int}
  byvalue::Dict{Int32,Vector{Int16}}
  effectby::Dict{Tuple{Int32,Int16},Float32}       # (gate value, slot) -> effect, for diagnostics
  valcount::Dict{Int32,Int}                         # gate value -> its records
end

const MAXGATEVALS = 64                              # slots with at most this many values can be gates
const GATEMIN = 2000                                # a gate value opens slots only with at least this many records
const GATEBLOCK = 14                                # gate effects: halves are alternating blocks of 2^GATEBLOCK records

"""
An item's rings (see Climbing in the header): `order` lists the items of its innermost rings,
nearest first (the item itself first); rings 1..length(cuts) are the prefixes order[1:cuts[L]];
then each of `outer` (each containing the previous rings).
"""
struct Climb
  order::Vector{UInt64}
  cuts::Vector{Int}
  outer::Vector{Vector{UInt64}}
end
nlevels(c::Climb) = length(c.cuts) + length(c.outer)

mutable struct ChainModel <: AbstractModel
  recs::Vector{CandRec}
  items::Vector{UInt64}
  ids::Vector{Int32}                             # items[q] as a small id (0 for ABSENT)
  idof::Dict{UInt64,Int32}
  want::Vector{Int32}                            # id -> trial index during a scan (0 otherwise)
  infilt::BitVector                              # id -> in the scanned view's filter
  typerecs::Dict{UInt64,Vector{Int32}}           # decision type -> all its records
  itemcount::Dict{UInt64,Int}                    # item -> number of records containing it
  itemtype::Dict{UInt64,UInt64}                  # item -> its decision type (items are type-specific)
  postings::Dict{UInt64,Vector{Int32}}           # item -> its records, if fewer than keepfrac of its type's
  keepfrac::Float64
  prior::Dict{UInt64,Tuple{Float64,Float64}}
  opts::ChainOpts
  cache::Dict{UInt64,Tuple{Entry,Int}}           # set key -> ((records, counts per label), records read; EXACT if all)
  cached::Int                                    # list entries built and held by the cache
  maxcached::Int
  memo::Dict{UInt64,Vector{Float64}}             # (type, labels, items) -> final scores
  frequent::Set{UInt64}                          # items with joint tables (see joint tables)
  single::Dict{UInt64,LabelCounts}               # frequent item -> counts per label
  pairs::Dict{UInt64,LabelCounts}                # key of a pair of frequent items -> counts per label
  screened::Set{UInt64}                          # items treated as absent (see screening)
  climb::Dict{UInt64,Climb}                      # items that climb rings (see Climbing)
  vclimb::Dict{UInt64,Tuple{UInt64,UInt32}}      # climbing trial key -> (item, label it's for)
  rtrial::Vector{Int32}                          # id -> climbing trial index during a scan (0 otherwise)
  rlevel::Vector{Int8}                           # id -> its innermost ring level in that trial
  reclen::Float64                                # mean number of items per record
  slotof::Vector{Int16}                          # id -> its position in its records (-1: varies)
  usepos::Bool                                   # passes read only the needed positions
  nlab::Int                                      # labels are 0..nlab-1 over all types (dense tallies use tnlab, per type)
  pool::Dict{UInt64,Vector{Tuple{Int16,Int16}}}  # decision type -> ordered entries (slot, slot or 0)
  lsid::Vector{Int32}                            # record -> its label set (grouped decision), 0 if ungrouped
  lsets::Vector{Vector{UInt32}}                  # label sets of grouped decisions
  pcounts::Any                                   # set key -> counts from earlier runs: PackedCounts or Dict (see Persistent counts)
  newcounts::Dict{UInt64,LabelCounts}            # set key -> counts computed in this run
  keepcounts::Bool
  keepmin::Int                                   # records a count must have read to be kept
  tnlab::Dict{UInt64,Int}                        # decision type -> its labels are 0..tnlab-1 (dense tally rows)
  tls::Dict{UInt64,Vector{Int32}}                # decision type -> its label sets (global ids, in local order)
  lsloc::Vector{Int32}                           # label set -> its index among its type's
  noleft::Dict{UInt64,Set{Int}}                  # decision type -> slots never used as leftovers (chains only)
  gates::Dict{UInt64,Gates}                      # decision type -> relevance gates (see Gates)
  leftalpha::Dict{UInt64,Float64}                # decision type -> weight of the greedy leftovers' corrections (default 1)
  generalize::Dict{UInt64,Dict{Int,Vector{Int}}} # decision type -> form slot -> its generalization slots (soft pool)
  nochain::Dict{UInt64,Set{Int}}                 # decision type -> slots never chosen in chains (nor leftovers): stand-ins only
  ctxtypes::Set{UInt64}                          # decision types that use ctxmode (empty: all; the others keep greedy contexts)
  ci::Dict{UInt64,Matrix{Float32}}               # decision type -> D[A, B]: B's effect on o(x) given A (ci_table)
  civ::Dict{UInt64,Dict{Tuple{Int16,UInt64,Int16},Float32}}   # decision type -> (A, A's value a, B) -> B's effect on o(x) given A = a (ci_values)
  cmemo::Dict{Tuple{UInt64,Int},Float64}         # (candmemo) (candidate key, part) -> its score: 1 chain + leftovers, 2 after a restart (NaN: none), 3 forked
end

const MAXMEMO = 2_000_000
const EXACT = typemax(Int)


"Order-independent key of a feature set, built one item at a time (the empty set is 0)."
extendkey(key::UInt64, x::UInt64) = key + hash(x, 0x51ed270b27a1f6c3)

function ChainModel(dir::AbstractString; select = :clr_rms, tau = 0.1, mincount = 5, leftovers = :backoff,
  target = false, maxrestarts = 2, restart = :immediate, maxcached = 50_000_000, keepfrac = 0.1,
  jointmin = 0, screen = 0.0, screenby = :maxdev, sample = 0.0, climbleft = :deepest, usepos = true, minctx = 1, greedyby = :correction,
  maxctx = typemax(Int), relevance = 0.0, poolm = 0, poolsample = 20_000, poollam = 0.0, poolgain = 0.1,
  pairscore = :total, pairfilter = true, group = false, gate = 0.0, redundant = false, ctxsize = :min,
  stepscore = :abs, lockstep = :none, contest = false, pickfewest = 0, pairsteps = false, pairsfirst = false, poolsigma = 0.0,
  poolmode = :whole, poolmarginal = :off, poolkeepgen = false, poolscale = :logodds, ctxfromchain = false, ctxsupport = 0, leftsmooth = :odds, ctxmode = :greedy, ctxattach = false, cidrop = 0.0, pairorder = :slot, accumwin = 0, accumreset = false, greedymin = false, greedysel = false, driftsigma = 0.0, drifttau = 0.0, driftstep = 0.0, distback = false, shrink = 0.0, shrinkleft = 0.0, leftmin2 = 0, ciorder = :off, telescope = false, branches = 1, combine = :avg, smoothd = 1.0, smootha = 0.0, smoothn = :set, gatemincount = mincount, accumby = :odds, depweight = false, depmeasure = :yule, mindevtau = 0.1, mindevstop = :localmin, explainpool = false, chainshrink = 0.0, chainshrinkto = :prev, chainsel = 0.0, lookw = :dw, lookwsmooth = 0.0, lookwback = 0.0, lookwsoft = false, stickbreak = :off, posnegrestart = false, posnegseed = :own, dualchain = :off, dualcombine = :sum, dualw = 0.5, dualleft = true, dualturn = :both, posnegafter = false, posnegleftneg = false, posnegonly = :all, leftwback = 0.0, leftwovl = false, leftwcomb = :prod, wselcomb = :sum, wselmove = :weights, wselweight = false, tuplek = 0, tuplemin = 0.0, interleave = false, accumgreedy = 0.0, accumstop = 0.0, forkcap = 0, forkcomb = :max, forkleft = :each, accumpick = :abs, accumfloor = false, candmemo = false)
  select in (:maxdev, :avglogodds, :clr_mean_abs, :clr_rms, :maxpos, :maxabsscore, :maxposneg) || error("unknown select $select")
  leftovers in (:none, :marginal, :backoff, :backoff1, :greedy) || error("unknown leftovers $leftovers")
  greedyby in (:correction, :odds, :dep, :depcorr, :mindev) || error("unknown greedyby $greedyby")
  mindevstop in (:localmin, :exhaust, :pathmin) || error("unknown mindevstop $mindevstop")
  dualchain in (:off, :marginal, :inchain) || error("unknown dualchain $dualchain (use :off, :marginal or :inchain)")
  dualturn in (:both, :refine) || error("unknown dualturn $dualturn (use :both or :refine)")
  dualcombine in (:sum, :maxabs, :weighted, :sumdedup) || error("unknown dualcombine $dualcombine (use :sum, :maxabs, :weighted or :sumdedup)")
  posnegseed in (:own, :oldtop, :reorder, :reorderabs) || error("unknown posnegseed $posnegseed (use :own, :oldtop, :reorder or :reorderabs)")
  wselcomb in (:min, :sum) || error("unknown wselcomb $wselcomb (use :min or :sum)")
  interleave && (leftovers != :greedy || ctxmode != :accum || select != :maxdev || pairsteps || pickfewest > 0 || target || lockstep != :none || sample > 0 || dualchain != :off) &&
    error("interleave: needs select = :maxdev, leftovers = :greedy, ctxmode = :accum, and no pairsteps, pickfewest, target, lockstep, sample or dualchain")
  accumgreedy > 0 && (leftovers != :greedy || ctxmode != :accum) && error("accumgreedy: needs leftovers = :greedy, ctxmode = :accum")
  accumpick in (:abs, :signed) || error("unknown accumpick $accumpick (use :abs or :signed)")
  forkcomb in (:max, :avg) || error("unknown forkcomb $forkcomb (use :max or :avg)")
  forkleft in (:each, :chosen) || error("unknown forkleft $forkleft (use :each or :chosen)")
  forkcap > 0 && (select != :maxdev || stepscore != :abs || posnegrestart || dualchain != :off || interleave || target || lockstep != :none || sample > 0 || pairsteps || pickfewest > 0 || poolm > 0) &&
    error("forkcap: needs select = :maxdev with stepscore = :abs, and no posneg restart, dualchain, interleave, target, lockstep, sample, pairsteps, pickfewest or pool")
  forkcap > 0 && forkleft == :chosen && forkcomb != :max && error("forkleft = :chosen needs forkcomb = :max")
  accumstop > 0 && (leftovers != :greedy || ctxmode != :accum) && error("accumstop: needs leftovers = :greedy, ctxmode = :accum")
  interleave && !(stepscore in (:abs, :posneg)) && error("interleave: stepscore :abs or :posneg")
  wselmove in (:weights, :wcorr) || error("unknown wselmove $wselmove (use :weights or :wcorr)")
  leftwcomb in (:prod, :min, :max) || error("unknown leftwcomb $leftwcomb (use :prod, :min or :max)")
  posnegonly in (:all, :attach, :heads) || error("unknown posnegonly $posnegonly (use :all, :attach or :heads)")
  stickbreak in (:off, :renorm, :cond) || error("unknown stickbreak $stickbreak (use :off, :renorm or :cond)")
  lookw in (:dw, :int, :min, :max) || error("unknown lookw $lookw (use :dw, :int, :min or :max)")
  chainshrinkto in (:prev, :gen, :prior) || error("unknown chainshrinkto $chainshrinkto (use :prev, :gen or :prior)")
  depmeasure in DEP_MEASURES || error("unknown depmeasure $depmeasure (use $(join(DEP_MEASURES, ", ")))")
  accumby in (:odds, :dep, :depodds, :marg, :mindev, :mindevsup, :mintw, :mintwc, :bfs, :look1red, :look1ind, :look1nonred, :look1int, :wsel) ||
    error("unknown accumby $accumby (use :odds, :dep, :depodds, :marg, :mindev, :mindevsup, :mintw, :mintwc or :bfs, :look1red, :look1ind, :look1nonred, :look1int, :wsel)")
  ctxsize in (:min, :chain, :adaptive) || error("unknown ctxsize $ctxsize (use :min, :chain or :adaptive)")
  stepscore in (:abs, :signed, :maxmin, :maxmin2, :posneg, :posnegprior, :negpos, :bipolar, CRIT_STEPS...) ||
    error("unknown stepscore $stepscore (use :abs, :signed, :maxmin, :maxmin2, :posneg, :posnegprior, :negpos or :bipolar)")
  lockstep in (:none, :top2) || error("unknown lockstep $lockstep (use :none or :top2)")
  poolmarginal in (:off, :leftover, :drop) || error("unknown poolmarginal $poolmarginal (use :off, :leftover or :drop)")
  pairorder in (:slot, :maxdev) || error("unknown pairorder $pairorder (use :slot or :maxdev)")
  ctxmode in (:greedy, :backoff, :limited, :cover, :accum) || error("unknown ctxmode $ctxmode (use :greedy, :backoff, :limited, :cover or :accum)")
  smoothn in (:set, :parent) || error("unknown smoothn $smoothn (use :set or :parent)")
  combine in (:avg, :ie, :mix) || error("unknown combine $combine (use :avg, :ie or :mix)")
  branches >= 1 || error("branches must be at least 1")
  ciorder in (:off, :slot, :value) || error("unknown ciorder $ciorder (use :off, :slot or :value)")
  leftsmooth in (:odds, :evidence) || error("unknown leftsmooth $leftsmooth (use :odds or :evidence)")
  poolscale in (:logodds, :z) || error("unknown poolscale $poolscale (use :logodds or :z)")
  poolmode in (:whole, :local) || error("unknown poolmode $poolmode (use :whole or :local)")
  restart in (:immediate, :exhaust) || error("unknown restart $restart")
  screenby in (:maxdev, :clr_rms) || error("unknown screenby $screenby")
  climbleft in (:deepest, :specific, :none) || error("unknown climbleft $climbleft")
  0 < keepfrac <= 1 || error("keepfrac must be in (0, 1]")
  pairscore in (:total, :min, :increment, :synergy) || error("unknown pairscore $pairscore")
  screen > 0 && jointmin <= 0 && error("screening needs the joint tables (jointmin > 0)")
  poolm > 0 && group && error("the candidate pool needs group = false")
  target && restart == :exhaust && select != :maxdev && error("restart = :exhaust is implemented for :maxdev only")
  recs, items = load_events(dir)
  prior = Dict{UInt64,Tuple{Float64,Float64}}()
  for r in recs                                    # priors from the records as recorded
    for pk in unique((lpriorkey(r.typekey, UInt32(0)), lpriorkey(r.typekey, r.label)))
      (c, oc) = get(prior, pk, (0.0, 0.0))
      prior[pk] = r.chosen == 1 ? (c + 1, oc) : (c, oc + 1)
    end
  end
  recs, lsid, lsets = group ? group_records(recs, items) : (recs, zeros(Int32, length(recs)), Vector{UInt32}[])
  typerecs = Dict{UInt64,Vector{Int32}}()
  ids, idof = zeros(Int32, length(items)), Dict{UInt64,Int32}()
  counts, types, slots = Int[], UInt64[], Int16[]   # by id
  for (j, r) in enumerate(recs)
    push!(get!(() -> Int32[], typerecs, r.typekey), Int32(j))
    for q in r.start:r.start+r.n-1
      x = items[q]
      x == ABSENT && continue
      pos = Int16(q - r.start + 1)
      id = get!(idof, x) do
        push!(counts, 0); push!(types, r.typekey); push!(slots, pos)
        Int32(length(counts))
      end
      slots[id] == pos || (slots[id] = Int16(-1))
      ids[q] = id
      counts[id] += 1
    end
  end
  itemcount = Dict(x => counts[id] for (x, id) in idof)
  itemtype = Dict(x => types[id] for (x, id) in idof)
  postings = Dict(x => sizehint!(Int32[], k) for (x, k) in itemcount if k < keepfrac * length(typerecs[itemtype[x]]))
  for (j, r) in enumerate(recs), q in r.start:r.start+r.n-1
    L = get(postings, items[q], nothing)          # (ABSENT has no posting)
    L === nothing || push!(L, Int32(j))           # records in order, so lists come out sorted
  end
  m = ChainModel(recs, items, ids, idof, zeros(Int32, length(counts)), falses(length(counts)),
    typerecs, itemcount, itemtype, postings, keepfrac, prior,
    ChainOpts(select, tau, mincount, leftovers, target, maxrestarts, restart, screen, screenby, sample, climbleft,
      minctx, greedyby, maxctx, relevance, poolm, pairscore, pairfilter, gate, redundant, ctxsize == :chain, stepscore,
      lockstep == :top2, contest, pickfewest, ctxsize == :adaptive, pairsteps, pairsfirst, poolsigma, poolmode, poolmarginal, poolkeepgen, poolscale, ctxfromchain, ctxsupport, leftsmooth, ctxmode, ctxattach, cidrop, pairorder, accumwin, accumreset, greedymin, greedysel, driftsigma, drifttau, driftstep, distback, shrink, shrinkleft, leftmin2, ciorder, telescope, branches, combine, smoothd, smootha, accumby, depweight, smoothn, depmeasure, mindevtau, mindevstop, explainpool, chainshrink, chainshrinkto, lookw, dualchain, dualcombine, dualw, dualturn, dualleft, leftwback, leftwovl, leftwcomb, wselcomb, wselmove, wselweight, tuplek, interleave, candmemo, accumpick, accumfloor, forkcap, forkcomb, forkleft, accumstop, accumgreedy, tuplemin, posnegonly, posnegafter, posnegleftneg, posnegseed, posnegrestart, stickbreak, lookwback, lookwsoft, lookwsmooth, chainsel),
    Dict{UInt64,Tuple{Entry,Int}}(), 0, maxcached, Dict{UInt64,Vector{Float64}}(),
    Set{UInt64}(), Dict{UInt64,LabelCounts}(), Dict{UInt64,LabelCounts}(), Set{UInt64}(),
    Dict{UInt64,Climb}(), Dict{UInt64,Tuple{UInt64,UInt32}}(), zeros(Int32, length(counts)), zeros(Int8, length(counts)),
    isempty(recs) ? 1.0 : sum(r -> r.n, recs) / length(recs), slots, usepos,
    isempty(recs) ? 1 : Int(max(maximum(r -> r.label, recs), maximum(maximum, lsets; init = UInt32(0)))) + 1,
    Dict{UInt64,Vector{Tuple{Int16,Int16}}}(), lsid, lsets,
    Dict{UInt64,LabelCounts}(), Dict{UInt64,LabelCounts}(), false, 10_000, Dict{UInt64,Int}(),
    Dict{UInt64,Vector{Int32}}(), zeros(Int32, length(lsets)), Dict{UInt64,Set{Int}}(), Dict{UInt64,Gates}(),
    Dict{UInt64,Float64}(), Dict{UInt64,Dict{Int,Vector{Int}}}(), Dict{UInt64,Set{Int}}(), Set{UInt64}(), Dict{UInt64,Matrix{Float32}}(),
    Dict{UInt64,Dict{Tuple{Int16,UInt64,Int16},Float32}}(), Dict{Tuple{UInt64,Int},Float64}())
  tally_shapes!(m)
  if gate > 0
    t = @elapsed build_gates!(m; mincount = gatemincount)
    @info "ChainModel: relevance gates for $(length(m.gates)) decision types (gate = $gate, $(round(t; digits = 1)) s): " *
      join(("$(sum(g.always)) of $(length(g.always)) slots always, $(length(g.byvalue)) gate values" for g in values(m.gates)), "; ")
  end
  group && @info "ChainModel: grouped decisions into $(length(recs)) records ($(count(!iszero, lsid)) grouped, $(length(lsets)) label sets)"
  if poolm > 0
    t = @elapsed pool_order!(m, poolsample, poollam, poolgain)
    @info "ChainModel: candidate-pool orderings for $(length(m.pool)) decision types, $(sum(length, values(m.pool); init = 0)) entries ($(round(t; digits = 1)) s)"
  end
  if jointmin > 0
    t = @elapsed members = joint_tables!(m, jointmin)
    @info "ChainModel: joint tables for $(length(m.frequent)) frequent items, $(length(m.pairs)) pairs ($(round(t; digits = 1)) s)"
    if screen > 0
      screen_items!(m, members)
      @info "ChainModel: screened $(length(m.screened)) of $(length(itemcount)) items (screen = $screen, $screenby)"
    end
  end
  m
end

"""
Group consecutive records of one decision -- same type, same items, labels increasing -- into one
record with the gold label (chosen = 1; chosen = 0 if the group has none) and the label set's id.
"""
function group_records(recs, items)
  out, lsid = CandRec[], Int32[]
  lsets, lsidx = Vector{UInt32}[], Dict{Tuple{UInt64,Vector{UInt32}},Int32}()   # a label set belongs to one type
  i = 1
  while i <= length(recs)
    r, j = recs[i], i
    labs = UInt32[r.label]
    while j < length(recs)
      s = recs[j+1]
      (s.typekey == r.typekey && s.n == r.n && s.label > recs[j].label &&
        view(items, s.start:s.start+s.n-1) == view(items, r.start:r.start+r.n-1)) || break
      j += 1
      push!(labs, s.label)
    end
    nchosen = count(k -> recs[k].chosen == 1, i:j)
    if j == i || nchosen > 1                       # (more than one gold: not one decision; keep them apart)
      for k in i:j
        push!(out, recs[k]); push!(lsid, 0)
      end
    else
      g = findfirst(k -> recs[k].chosen == 1, i:j)
      id = get!(lsidx, (r.typekey, labs)) do
        push!(lsets, copy(labs))
        Int32(length(lsets))
      end
      push!(out, CandRec(r.typekey, g === nothing ? UInt32(0) : recs[i+g-1].label, g === nothing ? UInt32(0) : UInt32(1), r.start, r.n))
      push!(lsid, id)
    end
    i = j + 1
  end
  out, lsid, lsets
end

"Add record j's counts to `d` (a grouped decision: chosen for its gold label, not chosen for the rest of its set)."
function tally!(m::ChainModel, d::LabelCounts, j)
  r = m.recs[j]
  g = m.lsid[j]
  if g == 0
    (c1, c2) = get(d, r.label, (0.0, 0.0))
    d[r.label] = r.chosen == 1 ? (c1 + 1, c2) : (c1, c2 + 1)
  else
    for l in m.lsets[g]
      (c1, c2) = get(d, l, (0.0, 0.0))
      d[l] = r.chosen == 1 && l == r.label ? (c1 + 1, c2) : (c1, c2 + 1)
    end
  end
end

const NOLS = Int32[]

"""
Per decision type: the dense tally size (labels 0..max) and its label sets.  Sized per type, so a
type with many labels (a fine word-class level) doesn't make every other type's tallies large.
"""
function tally_shapes!(m::ChainModel)
  for (j, r) in enumerate(m.recs)
    m.tnlab[r.typekey] = max(get(m.tnlab, r.typekey, 1), Int(r.label) + 1)
    g = m.lsid[j]
    g == 0 && continue
    m.tnlab[r.typekey] = max(m.tnlab[r.typekey], Int(maximum(m.lsets[g])) + 1)
    if m.lsloc[g] == 0
      v = get!(() -> Int32[], m.tls, r.typekey)
      push!(v, g)
      m.lsloc[g] = length(v)
    end
  end
end

"Dense tally rows and label sets for records of the type of `base`'s records (all one type)."
tallyshape(m::ChainModel, base) =
  isempty(base) ? (1, NOLS) : (tk = m.recs[base[1]].typekey; (m.tnlab[tk], get(m.tls, tk, NOLS)))


"""
Effect of each slot over `recs` (all of one unlabeled type with `n` slots), per gate value: for
gate slot g, the records holding value v at g; with g = 0, all records (v = 0).  The records are
split into two halves (alternating blocks of 2^14 records, a few segments each: a token appears as a
candidate in many decisions, all recorded with its segment, so it stays in one half),
and for each value x of slot s the change d = log o(x,v) - log o(v) is measured in each half (o(x,v)
smoothed toward o(v); 0 unless x has ceil(mincount/2) records with v in that half).  The effect is
sqrt(max(0, mean of (dA - mean dA)(dB - mean dB))) over the records holding s: a split-half
covariance, so a real effect (the same sign in both halves) counts and chance deviations,
independent between halves, average out; centring removes a shift shared by all values (the
smoothing's low-count bias) -- an irrelevant feature gets about 0 even under one of many gate values.  Returns
(v, s) -> effect and v -> (chosen, not chosen).
"""
function slot_effects(m::ChainModel, recs, n, g, p0; mincount = m.opts.mincount)
  Z = (Int32(0), Int32(0), Int32(0), Int32(0))
  bump(t, h, ch) = h == 0 ? (ch ? (t[1] + Int32(1), t[2], t[3], t[4]) : (t[1], t[2] + Int32(1), t[3], t[4])) :
    (ch ? (t[1], t[2], t[3] + Int32(1), t[4]) : (t[1], t[2], t[3], t[4] + Int32(1)))
  pair = Dict{Tuple{Int32,Int16,Int32},NTuple{4,Int32}}()
  vc = Dict{Int32,NTuple{4,Int32}}()
  @inbounds for j in recs
    r = m.recs[j]
    v = g == 0 ? Int32(0) : m.ids[r.start+g-1]
    (g > 0 && v == 0) && continue
    ch, h = r.chosen == 1, (j >> GATEBLOCK) & 1
    vc[v] = bump(get(vc, v, Z), h, ch)
    for s in 1:n
      s == g && continue
      x = m.ids[r.start+s-1]
      x == 0 && continue
      k = (v, Int16(s), x)
      pair[k] = bump(get(pair, k, Z), h, ch)
    end
  end
  hmin = max(1, cld(mincount, 2))
  odds(a, b, o) = a + b >= hmin ? smoothOdds(Float64(a), Float64(b), o) : o
  ov = Dict(v => (odds(t[1], t[2], p0), odds(t[3], t[4], p0)) for (v, t) in vc)
  # the changes in each half, then their covariance about each half's mean change for the slot:
  # smoothOdds shifts low counts the same way in both halves (e.g. 0 or 1 chosen where 2.5 are
  # expected comes out below o(v)), a bias the covariance alone would count as an effect
  den = Dict{Tuple{Int32,Int16},Float64}()
  sums = Dict{Tuple{Int32,Int16},NTuple{3,Float64}}()     # (weight, sum w*dA, sum w*dB) over supported values
  deltas = Tuple{Tuple{Int32,Int16},Float64,Float64,Float64}[]
  for ((v, s, x), t) in pair
    k = (v, s)
    w = Float64(sum(t))
    den[k] = get(den, k, 0.0) + w
    (t[1] + t[2] >= hmin && t[3] + t[4] >= hmin) || continue
    oa, ob = ov[v]
    da = log(odds(t[1], t[2], oa)) - log(oa)
    db = log(odds(t[3], t[4], ob)) - log(ob)
    push!(deltas, (k, w, da, db))
    (sw, sa, sb) = get(sums, k, (0.0, 0.0, 0.0))
    sums[k] = (sw + w, sa + w * da, sb + w * db)
  end
  num = Dict{Tuple{Int32,Int16},Float64}()
  for (k, w, da, db) in deltas
    (sw, sa, sb) = sums[k]
    num[k] = get(num, k, 0.0) + w * (da - sa / sw) * (db - sb / sw)
  end
  Dict(k => Float32(sqrt(max(0.0, get(num, k, 0.0) / d))) for (k, d) in den),
    Dict(v => (t[1] + t[3], t[2] + t[4]) for (v, t) in vc)
end

"""
Build m.gates for every unlabeled decision type with a fixed record layout (others: every slot is
always admitted).  A slot is admitted always if its effect (see slot_effects) reaches `gate`;
otherwise when a value of a gate slot (one with few values) is present under which its effect
reaches `gate`.
"""
function build_gates!(m::ChainModel; mincount = m.opts.mincount)
  θ = m.opts.gate
  for (tk, recs) in m.typerecs
    isempty(recs) && continue
    n = Int(m.recs[recs[1]].n)
    all(j -> m.recs[j].label == 0 && m.lsid[j] == 0 && m.recs[j].n == n, recs) || continue
    p0 = priorodds(m, lpriorkey(tk, UInt32(0)))
    nv = [Set{Int32}() for _ in 1:n]
    for j in recs, s in 1:n
      length(nv[s]) > MAXGATEVALS && continue
      x = m.ids[m.recs[j].start+s-1]
      x == 0 || push!(nv[s], x)
    end
    gslots = [s for s in 1:n if 0 < length(nv[s]) <= MAXGATEVALS]
    e0, _ = slot_effects(m, recs, n, 0, p0; mincount)
    effect = Float32[get(e0, (Int32(0), Int16(s)), 0f0) for s in 1:n]
    always = BitVector(effect .>= θ)
    byvalue, effectby, valcount = Dict{Int32,Vector{Int16}}(), Dict{Tuple{Int32,Int16},Float32}(), Dict{Int32,Int}()
    for g in gslots
      eg, vc = slot_effects(m, recs, n, g, p0; mincount)
      for ((v, s), e) in eg
        valcount[v] = sum(vc[v])
        sum(vc[v]) >= mincount || continue
        effectby[(v, s)] = e
        sum(vc[v]) >= GATEMIN || continue                   # small cells: chance effects (see GATEMIN)
        (!always[s] && e >= θ) && push!(get!(() -> Int16[], byvalue, v), s)
      end
    end
    m.gates[tk] = Gates(n, effect, always, gslots, byvalue, effectby, valcount)
  end
end

"The slots of record `Ic` (type tk) that pass the relevance gates: all of them without gates."
function admitted(m::ChainModel, tk, Ic)
  g = get(m.gates, tk, nothing)
  (g === nothing || length(Ic) != g.nslots) && return collect(eachindex(Ic))
  ok = copy(g.always)
  for s in g.gateslots
    id = get(m.idof, Ic[s], Int32(0))
    id == 0 && continue
    for t in get(g.byvalue, id, Int16[])
      ok[t] = true
    end
  end
  findall(ok)
end

"Add record j to dense tallies (column t; N: grouped decisions per label set, expanded by expandgroups!)."
@inline function tallycol!(m::ChainModel, C1, C2, N, j, r, t)
  g = m.lsid[j]
  if g == 0
    r.chosen == 1 ? (C1[r.label+1, t] += 1) : (C2[r.label+1, t] += 1)
  else
    r.chosen == 1 && (C1[r.label+1, t] += 1; C2[r.label+1, t] -= 1)   # the gold label: chosen, not "not chosen"
    N[m.lsloc[g], t] += 1
  end
end

"Each grouped decision counts as not chosen for every label of its set (gold corrected by tallycol!); tls: the type's label sets."
function expandgroups!(m::ChainModel, C2, N, tls)
  @inbounds for g in axes(N, 1), t in axes(N, 2)
    n = N[g, t]
    n == 0 && continue
    for l in m.lsets[tls[g]]
      C2[l+1, t] += n
    end
  end
end

"Fill the joint tables (see the header); returns pair key -> its two items."
function joint_tables!(m::ChainModel, jointmin)
  for (x, k) in m.itemcount
    k >= jointmin && (push!(m.frequent, x); m.single[x] = LabelCounts())
  end
  acc = Dict{Tuple{UInt64,UInt32},Tuple{Int32,Int32}}()
  members = Dict{UInt64,Tuple{UInt64,UInt64}}()
  F = UInt64[]
  for (j, r) in enumerate(m.recs)
    empty!(F)
    for q in r.start:r.start+r.n-1
      x = m.items[q]
      x != ABSENT && x in m.frequent && push!(F, x)
    end
    for x in F
      tally!(m, m.single[x], j)
    end
    g = m.lsid[j]
    labs = g == 0 ? (r.label,) : m.lsets[g]
    for a in 1:length(F)-1, b in a+1:length(F)
      key = extendkey(extendkey(UInt64(0), F[a]), F[b])
      haskey(members, key) || (members[key] = (F[a], F[b]))
      for l in labs
        (c1, c2) = get(acc, (key, l), (Int32(0), Int32(0)))
        acc[(key, l)] = r.chosen == 1 && l == r.label ? (c1 + Int32(1), c2) : (c1, c2 + Int32(1))
      end
    end
  end
  for ((key, l), (c1, c2)) in acc
    get!(LabelCounts, m.pairs, key)[l] = (Float64(c1), Float64(c2))
  end
  members
end

"Fill `m.screened` (see screening in the header)."
function screen_items!(m::ChainModel, members)
  (; screen, screenby, mincount) = m.opts
  labelsof = Dict{UInt64,Vector{UInt32}}()               # decision type -> its labels
  for (j, r) in enumerate(m.recs)
    ls = get!(() -> UInt32[], labelsof, r.typekey)
    for l in (m.lsid[j] == 0 ? (r.label,) : m.lsets[m.lsid[j]])
      l in ls || push!(ls, l)
    end
  end
  typeof_ = m.itemtype
  # per-label λ of a set from its counts and the odds its subsets predict
  λ(c, l, e) = ((c1, c2) = get(c, l, (0, 0)); c1 + c2 < max(mincount, 1) ? 0.0 : log(smoothOdds(c1, c2, e)) - log(e))
  combine(v) = screenby == :maxdev || length(v) == 1 ? maximum(abs, v) : sqrt(sum(abs2, v .- sum(v) / length(v)) / length(v))
  lam1 = Dict(x => (ls = labelsof[typeof_[x]]; ls => [λ(m.single[x], l, priorodds(m, lpriorkey(typeof_[x], l))) for l in ls])
    for x in m.frequent)
  best = Dict(x => combine(last(lam1[x])) for x in m.frequent)
  for (key, (a, b)) in members
    ls, la = lam1[a]
    lb = last(lam1[b])
    tk = typeof_[a]
    v = [λ(m.pairs[key], l, priorodds(m, lpriorkey(tk, l)) * exp(la[j] + lb[j])) for (j, l) in enumerate(ls)]
    sc = combine(v)
    sc > best[a] && (best[a] = sc)
    sc > best[b] && (best[b] = sc)
  end
  for (x, k) in m.itemcount
    (k < max(mincount, 1) || (x in m.frequent && best[x] < screen)) && push!(m.screened, x)
  end
end

priorodds(m::ChainModel, pk::UInt64) = (p = get(m.prior, pk, (0.0, 0.0)); (p[1] + 0.5) / (p[2] + 0.5))
waitscore(m::ChainModel, st::State, key::AgendaKey) = log(priorodds(m, lpriorkey(typekey(key[1], key[2]), UInt32(0))))

"Intersection of two sorted lists (binary-search the longer one when sizes are very different)."
function intersect_sorted(a::AbstractVector{Int32}, b::AbstractVector{Int32})
  length(a) > length(b) && ((a, b) = (b, a))
  out = Int32[]
  if 16 * length(a) < length(b)
    lo = 1
    for x in a
      lo = searchsortedfirst(b, x, lo, length(b), Base.Order.Forward)
      lo > length(b) && break
      b[lo] == x && push!(out, x)
    end
  else
    i = j = 1
    while i <= length(a) && j <= length(b)
      a[i] < b[j] ? (i += 1) : a[i] > b[j] ? (j += 1) : (push!(out, a[i]); i += 1; j += 1)
    end
  end
  out
end

"Dense per-label tallies (index label+1) as a LabelCounts: an entry for each label seen, as tally! makes."
function todict(c1::AbstractVector{Float64}, c2::AbstractVector{Float64})
  d = LabelCounts()
  @inbounds for k in eachindex(c1)
    (c1[k] > 0 || c2[k] > 0) && (d[UInt32(k - 1)] = (c1[k], c2[k]))
  end
  d
end

function labelcounts(m::ChainModel, list)
  nl, tls = tallyshape(m, list)
  C1, C2, N = zeros(nl, 1), zeros(nl, 1), zeros(length(tls), 1)
  @inbounds for j in list
    tallycol!(m, C1, C2, N, j, m.recs[j], 1)
  end
  expandgroups!(m, C2, N, tls)
  todict(view(C1, :, 1), view(C2, :, 1))
end

"Mark (or unmark) the view filter's ids in `m.infilt`."
markfilt!(m::ChainModel, filt, on::Bool) = foreach(x -> (m.infilt[m.idof[x]] = on), filt)

"Does record j contain every item of the marked filter (of length nf)?  (Items are unique within a record.)"
@inline function hasmarked(m::ChainModel, j, nf)
  r, n = m.recs[j], 0
  @inbounds for q in r.start:r.start+r.n-1
    id = m.ids[q]
    id > 0 && m.infilt[id] && (n += 1)
  end
  n == nf
end

"Positions of these items in their records, or nothing if positions are off or some item's varies."
function slotpositions(m::ChainModel, ids)
  m.usepos || return nothing
  out = Int16[]
  for id in ids
    s = m.slotof[id]
    s > 0 || return nothing
    s in out || push!(out, s)
  end
  out
end

"Does record j hold a marked filter item at each of `fslots` (nf items)?"
@inline function hasmarkedat(m::ChainModel, j, fslots, nf)
  r, n = m.recs[j], 0
  @inbounds for s in fslots
    s <= r.n || continue
    id = m.ids[r.start+s-1]
    id > 0 && m.infilt[id] && (n += 1)
  end
  n == nf
end

"Keep the records of `L` that contain every item of `filt`."
function filterrecs(m::ChainModel, L::Vector{Int32}, filt)
  isempty(filt) && return L
  markfilt!(m, filt, true)
  fslots = slotpositions(m, Int32[m.idof[x] for x in filt])
  try
    fslots === nothing ? Int32[j for j in L if hasmarked(m, j, length(filt))] :
      Int32[j for j in L if hasmarkedat(m, j, fslots, length(filt))]
  finally
    markfilt!(m, filt, false)
  end
end

"""
The view's records as a list of its own: starting from the shortest posting among its filter
items when that's shorter than the base (its records hold the item, so they're all in there).
"""
function buildlist(m::ChainModel, v::RecView)
  isempty(v.filt) && return v.base
  b = argmin(x -> length(get(m.postings, x, v.base)), v.filt)
  P = get(m.postings, b, nothing)
  (P === nothing || length(P) >= length(v.base)) && return filterrecs(m, v.base, v.filt)
  filterrecs(m, intersect_sorted(v.base, P), filter(!=(b), v.filt))
end

total(c::LabelCounts) = sum(((c1, c2),) -> c1 + c2, values(c); init = 0.0)

"Cache an entry that read `nread` records (EXACT: all); `built` list entries were newly made for it; `persist`: may be saved (see Persistent counts) if counting it read `cost` ≥ keepmin records."
function store!(m::ChainModel, key::UInt64, e::Entry; built = 0, nread = EXACT, persist = true, cost = 0)
  # (not with climbing: sets built on a climbing step depend on the rings, not only the events)
  m.keepcounts && persist && nread == EXACT && cost >= m.keepmin && isempty(m.climb) && pcget(m.pcounts, key) === nothing &&
    (m.newcounts[key] = e[2])
  if built > 0
    m.cached + built > m.maxcached && (empty!(m.cache); m.cached = 0; CACHE_EMPTIED[] += 1)
    m.cached += built
  end
  m.cache[key] = (e, nread)
  e
end

"Records a scan may read (see Sampling in the header), from the candidates' current odds."
function samplecap(m::ChainModel, odds; nlabels = 1)
  m.opts.sample > 0 || return EXACT
  p = sort!([o / (1 + o) for o in odds]; rev = true)
  p2 = length(p) >= 2 ? p[2] : p[1]
  n = nlabels * m.opts.sample / max(p2, 1e-12)
  n >= typemax(Int) / 2 ? EXACT : ceil(Int, n)
end

"Evenly spaced positions 1..B, `n` of them."
sample_positions(B::Int, n::Int) = (1 + ((i - 1) * B) ÷ n for i in 1:n)

"""
Counts per label for each of `xs` (with ids `tids`) over the view's base records at `positions`.
With climbing trials (ids already marked in m.rtrial/m.rlevel), also each one's matching records
with their innermost ring level, returned as the second value.
"""
function scan_counts(m::ChainModel, v::RecView, tids::Vector{Int32}, positions; nclimb = 0, climbids = Int32[])
  SCAN_HOOK[] === nothing || SCAN_HOOK[](length(positions))
  # positions to read: the trial items', the climbing rings' (one slot each), the filter's
  tslots = slotpositions(m, isempty(climbids) ? tids : [tids; climbids])
  fslots = slotpositions(m, Int32[m.idof[x] for x in v.filt])
  (tslots === nothing || fslots === nothing) && (tslots = fslots = nothing)
  nl, tls = tallyshape(m, v.base)
  C1, C2 = zeros(nl, length(tids)), zeros(nl, length(tids))   # dense tallies: label+1 x trial
  N = zeros(length(tls), length(tids))                         # grouped decisions per label set of the type
  chits = [Tuple{Int32,Int8}[] for _ in 1:nclimb]
  for (t, id) in enumerate(tids)
    m.want[id] == 0 || error("duplicate trial items")
    m.want[id] = t
  end
  markfilt!(m, v.filt, true)
  try
    nfilt, hit = length(v.filt), Int32[]
    for p in positions                               # one pass: filter and every trial item at once
      j = v.base[p]
      r, nf = m.recs[j], 0
      empty!(hit)
      ct, cl = 0, Int8(0)
      if tslots === nothing                          # read every item
        @inbounds for q in r.start:r.start+r.n-1
          id = m.ids[q]
          id == 0 && continue
          t = m.want[id]
          t > 0 && push!(hit, t)
          m.infilt[id] && (nf += 1)
          nclimb > 0 && m.rtrial[id] > 0 && (ct = m.rtrial[id]; cl = m.rlevel[id])
        end
      else                                           # read only the needed positions
        @inbounds for s in fslots
          s <= r.n || continue
          id = m.ids[r.start+s-1]
          id > 0 && m.infilt[id] && (nf += 1)
        end
        nf == nfilt || continue
        @inbounds for s in tslots
          s <= r.n || continue
          id = m.ids[r.start+s-1]
          id == 0 && continue
          t = m.want[id]
          t > 0 && push!(hit, t)
          nclimb > 0 && m.rtrial[id] > 0 && (ct = m.rtrial[id]; cl = m.rlevel[id])
        end
      end
      nf == nfilt || continue
      @inbounds for t in hit
        tallycol!(m, C1, C2, N, j, r, t)
      end
      ct > 0 && push!(chits[ct], (Int32(j), cl))    # one value per slot, so one hit per climbing trial
    end
  finally
    markfilt!(m, v.filt, false)
    foreach(id -> (m.want[id] = 0), tids)
  end
  expandgroups!(m, C2, N, tls)
  tallies = [todict(view(C1, :, t), view(C2, :, t)) for t in eachindex(tids)]
  nclimb > 0 ? (tallies, chits) : tallies
end

"Diagnostics: if set, called with the number of records each scan reads; IN_GREEDY is true inside greedy leftovers."
const SCAN_HOOK = Ref{Any}(nothing)
const IN_GREEDY = Ref(false)
const REUSE_LISTS = Ref(true)                     # (off: the old behaviour, for timing comparisons)
const MATERIALIZED = Ref([0, 0, 0])               # diagnostics: lists built, of them kept for prefixes counted in an earlier run, records
const CACHE_EMPTIED = Ref(0)                     # diagnostics: times the cache was emptied for space
const PASS_HOOK = Ref{Any}(nothing)              # diagnostics: (trials, short trials, pass skipped) per decision

"Mark climbing trial t's ring items (innermost level per item); returns the marked ids."
function markclimb!(m::ChainModel, c::Climb, t::Int)
  marked = Int32[]
  mark(x, L) = (id = get(m.idof, x, Int32(0)); id > 0 && m.rtrial[id] == 0 &&
    (m.rtrial[id] = t; m.rlevel[id] = L; push!(marked, id)))
  lo = 1
  for (L, hi) in enumerate(c.cuts)
    foreach(x -> mark(x, Int8(L)), view(c.order, lo:hi))
    lo = hi + 1
  end
  for (k, ring) in enumerate(c.outer)
    foreach(x -> mark(x, Int8(length(c.cuts) + k)), ring)
  end
  marked
end

"""
A climbing trial's entry from its scan hits: the finest ring level whose records support
`label`, as a list of its own (or the outermost, unsupported, if none does).
"""
function climbentry(m::ChainModel, c::Climb, hits, label::UInt32)
  L = nlevels(c)
  per = [LabelCounts() for _ in 1:L]
  for (j, l) in hits
    tally!(m, per[l], j)
  end
  cum, best = LabelCounts(), L
  for l in 1:L
    for (lab, (c1, c2)) in per[l]
      (a, b) = get(cum, lab, (0.0, 0.0))
      cum[lab] = (a + c1, b + c2)
    end
    (c1, c2) = get(cum, label, (0.0, 0.0))
    c1 + c2 >= max(m.opts.mincount, 1) && (best = l; break)
  end
  counts = LabelCounts()
  for l in 1:best, (lab, (c1, c2)) in per[l]
    (a, b) = get(counts, lab, (0.0, 0.0))
    counts[lab] = (a + c1, b + c2)
  end
  (RecView(Int32[j for (j, l) in hits if l <= best]), counts)
end

"""
New odds for a candidate from an entry's counts, smoothed toward its current odds (or unchanged
if unsupported).  A set with no records at all is never supported, whatever `mincount` is: it
carries no information, and treating it as supported would stop leftover backoff at a depth
with no data instead of moving to a shallower prefix that has some.
"""
function update_odds(m::ChainModel, e, label::UInt32, x::Float64; pn = NaN)
  (c1, c2) = get(e[2], label, (0.0, 0.0))
  c1 + c2 < max(m.opts.mincount, 1) && return x, false
  m.opts.shrink > 0 && return shrinkOdds(c1, c2, x, m.opts.shrink), true
  smoothOdds(c1, c2, x, smoothing(m, c1 + c2, pn)), true
end

"The smoothing amount d for a set of n records whose parent has pn (NaN: unknown, n is used)."
smoothing(m::ChainModel, n, pn = NaN) =
  m.opts.smoothd + m.opts.smootha * (m.opts.smoothn == :parent && !isnan(pn) ? pn : n)

"Records (chosen + not) of chain state `st`'s last prefix for `label`: the parent of its next step."
function parentn(m::ChainModel, st, label::UInt32)
  if depth(st) == 0
    isempty(st.views[1].base) && return NaN
    (c, oc) = get(m.prior, lpriorkey(m.recs[st.views[1].base[1]].typekey, label), (NaN, NaN))
    return c + oc
  end
  (c1, c2) = get(st.cnt[end], label, (0.0, 0.0))
  c1 + c2
end

"""
Beta shrinkage (ChainOpts.shrink = k): the probability (c1 + k p0) / (c1 + c2 + k) with p0 the parent's (prior
odds e), as odds -- a set of n records moves the estimate from the parent's in proportion to n / (n + k), however
one-sided its counts.
"""
function shrinkOdds(c1, c2, e, k)
  c1, c2 = max(c1, 0.0), max(c2, 0.0)            # (sampled counts can come out slightly inconsistent)
  p0 = isnan(e) ? 0.5 : isinf(e) ? 1.0 : e / (1 + e)
  p = clamp((c1 + k * p0) / (c1 + c2 + k), 1e-9, 1 - 1e-9)
  p / (1 - p)
end

##======== criteria

clr(lx) = lx .- (sum(lx) / length(lx))

function criterion(sel::Symbol, x::Vector{Float64}, y::Vector{Float64})
  lx, ly = log.(x), log.(y)
  sel in (:maxpos, :maxposneg) && return maximum(ly .- lx)   # (one shared chain: the largest positive change over the values)
  sel == :maxabsscore && return maximum(abs.(ly) .- abs.(lx))   # (the largest growth of a value's |log odds|, either sign)
  (sel == :avglogodds || sel == :maxdev || length(x) == 1) && return sum(abs.(ly .- lx)) / length(x)
  Δ = clr(ly) .- clr(lx)
  sel == :clr_mean_abs ? sum(abs.(Δ)) / length(Δ) : sqrt(sum(Δ .^ 2) / length(Δ))
end

top2(x) = (p = sortperm(x; rev = true); length(p) == 1 ? (p[1], 0) : (p[1], p[2]))

##======== chains

"""
One candidate's chain: after each step k (k = 0 first), the set's key, its records (a view) and
size, and the odds; `members` are the items added, in order.
"""
mutable struct ChainState
  keys::Vector{UInt64}
  views::Vector{RecView}
  sizes::Vector{Int}
  members::Vector{UInt64}
  odds::Vector{Float64}
  alts::Vector{Dict{UInt64,Float64}}   # poolmode :local: per member, the odds every alternative reached at its step
  cnt::Vector{LabelCounts}             # per prefix, its counts per label (the root's: NOCOUNTS; see parentn)
end
ChainState(o::Float64, typerecs::Vector{Int32}) = ChainState(UInt64[0], [RecView(typerecs)], [0], UInt64[], [o], Dict{UInt64,Float64}[], [NOCOUNTS])
ChainState(keys, views, sizes, members, odds) = ChainState(keys, views, sizes, members, odds, Dict{UInt64,Float64}[], fill(NOCOUNTS, length(keys)))
depth(st::ChainState) = length(st.keys) - 1

"The view for prefix view `v` (with `n` items) plus x, before any list is built for it."
childview(m::ChainModel, v::RecView, n::Int, x::UInt64) =
  n == 0 && haskey(m.postings, x) ? RecView(m.postings[x]) : RecView(v.base, [v.filt; x])

"""
Entries (records, counts per label) for chain prefix k (0..depth) plus each of `xs`, from the
cache, the joint tables, the rarity shortcut, intersections with rare postings, or one pass over
the prefix view's base (see Records in the header), reading at most `cap` of its records when
sampling (see Sampling).
"""
function extend_all(m::ChainModel, st::ChainState, k::Int, xs::Vector{UInt64}; cap::Int = EXACT)
  v, pkey, n = st.views[k+1], st.keys[k+1], st.sizes[k+1]
  out = Vector{Entry}(undef, length(xs))
  todo, climbs = Int[], Int[]
  for (i, x) in enumerate(xs)
    key = extendkey(pkey, x)
    ce = get(m.cache, key, nothing)
    if ce !== nothing && ce[2] >= cap
      out[i] = ce[1]
    elseif haskey(m.vclimb, x)
      push!(climbs, i)
    elseif get(m.itemcount, x, 0) < max(m.opts.mincount, 1)
      out[i] = NOENTRY                               # no set containing x can be supported
    elseif x in m.frequent && n == 0
      out[i] = (childview(m, v, n, x), m.single[x])
    elseif x in m.frequent && n == 1 && st.members[1] in m.frequent
      out[i] = (childview(m, v, n, x), get(m.pairs, key, NOCOUNTS))
    elseif (pc = pcget(m.pcounts, key)) !== nothing   # counted in an earlier run
      out[i] = (childview(m, v, n, x), pc)
    else
      push!(todo, i)
    end
  end
  (isempty(todo) && isempty(climbs)) && return out
  v = materialize!(m, st, k)                         # counting has to read the prefix's records now
  B = length(v.base)
  # skip the pass when every trial item is cheaper to count from its own list (binary-search
  # intersection with the base: ~|list| log₂|base|) than a pass over the base (~|base| × record length)
  short(i) = (P = get(m.postings, xs[i], nothing); P !== nothing && 16 * length(P) < B)
  rare(todo) = !isempty(todo) && all(short, todo) &&
    sum(i -> length(m.postings[xs[i]]), todo) * log2(max(B, 2)) < B * m.reclen
  PASS_HOOK[] === nothing || isempty(todo) || PASS_HOOK[](length(todo), count(short, todo), rare(todo))
  if cap < B && !isempty(todo) && !rare(todo)        # sample; items with few sampled records stay exact
    pos = sample_positions(B, cap)
    T = scan_counts(m, v, Int32[m.idof[xs[i]] for i in todo], pos)
    sc = B / cap
    exact_ = Int[]
    for (t, i) in enumerate(todo)
      if total(T[t]) * sc < 10 * max(m.opts.mincount, 1)
        push!(exact_, i)
      else
        c = LabelCounts(l => (c1 * sc, c2 * sc) for (l, (c1, c2)) in T[t])
        out[i] = store!(m, extendkey(pkey, xs[i]), (childview(m, v, n, xs[i]), c); nread = cap)
      end
    end
    todo = exact_
  end
  if rare(todo)
    for i in todo                                    # all rare: intersect their postings with the base
      L = n == 0 ? m.postings[xs[i]] : filterrecs(m, intersect_sorted(v.base, m.postings[xs[i]]), v.filt)
      out[i] = store!(m, extendkey(pkey, xs[i]), (RecView(L), labelcounts(m, L)); built = n == 0 ? 0 : length(L),
        cost = length(m.postings[xs[i]]))
    end
    todo = Int[]
  end
  (isempty(todo) && isempty(climbs)) && return out
  # one exact pass for the remaining items and every climbing trial
  marked = Int32[]
  try
    for (t, i) in enumerate(climbs)
      append!(marked, markclimb!(m, m.climb[first(m.vclimb[xs[i]])], t))
    end
    tallies, chits = scan_counts(m, v, Int32[m.idof[xs[i]] for i in todo], 1:B; nclimb = max(length(climbs), 1),
      climbids = marked)
    for (t, i) in enumerate(todo)
      out[i] = store!(m, extendkey(pkey, xs[i]), (childview(m, v, n, xs[i]), tallies[t]); cost = B)
    end
    for (t, i) in enumerate(climbs)
      x, label = m.vclimb[xs[i]]
      e = climbentry(m, m.climb[x], chits[t], label)
      out[i] = store!(m, extendkey(pkey, xs[i]), e; built = length(e[1].base), persist = false)
    end
  finally
    foreach(id -> (m.rtrial[id] = 0; m.rlevel[id] = 0), marked)
  end
  out
end

"Cumulative counts per ring level of climbing key `vk` at chain prefix k (level L: rings 1..L), exact."
function climblevels(m::ChainModel, st::ChainState, k::Int, vk::UInt64)
  x, _ = m.vclimb[vk]
  c = m.climb[x]
  v = st.views[k+1]
  marked = markclimb!(m, c, 1)
  hits = try
    last(scan_counts(m, v, Int32[], 1:length(v.base); nclimb = 1, climbids = marked))[1]
  finally
    foreach(id -> (m.rtrial[id] = 0; m.rlevel[id] = 0), marked)
  end
  per = [LabelCounts() for _ in 1:nlevels(c)]
  for (j, l) in hits
    tally!(m, per[l], j)
  end
  for l in 2:length(per), (lab, (c1, c2)) in per[l-1]
    (a, b) = get(per[l], lab, (0.0, 0.0))
    per[l][lab] = (a + c1, b + c2)
  end
  per
end

"The climbing trial key for item x scored for `label` (registered so extend_all recognizes it)."
function climbkey!(m::ChainModel, x::UInt64, label::UInt32)
  k = hash((x, :climb, label))
  m.vclimb[k] = (x, label)
  k
end

"""
Add item x (entry e, new odds y) to the chain, or keep the set (x === nothing: the item is
absent).  The new prefix gets a list of its own once it holds fewer than `keepfrac` of its
view's base.
"""
function advance!(m::ChainModel, st::ChainState, x, e, y)
  if x === nothing
    push!(st.keys, st.keys[end]); push!(st.views, st.views[end]); push!(st.sizes, st.sizes[end])
    push!(st.cnt, st.cnt[end])
  else
    v, c = e
    key = extendkey(st.keys[end], x)
    # keepfrac < 1: the new prefix gets a list once it holds under keepfrac of its base; keepfrac = 1:
    # it stays a view until a count actually has to scan it (materialize!)
    if !isempty(v.filt) && m.keepfrac < 1 && total(c) < m.keepfrac * length(v.base)
      v = RecView(buildlist(m, v))
      store!(m, key, (v, labelcounts(m, v.base)); built = length(v.base))   # exact, whatever e was (not kept: it was counted already)
    end
    push!(st.keys, key); push!(st.views, v); push!(st.sizes, st.sizes[end] + 1)
    push!(st.cnt, e[2])
    push!(st.members, x)
  end
  push!(st.odds, y)
end

"""
Give chain prefix k a record list of its own if it's still a view (keepfrac = 1: lists are built
only when a count has to scan them), updating its cached entry; returns its records.
"""
function materialize!(m::ChainModel, st::ChainState, k::Int)
  v = st.views[k+1]
  (m.keepfrac >= 1 && !isempty(v.filt)) || return v
  key = st.keys[k+1]
  ce = get(m.cache, key, nothing)
  if REUSE_LISTS[] && ce !== nothing && isempty(ce[1][1].filt)   # built already (by another chain reaching this prefix)
    st.views[k+1] = ce[1][1]
    return ce[1][1]
  end
  L = buildlist(m, v)
  MATERIALIZED[][1] += 1; MATERIALIZED[][3] += length(L)
  v2 = RecView(L)
  st.views[k+1] = v2
  if ce !== nothing
    store!(m, key, (v2, ce[1][2]); built = length(L), nread = ce[2], persist = false)
  elseif REUSE_LISTS[] && (pc = pcget(m.pcounts, key)) !== nothing   # counts from an earlier run: keep the list too, for next time
    MATERIALIZED[][2] += 1
    store!(m, key, (v2, pc); built = length(L), persist = false)
  end
  v2
end

"Counts per label for an explicit item set of decision type `tk`, built step by step (tests, diagnostics)."
function setcounts(m::ChainModel, tk::UInt64, S::Vector{UInt64})
  st = ChainState(1.0, get(m.typerecs, tk, NOLIST))
  e = NOENTRY
  for x in S
    e = only(extend_all(m, st, depth(st), [x]))
    advance!(m, st, x, e, 1.0)
  end
  e[2], st
end

present_items(Ic, slots) = UInt64[Ic[s] for s in slots]

"""
Run the decision-level chain over slots for all candidates at once.  `I[c][s]` is candidate c's
item in slot s.  Returns the chain states and the slots left unused.  With a `target`, steps
are chosen to approach it; otherwise to maximize the criterion.
"""
function run_chain(m::ChainModel, I, labels, priors, typerecs; target = nothing, restarts = 0, tk = nothing)
  K, nslots = length(I), length(I[1])
  states = [ChainState(p, typerecs) for p in priors]
  remaining = collect(1:nslots)
  # each candidate's admitted slots (relevance gates; stand-in slots excluded): for the others, a step
  # on the slot leaves that candidate's set and odds as they are, as an absent item does
  adm = map(I) do Ic
    a = tk === nothing ? collect(1:nslots) : admitted(m, tk, Ic)
    tk !== nothing && haskey(m.nochain, tk) && (a = [s for s in a if !(s in m.nochain[tk])])
    BitSet(a)
  end
  absent(c, s) = I[c][s] == ABSENT || !(s in adm[c])
  dist(y) = criterion(m.opts.select, y, target)
  while !isempty(remaining)
    x = [s.odds[end] for s in states]
    best, bs, by, be = -Inf, 0, x, nothing
    nbest, nbs, nby, nbe = -Inf, 0, x, nothing        # (maxposneg) the step lowering some value most
    # each candidate's entries for all its remaining (present) items, in one pass per candidate
    cap = samplecap(m, x; nlabels = all(iszero, labels) ? 1 : K)
    E = map(enumerate(states)) do (c, st)
      slots = [s for s in remaining if !absent(c, s)]
      Dict(zip(slots, extend_all(m, st, depth(st), present_items(I[c], slots); cap)))
    end
    for s in remaining
      # a candidate whose item in this slot is absent (still pending) keeps its current set and odds
      es = [absent(c, s) ? nothing : E[c][s] for c in 1:K]
      y = similar(x)
      any_ok = false
      for c in 1:K
        es[c] === nothing && (y[c] = x[c]; continue)
        y[c], ok = update_odds(m, es[c], labels[c], x[c])
        any_ok |= ok
      end
      any_ok || continue
      sc = target === nothing ? criterion(m.opts.select, x, y) : dist(x) - dist(y)
      if m.opts.select == :maxposneg && target === nothing
        nsc = maximum(log.(x) .- log.(y))
        nsc > nbest && ((nbest, nbs, nby, nbe) = (nsc, s, y, es))
      end
      sc > best && ((best, bs, by, be) = (sc, s, y, es))
    end
    # maxposneg: no step raises a value by tau -- take the one lowering some value most, if by tau (then raising again)
    m.opts.select == :maxposneg && best < m.opts.tau && nbest >= m.opts.tau && ((best, bs, by, be) = (nbest, nbs, nby, nbe))
    (bs == 0 || best < m.opts.tau) && break
    for (c, st) in enumerate(states)
      advance!(m, st, be[c] === nothing ? nothing : I[c][bs], be[c], by[c])
    end
    deleteat!(remaining, findfirst(==(bs), remaining))
    if m.opts.target && restarts < m.opts.maxrestarts && K > 1
      changed = top2(by) != top2(x)
      if changed && (target === nothing || top2(by) != top2(target))
        return run_chain(m, I, labels, priors, typerecs; target = copy(by), restarts = restarts + 1, tk)
      end
    end
  end
  states, [Int[s for s in remaining if s in adm[c]] for c in 1:K]   # each candidate's leftovers: its admitted slots
end

"""
One step of a candidate's own chain (:maxdev).  Without a target: add the feature that moves
its odds the most; with target odds `t`: the one that brings its odds closest to `t`.  The
step must score at least `tau`, unless `exhaust`, when any feature with `mincount` joint
records will do.  Returns whether a feature was added.
"""
function step_single!(m::ChainModel, st::ChainState, Ic, label, remaining; target = nothing, exhaust = false,
  cap = EXACT, nosingle = NOSINGLE, peek = false)
  x = st.odds[end]
  ss = STEPSCORE_OVERRIDE[] === nothing ? m.opts.stepscore : STEPSCORE_OVERRIDE[]   # (posnegonly = :attach: labeled decisions :abs)
  # bipolar (user's design 2026-11-03): the first step by |Δ|; later ones keep the previous step's direction -- the
  # largest move that way, by tau -- and else turn (the largest the other way, by tau)
  ss == :bipolar && (ss = length(st.odds) < 2 ? :abs : st.odds[end] >= st.odds[end-1] ? :posneg : :negpos)
  pn = m.opts.smoothn == :parent ? parentn(m, st, label) : NaN
  best, bs, by, be = -Inf, 0, x, nothing
  slots = [s for s in remaining if Ic[s] != ABSENT]
  tr = CHAIN_TRACE[] === nothing ? nothing : Tuple{Int,Float64,Float64}[]
  k = target === nothing && !exhaust ? m.opts.pickfewest : 0
  passing = Tuple{Float64,Int,Float64,Any}[]          # with pickfewest: (score, slot, odds, entry) of those reaching tau
  usepairs = m.opts.pairsteps && target === nothing && !exhaust
  rec = m.opts.poolsigma > 0 && m.opts.poolmode == :local && target === nothing
  alts = rec ? Dict{UInt64,Float64}() : nothing    # every supported alternative's odds at this step
  single = Dict{Int,Tuple{Any,Float64,Float64}}()   # with pairsteps: slot -> (entry, odds, score) of supported singles
  negfall = ss in (:posneg, :posnegprior, :negpos)
  pbest, pbs, pby, pbe = -Inf, 0, x, nothing          # (peek) the step raising the odds most
  pc = ss in CRIT_STEPS ? prefixcounts(m, st, label) : (0.0, 0.0)   # (association scores) the prefix's counts
  nbest, nbs, nby, nbe = -Inf, 0, x, nothing
  unsup = Int[]                                     # (peek) the present slots the prefix doesn't support with
  for (s, e) in zip(slots, extend_all(m, st, depth(st), present_items(Ic, slots); cap))
    y, ok = update_odds(m, e, label, x; pn)
    tr === nothing || peek || push!(tr, (s, ok ? abs(log(y) - log(x)) : NaN, total(e[2])))
    ok || (push!(unsup, s); continue)
    sc = target !== nothing ? abs(log(x) - log(target)) - abs(log(y) - log(target)) :
      ss in (:signed, :posneg, :posnegprior) ? log(y) - log(x) :   # the feature that raises the odds most
      ss == :negpos ? log(x) - log(y) :                                    # (negpos: lowers them most)
      ss == :maxmin ? minimum(o -> abs(log(y) - log(o)), st.odds) :   # away from every earlier stage
      ss == :maxmin2 ? maxmin2(e, label, st.odds) :                    # ...each smoothed toward that stage
      ss in CRIT_STEPS ? (abs(log(y) - log(x)) >= m.opts.tau ?         # association (if the step moves the odds by tau),
        m.opts.tau + step_crit(ss, pc..., get(e[2], label, (0.0, 0.0))...) : -Inf) :   # shifted so tau still stops the chain
      abs(log(y) - log(x))
    usepairs && (single[s] = (e, y, sc))
    rec && (alts[Ic[s]] = y)
    s in nosingle && continue                       # (pairsfirst: joins only as a pair member)
    if negfall && target === nothing                 # posneg(prior): the fallback, the step lowering the odds most
      nsc = ss == :posneg ? log(x) - log(y) : ss == :negpos ? log(y) - log(x) : log(st.odds[1]) - log(y)   # below the chain / above it (negpos) / below the prior
      nsc > nbest && ((nbest, nbs, nby, nbe) = (nsc, s, y, e))
    end
    sc > best && ((best, bs, by, be) = (sc, s, y, e))
    peek && (rs = log(y) - log(x); rs > pbest && ((pbest, pbs, pby, pbe) = (rs, s, y, e)))
    k > 0 && sc >= m.opts.tau && push!(passing, (sc, s, y, e))
  end
  # peek (interleave): the best step and, for posneg, the fallback step lowering the odds most, without taking either
  peek && return (x = x, best = best, bs = bs, by = by, be = be, nbest = negfall ? nbest : -Inf, nbs = nbs, nby = nby, nbe = nbe,
    unsup = unsup, alts = alts, pbest = pbest, pbs = pbs, pby = pby, pbe = pbe)
  pair = nothing                                    # (slot a, slot b, pair entry, odds) if a pair wins
  if usepairs
    fs = [s for s in slots if Ic[s] in m.frequent]
    prs = [(a, b) for (i, a) in enumerate(fs) for b in fs[i+1:end]
           if haskey(m.pairs, extendkey(extendkey(UInt64(0), Ic[a]), Ic[b]))]
    if !isempty(prs)
      ep = extend_pairs(m, st, depth(st), [(Ic[a], Ic[b]) for (a, b) in prs])
      for ((a, b), e) in zip(prs, ep)
        (haskey(single, a) && haskey(single, b)) || continue    # both slots supported alone, so the pair can be split
        y, ok = update_odds(m, e, label, x; pn)
        ok || continue
        sc = abs(log(y) - log(x))
        # (:min can't beat the pair's better member, which competes as a single: no pair would win)
        m.opts.pairscore == :min && (sc = min(sc, single[a][3], single[b][3]))
        # :synergy: only a pair that beats its better member by tau, then on its full change
        m.opts.pairscore == :synergy && sc - max(single[a][3], single[b][3]) < m.opts.tau && continue
        m.opts.pairscore == :increment && (sc -= max(single[a][3], single[b][3]))
        sc > best && ((best, pair) = (sc, (a, b, e, y)))             # strictly better: ties go to the single
      end
    end
  end
  if k > 0 && !isempty(passing)                     # of the k best, the one with the fewest records (ties: better score)
    top = sort!(passing; by = p -> -p[1])[1:min(k, end)]
    (_, bs, by, be) = top[argmin([(total(p[4][2]), -p[1]) for p in top])]
  end
  # posneg(prior): no step raises the odds by tau -- take the one lowering them most, if by tau (then raising again)
  negfall && !exhaust && best < m.opts.tau && nbest >= m.opts.tau && pair === nothing && ((best, bs, by, be) = (nbest, nbs, nby, nbe))
  stop = (bs == 0 && pair === nothing) || (!exhaust && best < m.opts.tau)
  tr === nothing || CHAIN_TRACE[](:step, (cand = CHAIN_TRACE_CAND[], before = x, tried = tr,
    chosen = stop ? 0 : bs, after = stop ? x : by))
  stop && return false
  if pair !== nothing                                # (a pair only wins with best above every single's score)
    a, b, e, y = pair                                  # the pair as two members: a's own odds, then the pair's
    # (pairorder :maxdev: the member with the larger single effect first; set keys don't depend on order)
    m.opts.pairorder == :maxdev && single[b][3] > single[a][3] && ((a, b) = (b, a))
    advance!(m, st, Ic[a], single[a][1], single[a][2])
    advance!(m, st, Ic[b], e, y)
    rec && (push!(st.alts, alts); push!(st.alts, Dict{UInt64,Float64}()))   # (b's step: no alternatives: weight 1)
    deleteat!(remaining, findfirst(==(a), remaining))
    deleteat!(remaining, findfirst(==(b), remaining))
    return true
  end
  advance!(m, st, Ic[bs], be, by)
  rec && push!(st.alts, alts)
  deleteat!(remaining, findfirst(==(bs), remaining))
  true
end

const CRIT_STEPS = (:pmi, :mi, :oddspmi, :moi, :logor)

"Counts (chosen, not) for `label` in chain state st's last prefix (the decision type's prior at depth 0)."
function prefixcounts(m::ChainModel, st::ChainState, label::UInt32)
  depth(st) > 0 && return get(st.cnt[end], label, (0.0, 0.0))
  isempty(st.views[1].base) && return (0.0, 0.0)
  get(m.prior, lpriorkey(m.recs[st.views[1].base[1]].typekey, label), (0.0, 0.0))
end

"""
Association step scores (user's request 2026-11-02) between C (the candidate chosen) and feature x within the
prefix's records, from its counts (c1 chosen, c2 not) and the step's (k1, k2: those with x as well).  Every smoothing is
one count (user 2026-11-02: counts are discrete, so less than one is not enough):
  :pmi      |log p(C, x) / (p(C) p(x))|, 1 added to each cell of the 2x2 table
  :mi       the table's mutual information, sum of p log p / (p p) over its four cells (1 added to each)
  :oddspmi  |log O(C & x) - log O_ind| -- odds O(e) = n(e) / (N - n(e)), the joint smoothed (smoothOdds) toward its
            independence expectation O_ind, the odds of p(C) p(x) (marginals with 1 added).  (The word-embeddings
            project's calcBinaryMoi takes O_ind = O(C) O(x), which holds only for rare events: C here is common.)
  :moi      mutual odds information, |p(C & x) * oddspmi| with p from the smoothed joint odds
  :logor    |log odds ratio|, |log (n(C x) n(~C ~x)) / (n(C ~x) n(~C x))| = |log O(C | x) - log O(C | ~x)|, 1 added to each cell
"""
function step_crit(ss::Symbol, c1, c2, k1, k2)
  N = c1 + c2
  if ss in (:pmi, :mi, :logor)
    n = (max(k1, 0.0), max(c1 - k1, 0.0), max(k2, 0.0), max(c2 - k2, 0.0)) .+ 1.0   # (C x, C ~x, ~C x, ~C ~x)
    T = sum(n)
    pC, px = (n[1] + n[2]) / T, (n[1] + n[3]) / T
    ss == :pmi && return abs(log(n[1] / T / (pC * px)))
    ss == :logor && return abs(log(n[1] * n[4] / (n[2] * n[3])))
    marg = ((pC, px), (pC, 1 - px), (1 - pC, px), (1 - pC, 1 - px))
    return sum(n[i] / T * log(n[i] / T / (marg[i][1] * marg[i][2])) for i in 1:4)
  end
  nx = k1 + k2
  pind = (c1 + 1) / (N + 2) * (nx + 1) / (N + 2)
  oind = pind / (1 - pind)
  lj = log(smoothOdds(k1, N - k1, oind))
  d = lj - log(oind)
  ss == :oddspmi ? abs(d) : abs(d * exp(lj) / (1 + exp(lj)))
end

"""
Max-min step score with the counts smoothed toward each earlier stage in turn: the smallest
|log smoothOdds(c1, c2, o) - log o| over the chain's odds o so far (prior first).
"""
function maxmin2(e, label, odds)
  (c1, c2) = get(e[2], label, (0.0, 0.0))
  minimum(o -> abs(log(smoothOdds(c1, c2, o)) - log(o)), odds)
end

"""
Optional trace hook: when set to a function, `f(event, data)` is called for :maxdev chains with
event :step (data: candidate, odds before, each tried slot's score (NaN: unsupported) and
records, the chosen slot or 0, odds after) and :final (candidate, chain log-odds with leftovers,
the chain's members and its odds after each, from the prior).
For diagnostics; the candidate being scored is `CHAIN_TRACE_CAND[]`.
"""
const CHAIN_TRACE = Ref{Any}(nothing)
const CHAIN_TRACE_CAND = Ref{Any}(nothing)
const TRACE_KEY = Ref{Any}(nothing)               # the decision being scored, while CHAIN_TRACE is set

"""
Optional hook for greedy leftovers: `f(slot, included, ctxslots, corr)` for each leftover tried
(after the relevance filter): whether it was included, the slots of its context, its correction.
"""
const LEFTOVER_TRACE = Ref{Any}(nothing)

"""
Optional hook for calibration: `f(chain_logodds, leftover_correction)` for each candidate scored
(leftovers :none or :greedy), in candidate order; `leftalpha` weights the correction afterwards.
"""
const PARTS_HOOK = Ref{Any}(nothing)

"""
Fill `m.pool` (see Candidate pool in the header): per decision type with a fixed record layout,
slots and slot pairs (gain over their stronger slot ≥ `gainmin`, mean |λ| ≥ `lammin`) ordered by
mean total effect over a record sample.
"""
function pool_order!(m::ChainModel, nsample::Int, lammin::Float64, gainmin::Float64)
  mc = max(m.opts.mincount, 1)
  for (tk, recl) in m.typerecs
    n0 = m.recs[recl[1]].n
    all(j -> m.recs[j].n == n0, recl) || continue
    js = length(recl) <= nsample ? recl : recl[collect(sample_positions(length(recl), nsample))]   # evenly spaced
    lab = [m.recs[j].label for j in js]
    val(j, q) = m.items[m.recs[j].start+q-1]
    p = Dict(l => priorodds(m, lpriorkey(tk, l)) for l in unique(lab))
    sm(c, l, e) = ((c1, c2) = c; c1 + c2 < mc ? e : smoothOdds(c1, c2, e))
    count1(q) = (d = Dict{Tuple{UInt64,UInt32},Tuple{Float64,Float64}}();
      for (r, j) in enumerate(js)
        x = val(j, q); x == ABSENT && continue
        (a, b) = get(d, (x, lab[r]), (0.0, 0.0))
        d[(x, lab[r])] = m.recs[j].chosen == 1 ? (a + 1, b) : (a, b + 1)
      end; d)
    c1s = [count1(q) for q in 1:n0]
    # each record's smoothed odds for its own value in each slot
    o = [let x = val(j, q); x == ABSENT ? NaN : sm(c1s[q][(x, lab[r])], lab[r], p[lab[r]]) end
         for (r, j) in enumerate(js), q in 1:n0]
    entries = Tuple{Float64,Int16,Int16}[]
    for q in 1:n0
      v = [abs(log(o[r, q]) - log(p[lab[r]])) for r in eachindex(js) if !isnan(o[r, q])]
      push!(entries, (isempty(v) ? 0.0 : sum(v) / length(js), Int16(q), Int16(0)))
    end
    for q1 in 1:n0-1, q2 in q1+1:n0
      d = Dict{Tuple{UInt64,UInt64,UInt32},Tuple{Float64,Float64}}()
      for (r, j) in enumerate(js)
        x, y = val(j, q1), val(j, q2)
        (x == ABSENT || y == ABSENT) && continue
        (a, b) = get(d, (x, y, lab[r]), (0.0, 0.0))
        d[(x, y, lab[r])] = m.recs[j].chosen == 1 ? (a + 1, b) : (a, b + 1)
      end
      lam = tot = 0.0
      for (r, j) in enumerate(js)
        x, y = val(j, q1), val(j, q2)
        (x == ABSENT || y == ABSENT) && continue
        l, pl = lab[r], p[lab[r]]
        e = pl * (o[r, q1] / pl) * (o[r, q2] / pl)          # what the two values predict separately
        oxy = sm(d[(x, y, l)], l, e)
        lam += abs(log(oxy) - log(e)); tot += abs(log(oxy) - log(pl))
      end
      tot /= length(js)
      gain = tot - max(entries[q1][1], entries[q2][1])        # over its stronger slot (entries 1:n0 are the singles)
      score = m.opts.pairscore == :min ? min(tot, entries[q1][1], entries[q2][1]) :
              m.opts.pairscore == :increment ? gain : tot
      lam / length(js) >= lammin && gain >= gainmin && push!(entries, (score, Int16(q1), Int16(q2)))
    end
    sort!(entries; by = e -> (-e[1], e[3] != 0))                # ties: singles first
    m.pool[tk] = [(e[2], e[3]) for e in entries]
  end
end

"A copy of chain state `st` cut back to its prefix k."
truncstate(st::ChainState, k::Int) = ChainState(st.keys[1:k+1], st.views[1:k+1], st.sizes[1:k+1],
  st.members[1:st.sizes[k+1]], st.odds[1:k+1])

"""
Entries for chain prefix k plus both items of each pair: from the cache, the pairs table (empty
prefix), or one pass reading the filter's and the pairs' positions (else, two steps per pair).
"""
function extend_pairs(m::ChainModel, st::ChainState, k::Int, prs::Vector{Tuple{UInt64,UInt64}})
  v, pkey, n = st.views[k+1], st.keys[k+1], st.sizes[k+1]
  mc = max(m.opts.mincount, 1)
  out = Vector{Entry}(undef, length(prs))
  todo = Int[]
  for (i, (x, y)) in enumerate(prs)
    key = extendkey(extendkey(pkey, x), y)
    ce = get(m.cache, key, nothing)
    if ce !== nothing
      out[i] = ce[1]
    elseif get(m.itemcount, x, 0) < mc || get(m.itemcount, y, 0) < mc
      out[i] = NOENTRY
    elseif n == 0 && x in m.frequent && y in m.frequent
      out[i] = (RecView(v.base, [v.filt; x; y]), get(m.pairs, extendkey(extendkey(UInt64(0), x), y), NOCOUNTS))
    elseif (pc = pcget(m.pcounts, key)) !== nothing   # counted in an earlier run
      out[i] = (RecView(v.base, [v.filt; x; y]), pc)
    else
      push!(todo, i)
    end
  end
  isempty(todo) && return out
  v = materialize!(m, st, k)
  fslots = slotpositions(m, Int32[m.idof[x] for x in v.filt])
  ps = [(slotpositions(m, Int32[m.idof[prs[i][1]]]), slotpositions(m, Int32[m.idof[prs[i][2]]])) for i in todo]
  if fslots === nothing || any(p -> p[1] === nothing || p[2] === nothing, ps)
    for i in todo                                    # no positions: two steps
      x, y = prs[i]
      t = truncstate(st, k)
      ex = only(extend_all(m, t, k, [x]))
      advance!(m, t, x, ex, t.odds[end])
      out[i] = only(extend_all(m, t, k + 1, [y]))
    end
    return out
  end
  q1 = [p[1][1] for p in ps]; q2 = [p[2][1] for p in ps]
  id1 = [m.idof[prs[i][1]] for i in todo]; id2 = [m.idof[prs[i][2]] for i in todo]
  nl, tls = tallyshape(m, v.base)
  C1, C2 = zeros(nl, length(todo)), zeros(nl, length(todo))
  N = zeros(length(tls), length(todo))
  B = length(v.base)
  # a pair with a value whose own record list is short is counted from that list (every record
  # holding both values is in it); the rest share one pass over the base
  shortlist(t) = (Px = get(m.postings, prs[todo[t]][1], nothing); Py = get(m.postings, prs[todo[t]][2], nothing);
    P = Px === nothing ? Py : Py === nothing ? Px : (length(Px) <= length(Py) ? Px : Py);
    P !== nothing && (n == 0 || 16 * length(P) < B) ? P : nothing)
  lists = [shortlist(t) for t in eachindex(todo)]
  markfilt!(m, v.filt, true)
  try
    nfilt = length(v.filt)
    passes(r) = (nf = 0; @inbounds for s in fslots
        s <= r.n || continue
        id = m.ids[r.start+s-1]
        id > 0 && m.infilt[id] && (nf += 1)
      end; nf == nfilt)
    has(r, t) = q1[t] <= r.n && q2[t] <= r.n && m.ids[r.start+q1[t]-1] == id1[t] && m.ids[r.start+q2[t]-1] == id2[t]
    for t in eachindex(todo)                         # pairs counted from a short list
      lists[t] === nothing && continue
      L = n == 0 ? lists[t] : intersect_sorted(v.base, lists[t])
      @inbounds for j in L
        r = m.recs[j]
        (passes(r) && has(r, t)) || continue
        tallycol!(m, C1, C2, N, j, r, t)
      end
    end
    long = [t for t in eachindex(todo) if lists[t] === nothing]
    if !isempty(long)                                # the others: one pass over the base
      @inbounds for j in v.base
        r = m.recs[j]
        passes(r) || continue
        for t in long
          has(r, t) || continue
          tallycol!(m, C1, C2, N, j, r, t)
        end
      end
    end
  finally
    markfilt!(m, v.filt, false)
  end
  expandgroups!(m, C2, N, tls)
  for (t, i) in enumerate(todo)
    x, y = prs[i]
    out[i] = store!(m, extendkey(extendkey(pkey, x), y), (RecView(v.base, [v.filt; x; y]), todict(view(C1, :, t), view(C2, :, t)));
      cost = lists[t] === nothing ? B : length(lists[t]))
  end
  out
end

"""
One pooled chain step (see Candidate pool in the header).  `dead` collects entries found
unsupported in this chain.  Returns whether an entry was added.
"""
function step_pool!(m::ChainModel, st::ChainState, Ic, label, remaining, order, dead::Set{Tuple{Int16,Int16}})
  x0, k = st.odds[end], depth(st)
  rem = Set(remaining)
  ok_(q) = q in rem && Ic[q] != ABSENT
  considered = Tuple{Float64,Int16,Int16,Entry,Float64}[]      # (score, q1, q2, entry, new odds)
  skip = Set{Int16}()                                          # slots of supported pairs, this step
  pos = 1
  while length(considered) < m.opts.poolm && pos <= length(order)
    batch = Tuple{Int16,Int16}[]
    while pos <= length(order) && length(batch) < m.opts.poolm - length(considered) + 2
      e = order[pos]; pos += 1
      e in dead && continue
      e[2] == 0 ? (ok_(e[1]) && !(e[1] in skip) && push!(batch, e)) :
        (ok_(e[1]) && ok_(e[2]) && push!(batch, e))
    end
    isempty(batch) && break
    sb = [e for e in batch if e[2] == 0]; pb = [e for e in batch if e[2] != 0]
    if m.opts.pairscore != :total                              # a pair's score needs its slots' effects too
      for e in pb, q in (e[1], e[2])
        (q, Int16(0)) in sb || push!(sb, (q, Int16(0)))
      end
    end
    es = Dict(zip(sb, extend_all(m, st, k, UInt64[Ic[e[1]] for e in sb])))
    ep = Dict(zip(pb, extend_pairs(m, st, k, [(Ic[e[1]], Ic[e[2]]) for e in pb])))
    for e in batch                                             # in order, applying the pair filter
      length(considered) >= m.opts.poolm && break
      e[2] == 0 && e[1] in skip && continue
      en = e[2] == 0 ? es[e] : ep[e]
      y, ok = update_odds(m, en, label, x0)
      ok || (push!(dead, e); continue)
      sc = abs(log(y) - log(x0))
      if e[2] != 0 && m.opts.pairscore != :total
        eq = [(r = update_odds(m, es[(q, Int16(0))], label, x0); r[2] ? abs(log(r[1]) - log(x0)) : 0.0) for q in (e[1], e[2])]
        sc = m.opts.pairscore == :min ? min(sc, eq...) : sc - maximum(eq)   # :increment: excess over the stronger slot
      end
      push!(considered, (sc, e[1], e[2], en, y))
      e[2] == 0 || !m.opts.pairfilter || (push!(skip, e[1]); push!(skip, e[2]))
    end
  end
  isempty(considered) && return false
  # ties: a single before a pair, then the lower slot (as step_single!)
  sc, q1, q2, en, y = considered[argmax([(c[1], c[3] == 0, -Int(c[2])) for c in considered])]
  sc < m.opts.tau && return false
  if q2 == 0
    advance!(m, st, Ic[q1], en, y)
  else
    ex = only(extend_all(m, st, k, [Ic[q1]]))                  # the pair as two members, one step's odds
    yx, _ = update_odds(m, ex, label, x0)
    advance!(m, st, Ic[q1], ex, yx)
    advance!(m, st, Ic[q2], en, y)
    deleteat!(remaining, findfirst(==(Int(q2)), remaining))
  end
  deleteat!(remaining, findfirst(==(Int(q1)), remaining))
  true
end

const NOSINGLE = Set{Int}()

"""
With pairsteps and pairsfirst: the slots of record Ic (among `slots`) that form a precomputed pair
with another of them -- they may join the chain only as pair members.
"""
function pair_covered(m::ChainModel, Ic, slots)
  fs = [s for s in slots if Ic[s] != ABSENT && Ic[s] in m.frequent]
  out = Set{Int}()
  for (i, a) in enumerate(fs), b in fs[i+1:end]
    haskey(m.pairs, extendkey(extendkey(UInt64(0), Ic[a]), Ic[b])) && (push!(out, a); push!(out, b))
  end
  out
end

"""
Forked chains (forkcap; user's design 2026-11-02): a candidate's maxdev chain, except that where the step with the
largest |Δ log odds| lowers the odds and a different feature raises them most (by at least tau), the chain splits --
one branch takes the largest |step|, the other the largest raise -- and each grows on independently over all its
remaining features, splitting again, until no step reaches tau.  At most forkcap chains (later steps then follow
maxdev only).  Returns the finished chains as (state, remaining) pairs, the plain maxdev chain first.
"""
function fork_chains(m::ChainModel, Ic, label, prior, typerecs; tk = nothing)
  st0 = ChainState(prior, typerecs)
  rem0 = tk === nothing ? collect(eachindex(Ic)) : admitted(m, tk, Ic)
  if tk !== nothing && haskey(m.nochain, tk)
    nc = m.nochain[tk]
    rem0 = [s for s in rem0 if !(s in nc)]
  end
  branch(s) = ChainState(copy(s.keys), copy(s.views), copy(s.sizes), copy(s.members), copy(s.odds),
    copy(s.alts), copy(s.cnt))
  live, done_ = Any[(st0, rem0)], Any[]
  take!(st, rem, s, e, y) = (advance!(m, st, Ic[s], e, y); deleteat!(rem, findfirst(==(s), rem)))
  while !isempty(live)
    st, rem = popfirst!(live)
    pk = step_single!(m, st, Ic, label, rem; peek = true)
    if pk.bs == 0 || pk.best < m.opts.tau
      push!(done_, (st, rem))
      continue
    end
    lowers = log(pk.by) < log(pk.x)
    if lowers && pk.pbs != 0 && pk.pbs != pk.bs && pk.pbest >= m.opts.tau && length(live) + length(done_) + 2 <= m.opts.forkcap
      st2, rem2 = branch(st), copy(rem)
      take!(st2, rem2, pk.pbs, pk.pbe, pk.pby)          # the raise branch (after the maxdev one, so that stays first)
      take!(st, rem, pk.bs, pk.be, pk.by)
      pushfirst!(live, (st2, rem2))
      pushfirst!(live, (st, rem))
      (d = DC_STATS[]) === nothing || (d[:fk_forks] = get(d, :fk_forks, 0.0) + 1)
    else
      take!(st, rem, pk.bs, pk.be, pk.by)
      pushfirst!(live, (st, rem))
    end
  end
  (d = DC_STATS[]) === nothing || (d[:fk_cands] = get(d, :fk_cands, 0.0) + 1; d[:fk_leaves] = get(d, :fk_leaves, 0.0) + length(done_))
  done_
end

"A candidate's score from its forked chains (forkcomb, forkleft)."
function fork_score(m::ChainModel, Ic, label, prior, typerecs; tk = nothing)
  chains = fork_chains(m, Ic, label, prior, typerecs; tk)
  if m.opts.forkleft == :chosen                     # the chain alone picks; only the best gets its leftovers
    b = argmax([log(c[1].odds[end]) for c in chains])
    return chain_logodds(m, chains[b][1], Ic, label, prior, chains[b][2])
  end
  sc = [chain_logodds(m, c[1], Ic, label, prior, c[2]) for c in chains]
  m.opts.forkcomb == :max ? maximum(sc) : sum(sc) / length(sc)
end

"The leftover context mode for decision type tk (ctxmode, unless ctxattach / ctxtypes keep it greedy)."
leftmode(m::ChainModel, tk) = (m.opts.ctxattach && tk != typekey(:token, 1)) || (!isempty(m.ctxtypes) && !(tk in m.ctxtypes)) ? :greedy : m.opts.ctxmode
"Does decision type tk build its chain interleaved with the leftovers (interleave, :accum contexts)?"
interleaves(m::ChainModel, tk) = m.opts.interleave && tk !== nothing && leftmode(m, tk) == :accum

"Per-candidate chain (:maxdev): each candidate picks its own features."
function run_single(m::ChainModel, Ic, label, prior, typerecs; tk = nothing, seed = UInt64[])
  st = ChainState(prior, typerecs)
  remaining = tk === nothing ? collect(eachindex(Ic)) : admitted(m, tk, Ic)
  if tk !== nothing && haskey(m.nochain, tk)        # stand-in slots (e.g. word classes): only for leftover contexts
    nc = m.nochain[tk]
    remaining = [s for s in remaining if !(s in nc)]
  end
  for x in seed                                      # (posnegrestart) seed features first, in order, where supported
    sx = findfirst(==(x), Ic)
    (sx === nothing || !(sx in remaining)) && continue
    e = only(extend_all(m, st, depth(st), [x]))
    y, ok = update_odds(m, e, label, st.odds[end])
    ok || continue
    advance!(m, st, x, e, y)
    deleteat!(remaining, findfirst(==(sx), remaining))
  end
  order = m.opts.poolm > 0 && tk !== nothing ? get(m.pool, tk, nothing) : nothing
  (m.opts.tuplek > 0 || m.opts.tuplemin > 0) && isempty(seed) && tuple_step!(m, st, Ic, label, remaining, tk)
  if order === nothing
    nosingle = m.opts.pairsteps && m.opts.pairsfirst ? pair_covered(m, Ic, remaining) : NOSINGLE
    interleaves(m, tk) || while step_single!(m, st, Ic, label, remaining; nosingle) end   # (interleave: in chain_logodds)
  else
    dead = Set{Tuple{Int16,Int16}}()
    while step_pool!(m, st, Ic, label, remaining, order, dead) end
  end
  st, remaining
end

"""
Tuple first step (tuplek / tuplemin; user's design 2026-10-31): the pool is the features whose single step
|Δ log odds| reaches tau (and tuplemin), largest first, cut to the tuplek largest; of its supported triples (else
pairs, else singles) the one with the largest max_x l(X = x | tuple) -- each value's log odds after inserting the
members in pool order, as the chain would (unlabeled decisions: |l|, the larger of attached / not) -- joins the chain,
member by member.  Returns whether a tuple was added.
"""
function tuple_step!(m::ChainModel, st::ChainState, Ic, label, remaining, tk)
  depth(st) == 0 || return false
  x = st.odds[end]
  slots = [s for s in remaining if Ic[s] != ABSENT]
  E1 = extend_all(m, st, 0, present_items(Ic, slots))
  pool = Tuple{Float64,Int,Any,Float64}[]           # (single step, slot, entry, own odds)
  for (s, e) in zip(slots, E1)
    y, ok = update_odds(m, e, label, x)
    ok || continue
    d = abs(log(y) - log(x))
    d >= max(m.opts.tau, m.opts.tuplemin) && push!(pool, (d, s, e, y))
  end
  sort!(pool; by = p -> -p[1])
  m.opts.tuplek > 0 && length(pool) > m.opts.tuplek && resize!(pool, m.opts.tuplek)
  isempty(pool) && return false
  # each value's prior: the chain's own odds for the candidate's label, the type's prior for the others
  prior(l) = l == label || tk === nothing ? x : priorodds(m, lpriorkey(tk, l))
  # max_x l(X = x | members), inserting the members' entries in order, each smoothed toward the last
  function tscore(es)
    ls = label == 0 ? (UInt32(0),) : keys(es[end][2])
    top = -Inf                                      # (names apart from the enclosing function's: closures share them)
    for l in ls
      ol = prior(l)
      for e in es
        ol = first(update_odds(m, e, l, ol))
      end
      top = max(top, label == 0 ? abs(log(ol)) : log(ol))
    end
    top
  end
  branch(s) = ChainState(copy(s.keys), copy(s.views), copy(s.sizes), copy(s.members), copy(s.odds),
    Dict{UInt64,Float64}[], copy(s.cnt))
  n = length(pool)
  best, bi = -Inf, Int[]                            # bi: pool indices of the winner, its entries in pairs / triples
  bes = Any[]
  ytrip = Dict{Tuple{Int,Int},Tuple{Any,Float64,ChainState}}()   # supported pairs: (entry, own odds, state)
  for i in 1:n-1
    sa = branch(st)
    advance!(m, sa, Ic[pool[i][2]], pool[i][3], pool[i][4])
    js = collect(i+1:n)
    for (j, e) in zip(js, extend_all(m, sa, 1, UInt64[Ic[pool[j][2]] for j in js]))
      y, ok = update_odds(m, e, label, pool[i][4])
      ok || continue
      sab = branch(sa)
      advance!(m, sab, Ic[pool[j][2]], e, y)
      ytrip[(i, j)] = (e, y, sab)
    end
  end
  for ((i, j), (eab, yab, sab)) in sort!(collect(ytrip); by = first)
    ls = collect(j+1:n)
    isempty(ls) && continue
    for (l, e) in zip(ls, extend_all(m, sab, 2, UInt64[Ic[pool[l][2]] for l in ls]))
      _, ok = update_odds(m, e, label, yab)
      ok || continue
      sc = tscore(Any[pool[i][3], eab, e])
      sc > best && ((best, bi, bes) = (sc, [i, j, l], Any[pool[i][3], eab, e]))
    end
  end
  if isempty(bi)                                    # no supported triple: the best pair
    for ((i, j), (eab, _, _)) in sort!(collect(ytrip); by = first)
      sc = tscore(Any[pool[i][3], eab])
      sc > best && ((best, bi, bes) = (sc, [i, j], Any[pool[i][3], eab]))
    end
  end
  if isempty(bi)                                    # no supported pair: the best single
    for i in 1:n
      sc = tscore(Any[pool[i][3]])
      sc > best && ((best, bi, bes) = (sc, [i], Any[pool[i][3]]))
    end
  end
  rec = m.opts.poolsigma > 0 && m.opts.poolmode == :local
  o = x
  for (i, e) in zip(bi, bes)
    o = first(update_odds(m, e, label, o))
    advance!(m, st, Ic[pool[i][2]], e, o)
    rec && push!(st.alts, Dict{UInt64,Float64}())
    deleteat!(remaining, findfirst(==(pool[i][2]), remaining))
  end
  (d = DC_STATS[]) === nothing || (d[Symbol(:tuple, length(bi))] = get(d, Symbol(:tuple, length(bi)), 0.0) + 1)
  true
end

"""
:maxdev chains in rounds (one step per still-running candidate per round), so the ranking, and
with it the sampling cap, is known at each step.  Without sampling it gives what run_single does.
"""
function run_maxdev_rounds(m::ChainModel, I, labels, priors, typerecs; tk = nothing)
  K = length(I)
  nl = all(iszero, labels) ? 1 : K
  states = [ChainState(p, typerecs) for p in priors]
  rems = [tk === nothing ? collect(eachindex(I[c])) : admitted(m, tk, I[c]) for c in 1:K]
  running = trues(K)
  while any(running)
    cap = samplecap(m, [s.odds[end] for s in states]; nlabels = nl)
    for c in 1:K
      running[c] && (running[c] = step_single!(m, states[c], I[c], labels[c], rems[c]; cap))
    end
  end
  states, rems
end

"""
:maxdev chains where only the two leading candidates advance: every candidate's chain takes its
first step (that gives the ranking), then the two with the highest odds step in lockstep while the
rest stay frozen.  When a frozen candidate overtakes one of the two, it resumes its own chain (a
maxdev chain doesn't depend on the other candidates, so starting over would repeat it) and the
displaced one freezes.  Ends when both leaders can't add a feature; each chain is thus cut where
it dropped out of the top two.
"""
function run_top2(m::ChainModel, I, labels, priors, typerecs; tk = nothing)
  K = length(I)
  states = [ChainState(p, typerecs) for p in priors]
  rems = [tk === nothing ? collect(eachindex(I[c])) : admitted(m, tk, I[c]) for c in 1:K]
  done = [!step_single!(m, states[c], I[c], labels[c], rems[c]) for c in 1:K]
  while true
    lead = top2([s.odds[end] for s in states])
    active = [c for c in lead if c > 0 && !done[c]]
    isempty(active) && break
    for c in active
      done[c] = !step_single!(m, states[c], I[c], labels[c], rems[c])
    end
  end
  states, rems
end

"""
A contest chain between candidates a and b: from zero features, each step adds the slot (present
and supported for both) that most changes the log ratio of their odds,
|(log ya - log xa) - (log yb - log xb)|, each with its own value; it ends when no slot changes it
by `tau`.  Returns their final log odds.  No leftovers.
"""
function contest_chain(m::ChainModel, Ia, Ib, la, lb, pa, pb, typerecs, tk)
  sa, sb = ChainState(pa, typerecs), ChainState(pb, typerecs)
  rem = intersect(admitted(m, tk, Ia), admitted(m, tk, Ib))
  filter!(s -> Ia[s] != ABSENT && Ib[s] != ABSENT, rem)
  while !isempty(rem)
    ea = extend_all(m, sa, depth(sa), UInt64[Ia[s] for s in rem])
    eb = extend_all(m, sb, depth(sb), UInt64[Ib[s] for s in rem])
    xa, xb = sa.odds[end], sb.odds[end]
    best, bi, bya, byb = -Inf, 0, xa, xb
    for i in eachindex(rem)
      ya, oka = update_odds(m, ea[i], la, xa)
      yb, okb = update_odds(m, eb[i], lb, xb)
      (oka && okb) || continue
      sc = abs((log(ya) - log(xa)) - (log(yb) - log(xb)))
      sc > best && ((best, bi, bya, byb) = (sc, i, ya, yb))
    end
    (bi == 0 || best < m.opts.tau) && break
    advance!(m, sa, Ia[rem[bi]], ea[bi], bya)
    advance!(m, sb, Ib[rem[bi]], eb[bi], byb)
    deleteat!(rem, bi)
  end
  log(sa.odds[end]), log(sb.odds[end])
end

"""
Contest re-ranking (see ChainOpts.contest): after the usual scores, the top two are re-decided by a
contest chain on their odds ratio, their scores replaced by its log odds; if that leaves one of them
below a third candidate, the new top two contest again (from zero features), up to `maxrestarts`
times.  The others keep their scores, and the margin is then the contest's log ratio.
"""
function contest!(m::ChainModel, scores, I, labels, priors, typerecs, tk)
  length(scores) < 2 && return scores
  done = Set{Tuple{Int,Int}}()
  for _ in 0:m.opts.maxrestarts
    o = sortperm(scores; rev = true)
    pair = (min(o[1], o[2]), max(o[1], o[2]))
    pair in done && break                            # this pair has had its contest
    push!(done, pair)
    a, b = pair
    scores[a], scores[b] = contest_chain(m, I[a], I[b], labels[a], labels[b], priors[a], priors[b], typerecs, tk)
    length(scores) < 3 && break
    o2 = sortperm(scores; rev = true)
    Set(o2[1:2]) == Set(pair) && break               # both still lead: settled
  end
  scores
end

"""
:maxdev chains in target mode: the candidates' chains advance in step, one feature each per
round, so the ranking can be checked after every round (see `restart` in the header).
"""
function run_maxdev_target(m::ChainModel, I, labels, priors, typerecs; tk = nothing)
  K = length(I)
  target, restarts = nothing, 0
  while true
    states = [ChainState(p, typerecs) for p in priors]
    rems = [tk === nothing ? collect(eachindex(I[c])) : admitted(m, tk, I[c]) for c in 1:K]
    exhausting, restart_to = false, nothing
    while true
      x = [s.odds[end] for s in states]
      cap = samplecap(m, x; nlabels = all(iszero, labels) ? 1 : K)
      advanced = false
      for c in 1:K
        advanced |= step_single!(m, states[c], I[c], labels[c], rems[c];
          target = target === nothing ? nothing : target[c], exhaust = exhausting, cap)
      end
      advanced || break
      exhausting && continue
      y = [s.odds[end] for s in states]
      if K > 1 && restarts < m.opts.maxrestarts && top2(y) != top2(x) &&
         (target === nothing || top2(y) != top2(target))
        m.opts.restart == :immediate ? (restart_to = y; break) : (exhausting = true)
      end
    end
    exhausting && (restart_to = [s.odds[end] for s in states])
    restart_to === nothing && return states, rems
    target, restarts = restart_to, restarts + 1
  end
end

"""
chainshrink = k > 0: the chain's log odds with each step's size discounted by its support, the chain itself (which
features, in which order) unchanged.  Step j (records c1/c2 of its prefix, from odds o_{j-1}) contributes
log shrinkOdds(c1, c2, t_j, k) - log o_{j-1}, with t_j: :prev o_{j-1}; :gen o(x | prefix_{j-1} + the feature's nearest
generalization) (o_{j-1} if it has none or it's unsupported).  :prior: the final prefix's records shrunk toward the
prior, log shrinkOdds(c1, c2, prior, k).
"""
function chain_shrunk_logodds(m::ChainModel, st::ChainState, Ic, label, prior)
  k = m.opts.chainshrink
  depth(st) == 0 && return log(st.odds[end])
  if m.opts.chainsel <= 0 && m.opts.chainshrinkto == :prior
    c = get(st.cnt[end], label, (0.0, 0.0))
    return log(shrinkOdds(c[1], c[2], prior, k))
  end
  G = (m.opts.chainsel > 0 || m.opts.chainshrinkto == :gen) && !isempty(st.views[1].base) && length(st.members) == depth(st) ?
    get(m.generalize, m.recs[st.views[1].base[1]].typekey, nothing) : nothing
  lo = log(st.odds[1])
  for j in 1:depth(st)
    oprev = st.odds[j]
    c = get(st.cnt[j+1], label, (0.0, 0.0))
    t = oprev
    if G !== nothing && (s = findfirst(==(st.members[j]), Ic)) !== nothing
      gens = UInt64[Ic[g] for g in get(G, s, Int[]) if Ic[g] != ABSENT && Ic[g] != st.members[j]]
      if !isempty(gens)
        g = gens[argmin([get(m.itemcount, x, typemax(Int)) for x in gens])]   # the nearest (rarest) generalization
        e = only(extend_all(m, st, j - 1, [g]))
        y, ok = update_odds(m, e, label, oprev)
        ok && (t = y)
      end
    end
    if m.opts.chainsel > 0
      # selective (positive-part James-Stein): the step's log odds pulled toward the reference t by
      # w = max(0, 1 - z0^2 / z^2), z = (log o_j - log t) / SE, SE^2 = 1/(c1 + 1/2) + 1/(c2 + 1/2)
      d = log(st.odds[j+1]) - log(t)
      z2 = d^2 / (1 / (c[1] + 0.5) + 1 / (c[2] + 0.5))
      w = z2 <= m.opts.chainsel^2 ? 0.0 : 1 - m.opts.chainsel^2 / z2
      if (ds = DC_STATS[]) !== nothing
        kk = w == 0 ? :cs_zero : w < 0.9 ? :cs_part : :cs_keep
        ds[kk] = get(ds, kk, 0.0) + 1
      end
      lo += log(t) + w * d - log(oprev)
    else
      lo += log(shrinkOdds(c[1], c[2], t, k)) - log(oprev)
    end
  end
  lo
end

"""
Two chains for one candidate (dualchain; user's design 2026-10-30): P for positive evidence, N for negative.
:marginal -- P's candidates are the features raising o(x) alone (vs the prior), N's those lowering it; each grows by
maxdev within its own context (|Δ| >= tau).  :inchain -- P takes the largest positive in-chain Δ (>= tau), N the
largest negative.  A feature whose Δ, when added, has the opposite sign of its marginal effect is added to the other
chain too (a duplicate).  Returns (P, N, Δ of P's duplicates, Δ of N's duplicates, the slots used).
"""
function run_dual(m::ChainModel, Ic, label, prior, typerecs; tk = nothing)
  rem0 = tk === nothing ? collect(eachindex(Ic)) : admitted(m, tk, Ic)
  if tk !== nothing && haskey(m.nochain, tk)
    nc = m.nochain[tk]
    rem0 = [s for s in rem0 if !(s in nc)]
  end
  rem0 = [s for s in rem0 if Ic[s] != ABSENT]
  root = ChainState(prior, typerecs)
  d0 = Dict{Int,Float64}()                          # each slot's marginal effect
  for (s, e) in zip(rem0, extend_all(m, root, 0, present_items(Ic, rem0)))
    y, ok = update_odds(m, e, label, prior)
    d0[s] = ok ? log(y) - log(prior) : 0.0
  end
  P, N = ChainState(prior, typerecs), ChainState(prior, typerecs)
  inP, inN = Set{Int}(), Set{Int}()
  dup = [0.0, 0.0]                                  # Δ of the duplicate copies in P, N
  function addto!(C, inC, s, k)                     # add slot s to chain C (k: 1 = P, 2 = N) as a duplicate, if supported
    s in inC && return
    e = only(extend_all(m, C, depth(C), [Ic[s]]))
    y, ok = update_odds(m, e, label, C.odds[end])
    ok || return
    dup[k] += log(y) - log(C.odds[end])
    advance!(m, C, Ic[s], e, y)
    push!(inC, s)
  end
  function grow!(C, sg, inC, inO, O, k)
    while true
      cs = [s for s in rem0 if !(s in inC) && !(s in inO) &&
        (m.opts.dualchain == :inchain || (sg > 0 ? d0[s] > 0 : d0[s] < 0))]
      isempty(cs) && return
      x = C.odds[end]
      best, bs, by, be = -Inf, 0, x, nothing
      for (s, e) in zip(cs, extend_all(m, C, depth(C), present_items(Ic, cs)))
        y, ok = update_odds(m, e, label, x)
        ok || continue
        d = log(y) - log(x)
        sc = m.opts.dualchain == :inchain ? sg * d : abs(d)
        sc > best && ((best, bs, by, be) = (sc, s, y, e))
      end
      (bs == 0 || best < m.opts.tau) && return
      d = log(by) - log(x)
      turned = sign(d) != sign(d0[bs]) && d0[bs] != 0
      if turned && m.opts.dualturn == :refine
        # refine: does it turn in the other chain too?  If not, it goes there only; if so, to both
        eo = only(extend_all(m, O, depth(O), [Ic[bs]]))
        yo, oko = update_odds(m, eo, label, O.odds[end])
        if oko && sign(log(yo) - log(O.odds[end])) == sign(d0[bs])
          advance!(m, O, Ic[bs], eo, yo)
          push!(inO, bs)
          continue
        end
      end
      advance!(m, C, Ic[bs], be, by)
      push!(inC, bs)
      turned && addto!(O, inO, bs, 3 - k)              # it turned: the other chain too
    end
  end
  grow!(P, 1, inP, inN, N, 1)
  grow!(N, -1, inN, inP, P, 2)
  P, N, dup[1], dup[2], union(inP, inN)
end

"The dualchain score of a candidate: P's and N's changes from the prior combined by dualcombine (log odds)."
function dual_score(m::ChainModel, P, N, dupP, dupN, prior)
  lp = log(prior)
  dP, dN = log(P.odds[end]) - lp, log(N.odds[end]) - lp
  c = m.opts.dualcombine
  lp + (c == :sum ? dP + dN : c == :maxabs ? (abs(dP) >= abs(dN) ? dP : dN) :
    c == :weighted ? dP + m.opts.dualw * dN : dP + dN - dupP - dupN)
end

"""
Log-odds of a finished chain plus the corrections from its leftover slots.  Each leftover's
correction comes from the deepest prefix that supports it; the leftovers still open at a depth
are counted together, and the corrections are added in slot order.
"""
function chain_logodds(m::ChainModel, st::ChainState, Ic, label, prior, remaining; cap = EXACT)
  lo = m.opts.chainshrink > 0 || m.opts.chainsel > 0 ? chain_shrunk_logodds(m, st, Ic, label, prior) : log(st.odds[end])
  if !isempty(m.noleft) && !isempty(st.views[1].base) && !m.opts.interleave   # slots kept out of the leftovers
    skip = get(m.noleft, m.recs[st.views[1].base[1]].typekey, nothing)
    skip === nothing || (remaining = [s for s in remaining if !(s in skip)])
  end
  if m.opts.leftovers in (:none, :greedy)
    corr = m.opts.leftovers == :none ? 0.0 : greedy_leftovers(m, st, Ic, label, prior, remaining)
    m.opts.interleave && (lo = log(st.odds[end]))     # (interleave: the chain grew with the leftovers)
    PARTS_HOOK[] === nothing || PARTS_HOOK[](lo, corr)
    α = isempty(m.leftalpha) || isempty(st.views[1].base) ? 1.0 : get(m.leftalpha, m.recs[st.views[1].base[1]].typekey, 1.0)
    return lo + α * corr
  end
  slots = [s for s in remaining if Ic[s] != ABSENT]
  lowest = m.opts.leftovers == :backoff1 && depth(st) > 0 ? 1 : 0
  depths = m.opts.leftovers == :marginal ? (0:0) : (depth(st):-1:lowest)
  corr = zeros(length(slots))
  open_ = collect(eachindex(slots))
  # climbing slots under :none or :specific are handled apart from the usual depth search
  climbing = [t for t in open_ if haskey(m.vclimb, Ic[slots[t]])]
  if m.opts.climbleft != :deepest && !isempty(climbing)
    filter!(t -> !(t in climbing), open_)
    m.opts.climbleft == :specific && for t in climbing
      perk = Dict(k => climblevels(m, st, k, Ic[slots[t]]) for k in depths)
      done = false
      for L in 1:length(first(values(perk))), k in depths   # finest ring first, then the deepest prefix for it
        o = st.odds[k+1]
        y, ok = update_odds(m, (NOENTRY[1], perk[k][L]), label, o)
        ok && (corr[t] = log(y) - log(o); done = true; break)
      end
      done || push!(open_, t)                          # (unsupported: stays open, so no correction)
    end
    sort!(open_)
    pending_none = m.opts.climbleft == :none ? climbing : Int[]
  else
    pending_none = Int[]
  end
  unresolved = copy(pending_none)
  for k in depths
    isempty(open_) && break
    es = extend_all(m, st, k, UInt64[Ic[slots[t]] for t in open_]; cap)
    o = st.odds[k+1]
    still = Int[]
    for (t, e) in zip(open_, es)
      y, ok = update_odds(m, e, label, o)
      ok ? (corr[t] = log(y) - log(o)) : push!(still, t)
    end
    open_ = still
  end
  # climbing slots handled apart and found unsupported are still in open_; :none slots never get a correction
  for t in eachindex(slots)
    (t in open_ || t in unresolved) || (lo += corr[t])
  end
  lo
end

"""
Sum of the leftovers' corrections under :greedy (see Leftovers in the header).  Each leftover X
grows a context S (as a chain state, with a twin state for S∪X so both sets' records are at hand);
o(S∪f) is smoothed toward o(S) and o(S∪f∪X) toward o(S∪f), as along a chain.
"""
function greedy_leftovers(m::ChainModel, st::ChainState, Ic, label, prior, remaining)
  IN_GREEDY[] = true
  try
    _greedy_leftovers(m, st, Ic, label, prior, remaining)
  finally
    IN_GREEDY[] = false
  end
end

"""
A leftover's relevance (see :greedy in the header): the largest of |λ_X| (its effect alone) and
|λ(a, X)| over the chain's features a, for `label`; λ as in the lattice, from exact counts.
"""
function leftover_relevance(m::ChainModel, st::ChainState, X, label, prior, base)
  S0 = ChainState(prior, base)
  eX = only(extend_all(m, S0, 0, [X]))
  yX, okX = update_odds(m, eX, label, prior)
  oX = okX ? yX : prior
  r = abs(log(oX) - log(prior))
  for a in st.members
    S1 = ChainState(prior, base)
    ea = only(extend_all(m, S1, 0, [a]))
    ya, oka = update_odds(m, ea, label, prior)
    oka || continue
    advance!(m, S1, a, ea, ya)
    e = prior * (ya / prior) * (oX / prior)            # what a and X predict separately
    y, ok = update_odds(m, only(extend_all(m, S1, 1, [X])), label, e)
    ok && (r = max(r, abs(log(y) - log(e))))
  end
  r
end

"""
Slots among `remaining` that are redundant given the finished chain: supported at its final
prefix (`mincount` records with the prefix) but changing its odds by less than `tau`.
"""
function redundant_slots(m::ChainModel, st::ChainState, Ic, label, remaining)
  first.(redundant_divergence(m, st, Ic, label, remaining))
end

"""
The redundant slots with their divergence d = |log o(c | prefix, r) - log o(c | prefix)| (< tau).
poolmarginal (:leftover, :drop): only slots that are informative on their own, |log o(c | r) - log o(c)| >= tau,
so the pool holds features the chain makes redundant rather than ones that never said much; the others
are pushed to `weak` (poolkeepgen with poolsigma: generalizations of chain forms stay in the pool).
"""
function redundant_divergence(m::ChainModel, st::ChainState, Ic, label, remaining; weak = nothing)
  slots = [s for s in remaining if Ic[s] != ABSENT]
  out = Tuple{Int,Float64}[]
  isempty(slots) && return out
  x = st.odds[end]
  for (s, e) in zip(slots, extend_all(m, st, depth(st), UInt64[Ic[s] for s in slots]))
    y, ok = update_odds(m, e, label, x)
    d = abs(log(y) - log(x))
    ok && d < m.opts.tau && push!(out, (s, d))
  end
  if m.opts.poolmarginal != :off && !isempty(out)
    gen = Set{Int}()
    if m.opts.poolkeepgen && m.opts.poolsigma > 0
      G = get(m.generalize, m.recs[st.views[1].base[1]].typekey, nothing)
      for w in st.members
        sw = findfirst(==(w), Ic)
        G === nothing || sw === nothing || union!(gen, get(G, sw, Int[]))
      end
    end
    p = st.odds[1]
    S0 = ChainState(p, st.views[1].base)
    es = extend_all(m, S0, 0, UInt64[Ic[s] for (s, _) in out])
    keep = map(zip(out, es)) do ((s, _), e)
      s in gen && return true
      y, ok = update_odds(m, e, label, p)
      ok && abs(log(y) - log(p)) >= m.opts.tau
    end
    weak === nothing || append!(weak, [s for ((s, _), k) in zip(out, keep) if !k])
    out = out[keep]
  end
  out
end

"""
Soft pool weights (poolsigma): for each chain form w with generalization slots g in the pool (upos, xpos,
classes, ... of the same token), how well g stands in for w here: the chain's odds rebuilt with w
replaced by g, d_g = |log o(c | prefix with g for w) - log o(c | prefix)|, weight exp(-d_g / sigma)
(0 if the replaced chain isn't supported).  poolscale :z (whole mode): d_g in standard errors of the
chain's final log odds, SE = sqrt(1/n_chosen + 1/n_not) from its final prefix's counts for the
label (raw counts; the smoothing toward earlier stages is ignored).
"""
function generalization_weights!(poolw, m::ChainModel, st::ChainState, Ic, label, prior, base, pool)
  isempty(base) && return poolw
  G = get(m.generalize, m.recs[base[1]].typekey, nothing)
  G === nothing && return poolw
  o0 = st.odds[end]
  if m.opts.poolmode == :local                       # at w's own step: o(prefix before w + g) vs o(prefix before w + w)
    for (i, w) in enumerate(st.members)
      sw = findfirst(==(w), Ic)
      (sw === nothing || !haskey(G, sw) || i > length(st.alts) || isempty(st.alts[i])) && continue
      for sg in G[sw]
        g = Ic[sg]
        (g == ABSENT || !(g in pool)) && continue
        yg = get(st.alts[i], g, nothing)
        poolw[g] = yg === nothing ? 0.0 : exp(-abs(log(yg) - log(st.odds[i+1])) / m.opts.poolsigma)
      end
    end
    return poolw
  end
  scale = 1.0
  if m.opts.poolscale == :z && !isempty(st.members)  # the chain's final prefix counts, rebuilt (cached entries)
    S = ChainState(prior, base)
    e = nothing
    for f in st.members
      e = only(extend_all(m, S, depth(S), [f]))
      y, _ = update_odds(m, e, label, S.odds[end])
      advance!(m, S, f, e, y)
    end
    (c1, c2) = get(e[2], label, (0.0, 0.0))
    scale = c1 > 0 && c2 > 0 ? sqrt(1 / c1 + 1 / c2) : 1.0
  end
  for w in st.members
    sw = findfirst(==(w), Ic)
    (sw === nothing || !haskey(G, sw)) && continue
    for sg in G[sw]
      g = Ic[sg]
      (g == ABSENT || !(g in pool)) && continue
      S = ChainState(prior, base)
      ok = true
      for f in st.members
        x = f == w ? g : f
        e = only(extend_all(m, S, depth(S), [x]))
        y, ok = update_odds(m, e, label, S.odds[end])
        ok || break
        advance!(m, S, x, e, y)
      end
      poolw[g] = ok ? exp(-abs(log(S.odds[end]) - log(o0)) / scale / m.opts.poolsigma) : 0.0
    end
  end
  poolw
end

"""
A leftover X's correction with context `ctx` (in the order chosen): log o(c | ctx, X) - log o(c | ctx),
each stage smoothed toward the one before as in _greedy_leftovers.  Empty ctx: X's marginal correction.
"""
function context_correction(m::ChainModel, X, ctx, label, prior, base)
  S, SX = ChainState(prior, base), ChainState(prior, base)
  eX = only(extend_all(m, S, 0, [X]))
  yX, okX = update_odds(m, eX, label, prior)
  advance!(m, SX, X, eX, yX)
  corr = okX ? log(yX) - log(prior) : 0.0
  for f in ctx
    eS, eSX = only(extend_all(m, S, depth(S), [f])), only(extend_all(m, SX, depth(SX), [f]))
    y1, ok1 = update_odds(m, eS, label, S.odds[end])
    y2, ok2 = leftover_update(m, eS, eSX, label, y1)
    (ok1 && ok2) || break
    advance!(m, S, f, eS, y1); advance!(m, SX, f, eSX, y2)
    corr = log(y2) - log(y1)
  end
  corr
end

"""
o(x | S∪E) from the entries for context S (`eS`) and S∪E (`eSX`), given y1 = o(x | S) as smoothed:
:odds -- the odds in S∪E's records smoothed toward y1 (one count, smoothOdds); :evidence -- Bayes' rule,
y1 times the likelihood ratio P(E | x,S) / P(E | ¬x,S), each rate from E's odds among the x (¬x)
records of S smoothed toward E's odds in all of S.  Supported as update_odds (S∪E's records).
"""
function leftover_update(m::ChainModel, eS, eSX, label::UInt32, y1::Float64)
  if m.opts.leftsmooth == :odds
    m.opts.shrinkleft > 0 || return update_odds(m, eSX, label, y1)
    (c1, c2) = get(eSX[2], label, (0.0, 0.0))
    c1 + c2 < max(m.opts.mincount, 1) && return y1, false
    return shrinkOdds(c1, c2, y1, m.opts.shrinkleft), true
  end
  (k1, k2) = get(eSX[2], label, (0.0, 0.0))
  k1 + k2 < max(m.opts.mincount, 1) && return y1, false
  (n1, n2) = get(eS[2], label, (0.0, 0.0))
  rest = n1 + n2 - k1 - k2
  rest <= 0 && return y1, true                    # E in every record of S: no evidence either way
  oE = (k1 + k2) / rest
  prob(o) = isinf(o) ? 1.0 : o / (1 + o)
  ks = m.opts.shrinkleft > 0 ? m.opts.shrinkleft : m.opts.shrink
  sm(a, b) = ks > 0 ? shrinkOdds(a, b, oE, ks) : smoothOdds(a, b, oE, smoothing(m, a + b, n1 + n2))   # (parent: S's records)
  px, pnx = prob(sm(k1, n1 - k1)), prob(sm(k2, n2 - k2))
  (px > 0 && pnx > 0) || return y1, true
  y1 * px / pnx, true
end

"Does context `chosen` hold each chain feature itself or one of its generalizations (ctxmode :cover)?"
function covers(st::ChainState, Ic, G, chosen)
  for w in st.members
    w in chosen && continue
    sw = findfirst(==(w), Ic)
    (sw !== nothing && any(sg -> Ic[sg] != ABSENT && Ic[sg] in chosen, get(G, sw, Int[]))) || return false
  end
  true
end

"""
Leftover X's correction with a generalized chain as its context (ctxmode :backoff / :limited).  Each
chain feature has a ladder: itself, then its generalizations on the same token (G: xpos, upos,
basepos, classes, suffixes; rarer -- more specific -- first), then (:backoff only) dropped.  Start from
the chain; while context ∪ X is unsupported, take the one step up a ladder that changes the chain's
odds least, |log o(x | context) - log o(x | chain)|.  Returns (log o(x | C ∪ X) - log o(x | C), C's items)
for the first supported context C, or nothing (no supported context: :limited, or nothing left to try).
"""
function backoff_correction(m::ChainModel, st::ChainState, Ic, G, X, label, prior, base; mode = m.opts.ctxmode)
  ladders = Vector{Vector{UInt64}}()
  for w in st.members
    sw = findfirst(==(w), Ic)
    gens = sw === nothing ? UInt64[] : UInt64[Ic[sg] for sg in get(G, sw, Int[]) if Ic[sg] != ABSENT && Ic[sg] != w && Ic[sg] != X]
    sort!(gens; by = g -> get(m.itemcount, g, typemax(Int)))
    push!(ladders, [w; gens])
  end
  top(L) = mode == :backoff ? length(L) + 1 : length(L)          # length + 1: dropped
  rung = ones(Int, length(ladders))
  ochain = st.odds[end]
  # the context's state, final entry and odds (nothing if some step is unsupported)
  function build(r)
    S, e = ChainState(prior, base), nothing
    for (i, L) in enumerate(ladders)
      r[i] > length(L) && continue
      x = L[r[i]]
      e = only(extend_all(m, S, depth(S), [x]))
      y, ok = update_odds(m, e, label, S.odds[end])
      ok || return nothing
      advance!(m, S, x, e, y)
    end
    S, e
  end
  cur = build(rung)
  while true
    if cur !== nothing
      S, e = cur
      eX = only(extend_all(m, S, depth(S), [X]))
      if e === nothing                                # every feature dropped: X's marginal correction
        yX, okX = update_odds(m, eX, label, prior)
        okX && ctxsupported(m, eX, label) && return (log(yX) - log(prior), UInt64[])
      else
        y2, ok2 = leftover_update(m, e, eX, label, S.odds[end])
        ok2 && ctxsupported(m, eX, label) && return (log(y2) - log(S.odds[end]), copy(S.members))
      end
    end
    best, bi, bst = Inf, 0, nothing
    for i in eachindex(ladders)
      rung[i] < top(ladders[i]) || continue
      rung[i] += 1
      c = build(rung)
      rung[i] -= 1
      c === nothing && continue
      d = abs(log(c[1].odds[end]) - log(ochain))
      d < best && ((best, bi, bst) = (d, i, c))
    end
    bi == 0 && return nothing
    rung[bi] += 1
    cur = bst
  end
end

"""
Leftovers as a second chain over generalized contexts (ctxmode :accum).  The context starts as the chain;
each feature in it has a ladder (itself, its generalizations on the same token -- rarer first --, then
dropped) and a rung that only ever goes up.  Each round, every remaining leftover X is scored against the
context generalized only as far as X needs: walking the generalization path -- from the current rungs,
repeatedly the one step up a ladder that keeps the context's odds closest to the running estimate T (log
o(x | chain) plus the corrections taken so far) -- until context ∪ X is supported; its correction is
log o(x | C ∪ X) - log o(x | C) (leftover_update).  The X with the largest |correction| is taken: the
rungs it needed stay, X joins the context with its own ladder, T grows by the correction, and the next
round begins.  Leftovers already in the context (as a generalization of a feature) are redundant;
unsupported ones (even with everything dropped) are left out.  Returns the summed correction.
ciorder (:slot / :value, decision types with a ci table): each walk step instead moves the context feature with
the smallest dependence (see `dependence` below) whose step keeps the context supported.
telescope: on each token's ladder (form, then its generalizations, rarer first) a leftover waits while a coarser
rung is pending, is left out if a finer rung is in the chain, and is conditioned on its nearest coarser rung already
counted (chain or taken), kept in every context it's scored in (the anchor); so a token's rungs add up to its form's
effect once.  Leftovers with no supported context are dropped, letting their finer rungs go ahead.
branches (> 1): a leftover the current context doesn't support also gets the first supported context along the
paths that start with the 2nd, 3rd, ... best first step (walked as the main path); the corrections are combined by
`combine` (see combine_contexts), and the context carried forward is their union (each feature's finest rung).
"""
function accum_leftovers(m::ChainModel, st::ChainState, Ic, G, rel::Vector{Int}, label, prior, base;
  chainrem = nothing, skip = Set{Int}())
  function ladder(x, slot)                          # (items, their slots): x, then its generalizations, rarer first
    gens = slot === nothing ? Int[] : Int[sg for sg in get(G, slot, Int[]) if Ic[sg] != ABSENT && Ic[sg] != x]
    sort!(gens; by = sg -> get(m.itemcount, Ic[sg], typemax(Int)))
    [x; UInt64[Ic[sg] for sg in gens]], [something(slot, 0); gens]
  end
  lad = [ladder(w, findfirst(==(w), Ic)) for w in st.members]
  ladders, lslots = first.(lad), last.(lad)
  D = m.opts.cidrop > 0 && !isempty(base) ? get(m.ci, m.recs[base[1]].typekey, nothing) : nothing
  # may feature i (at rung r[i]) be dropped?  Only if some other kept feature A makes it redundant: D(A→B) < cidrop;
  # and (accumfloor) only while more than minctx features are kept
  function maydrop(r, i)
    m.opts.accumfloor && count(j -> r[j] <= length(ladders[j]), eachindex(ladders)) <= m.opts.minctx && return false
    D === nothing && return true
    sb = lslots[i][r[i]]
    sb == 0 && return false
    any(j -> j != i && r[j] <= length(ladders[j]) && lslots[j][r[j]] > 0 && D[lslots[j][r[j]], sb] < m.opts.cidrop, eachindex(ladders))
  end
  # ciorder: feature i's (at rung r[i]) dependence -- :slot, the mean of D[A, B] over all other slots A; :value, the
  # mean over the context's other features A at their values a of D(a → B) (D[A, B] if a is rare); smallest first
  C = m.opts.ciorder != :off && !isempty(base) ? get(m.ci, m.recs[base[1]].typekey, nothing) : nothing
  if C !== nothing
    V = get(m.civ, m.recs[base[1]].typekey, Dict{Tuple{Int16,UInt64,Int16},Float32}())
    gavg = [(v = Float64[C[a, b] for a in axes(C, 1) if a != b && !isnan(C[a, b])]; isempty(v) ? Inf : sum(v) / length(v)) for b in axes(C, 2)]
  end
  function dependence(r, i)
    sb = lslots[i][r[i]]
    sb == 0 && return Inf
    m.opts.ciorder == :slot && return gavg[sb]
    tot, k = 0.0, 0
    for j in eachindex(ladders)
      (j == i || r[j] > length(ladders[j])) && continue
      sa = lslots[j][r[j]]
      (sa == 0 || sa == sb) && continue
      d = get(V, (Int16(sa), ladders[j][r[j]], Int16(sb)), C[sa, sb])
      isnan(d) && continue
      tot += d; k += 1
    end
    k == 0 ? gavg[sb] : tot / k
  end
  rung = ones(Int, length(ladders))
  T = log(st.odds[end])
  total_ = 0.0
  pending = copy(rel)
  items(r) = unique!(UInt64[L[r[i]] for (i, L) in enumerate(ladders) if r[i] <= length(L)])   # (a generalization may equal another feature)
  cache = Dict{Tuple{Vector{Int},UInt64},Any}()
  function build(r, A = UInt64(0))                  # (state, final entry) of the context (+ anchor A), or nothing if unsupported
    get!(cache, (copy(r), A)) do
      S, e = ChainState(prior, base), nothing
      xs = items(r)
      A == 0 || A in xs || push!(xs, A)
      for x in xs
        e = only(extend_all(m, S, depth(S), [x]))
        y, ok = update_odds(m, e, label, S.odds[end]; pn = m.opts.smoothn == :parent ? parentn(m, S, label) : NaN)
        ok || return nothing
        advance!(m, S, x, e, y)
      end
      (S, e)
    end
  end
  # X's correction in context r (+ anchor A): (correction, its variance, (state, entry), X's entry), or nothing if
  # unsupported.  Variance: sum of 1/(count + 1/2) over X's cells (chosen, not) with the context and without it
  function xcorr(r, X, A = UInt64(0))
    c = build(r, A)
    c === nothing && return nothing
    S, e = c
    eX = only(extend_all(m, S, depth(S), [X]))
    corr = nothing
    if e === nothing
      yX, okX = update_odds(m, eX, label, prior; pn = m.opts.smoothn == :parent ? parentn(m, S, label) : NaN)
      okX && m.opts.shrinkleft > 0 && (yX = shrinkOdds(get(eX[2], label, (0.0, 0.0))..., prior, m.opts.shrinkleft))
      okX && ctxsupported(m, eX, label) && (corr = log(yX) - log(prior))
    else
      y2, ok2 = leftover_update(m, e, eX, label, S.odds[end])
      ok2 && ctxsupported(m, eX, label) && (corr = log(y2) - log(S.odds[end]))
    end
    corr === nothing && return nothing
    (k1, k2) = get(eX[2], label, (0.0, 0.0))
    v = 1 / (k1 + 0.5) + 1 / (k2 + 0.5)
    if e !== nothing
      (n1, n2) = get(e[2], label, (0.0, 0.0))
      v += 1 / (max(n1 - k1, 0.0) + 0.5) + 1 / (max(n2 - k2, 0.0) + 0.5)
    end
    (corr, v, c, eX)
  end
  # telescope: each token's ladder (form, then its generalizations, rarer first); a leftover's ancestors are the
  # coarser rungs of its ladder, its finer rungs the ones before it
  ancestors, finer = Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}()
  if m.opts.telescope
    for (f, gs) in G
      Ic[f] == ABSENT && continue
      tl = [f; sort!(Int[g for g in gs if Ic[g] != ABSENT && Ic[g] != Ic[f]]; by = g -> get(m.itemcount, Ic[g], typemax(Int)))]
      for (k, sl) in enumerate(tl)
        ancestors[sl] = tl[k+1:end]
        finer[sl] = tl[1:k-1]
      end
    end
    chainset = Set(st.members)                      # a leftover with a finer rung in the chain adds nothing
    filter!(s -> !any(f -> Ic[f] in chainset, get(finer, s, Int[])), pending)
  end
  taken = Set{UInt64}()
  # interleave (chainrem: the chain's remaining slots; user's design 2026-11-01): the chain grows inside this loop.  Each
  # round the best chain step (as step_single! would take it) competes with the best leftover -- the slots the current
  # prefix doesn't support, past the relevance gate -- scored by its (discounted) correction: by size (stepscore :abs),
  # or (:posneg) raises first, chain step vs positive correction, then, if neither reaches tau, the largest lowerings.
  # A chain winner joins the prefix and the context (its own ladder, rung 1) and moves T by its step; a leftover is
  # taken as usual.  When nothing reaches tau the chain stops: the end-of-chain filtering (redundant, weak) runs and
  # the remaining leftovers are taken as usual.
  chainon = chainrem !== nothing
  ss = STEPSCORE_OVERRIDE[] === nothing ? m.opts.stepscore : STEPSCORE_OVERRIDE[]
  pk = nothing                                       # the chain's peeked step (stale once the chain moves)
  function relevant(sx)
    m.opts.relevance <= 0 || leftover_relevance(m, st, Ic[sx], label, prior, base) >= m.opts.relevance
  end
  # the unsupported slots join the leftovers while the chain runs (relevance against the chain so far)
  function refresh!()
    pk === nothing || return
    pk = step_single!(m, st, Ic, label, chainrem; peek = true)
    for sx in pk.unsup
      (sx in pending || sx in skip || Ic[sx] in taken || Ic[sx] in items(rung)) && continue
      relevant(sx) && push!(pending, sx)
    end
  end
  # the chain stops: its remaining supported slots go through the usual end-of-chain filtering; the rest are leftovers
  function finish!()
    chainon = false
    gone = Int[]
    if m.opts.redundant
      weak = Int[]
      red = first.(redundant_divergence(m, st, Ic, label, chainrem; weak))
      gone = m.opts.poolmarginal == :drop ? union(red, weak) : red
    end
    for sx in chainrem
      (Ic[sx] == ABSENT || sx in gone || sx in pending || sx in skip || Ic[sx] in taken || Ic[sx] in items(rung)) && continue
      relevant(sx) && push!(pending, sx)
    end
  end
  function chainstep!(neg::Bool)
    s_, y_, e_ = neg ? (pk.nbs, pk.nby, pk.nbe) : (pk.bs, pk.by, pk.be)
    advance!(m, st, Ic[s_], e_, y_)
    m.opts.poolsigma > 0 && m.opts.poolmode == :local && push!(st.alts, pk.alts === nothing ? Dict{UInt64,Float64}() : pk.alts)
    deleteat!(chainrem, findfirst(==(s_), chainrem))
    T += log(y_) - log(pk.x)
    (li, ls) = ladder(Ic[s_], s_)
    push!(ladders, li); push!(lslots, ls); push!(rung, 1)
    filter!(sx -> !(Ic[sx] in items(rung)), pending)
    (d = DC_STATS[]) === nothing || (d[:il_chain] = get(d, :il_chain, 0.0) + 1)
    pk = nothing
  end
  # the anchor X is conditioned on: its nearest ancestor already counted (in the chain or taken); 0 if none
  function anchor(sx)
    for a in get(ancestors, sx, Int[])
      (Ic[a] in taken || Ic[a] in st.members) && return Ic[a]
    end
    UInt64(0)
  end
  # X waits while a coarser rung of its ladder is still pending
  blocked(sx) = any(a -> a in pending, get(ancestors, sx, Int[]))
  # (trace) each leftover's score in every round it was scored, the context then growing as leftovers join it:
  # slot -> [(round, running estimate T, context, its odds, X's records there, correction, the score it competed with)]
  hist = ACCUM_TRACE[] === nothing ? nothing : Dict{Int,Vector{Any}}()
  rnd = 0
  while true
    rnd += 1
    chainon && refresh!()
    isempty(pending) && !chainon && break
    # the generalization path from the current rungs (the same for every X), built as far as needed
    path = Any[(copy(rung), build(rung))]
    function step!(path)
      r = path[end][1]
      if C !== nothing                              # ciorder: the least dependent feature that can step up
        movable = [i for i in eachindex(ladders) if r[i] <= length(ladders[i]) && !(r[i] == length(ladders[i]) && !maydrop(r, i))]
        for i in sort!(movable; by = i -> dependence(r, i))
          r2 = copy(r); r2[i] += 1
          c = build(r2)
          c === nothing && continue
          push!(path, (r2, c))
          return true
        end
        return false
      end
      best, br, bc = Inf, nothing, nothing
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue      # (already dropped)
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue   # dropping B: only if redundant given a kept A
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        d = abs(log(c[1].odds[end]) - T)
        d < best && ((best, br, bc) = (d, r2, c))
      end
      br === nothing && return false
      push!(path, (br, bc))
      true
    end
    # accumby :dep: the move (feature i one rung up, or dropped) giving up the least of X's dependence on the context,
    # ρ(X, item now) - ρ(X, item after); ties by the running-estimate rule.  :depodds: the move with the smallest
    # imp * (dependence lost) + |log o(context after) - T|, imp = |X's marginal correction| (both in log odds)
    function depstep!(p, X, imp = 0.0)
      r = p[end][1]
      best, br, bc = (Inf, Inf), nothing, nothing
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        L = ladders[i]
        loss = depq(m, base, X, L[r[i]], label, prior) - (r2[i] <= length(L) ? depq(m, base, X, L[r2[i]], label, prior) : 0.0)
        d = abs(log(c[1].odds[end]) - T)
        sc = m.opts.accumby == :depodds ? (imp * loss + d, 0.0) : (loss, d)
        sc < best && ((best, br, bc) = (sc, r2, c))
      end
      br === nothing && return false
      push!(p, (br, bc))
      true
    end
    # branches (accum branches > 1): the paths starting with the 2nd, 3rd, ... best first step from the current
    # rungs (the main path starts with the best), each then walked as the main path; built when first needed
    bpaths = nothing
    function branchpaths()
      bpaths === nothing || return bpaths
      bpaths = Any[]
      r = path[1][1]
      moves = Tuple{Float64,Vector{Int},Any}[]
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        push!(moves, (abs(log(c[1].odds[end]) - T), r2, c))
      end
      sort!(moves; by = first)
      for (_, r2, c) in moves[2:min(end, m.opts.branches)]
        push!(bpaths, Any[path[1], (r2, c)])
      end
      bpaths
    end
    # accumby :marg: each leftover its own path, the move keeping log o(x | context) closest to log o(x | X)
    function margstep!(p, LX)
      r = p[end][1]
      best, br, bc = Inf, nothing, nothing
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        d = abs(log(c[1].odds[end]) - LX)
        d < best && ((best, br, bc) = (d, r2, c))
      end
      br === nothing && return false
      push!(p, (br, bc))
      true
    end
    # accumby :mindev / :mindevsup (user's design): each leftover its own path; among the moves whose context
    # supports X, the one with the smallest diff = |log o(x | X, C') - log o(x | C')| (X is then explained away
    # if diff < mindevtau); while no move supports X: :mindev the running-estimate rule, :mindevsup the move
    # keeping the most records with X.  (leftmin2 = k: supported means >= k chosen and not-chosen records.)
    function mdstep!(p, X, A)
      r = p[end][1]
      best, br, bc = Inf, nothing, nothing
      fb, fr, fc = Inf, nothing, nothing
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        res = xcorr(r2, X, A)
        if res !== nothing
          abs(res[1]) < best && ((best, br, bc) = (abs(res[1]), r2, c))
        else
          f = m.opts.accumby == :mindev ? abs(log(c[1].odds[end]) - T) :
            -sum(get(only(extend_all(m, c[1], depth(c[1]), [X]))[2], label, (0.0, 0.0)))
          f < fb && ((fb, fr, fc) = (f, r2, c))
        end
      end
      br === nothing && ((br, bc) = (fr, fc))
      br === nothing && return false
      push!(p, (br, bc))
      true
    end
    # accumby :mintw (user's design): each leftover W its own path; two independent proposals per step -- the move
    # with the smallest ΔT = |log o(x|C') - T| (the running-estimate rule) and, among moves where W has records
    # (>= 1) in both the current context C and C', the one with the smallest ΔW = |log o(x|W,C) - log o(x|W,C')|
    # (W's records smoothed toward the context's odds); the one proposing the smaller change is taken.
    twnote = Ref{Any}(nothing)                       # (trace) the last step's proposals
    function wlogodds(c, X)
      S = c[1]
      k = get(only(extend_all(m, S, depth(S), [X]))[2], label, (0.0, 0.0))
      k[1] + k[2] > 0 || return nothing
      log(smoothOdds(k[1], k[2], S.odds[end]))
    end
    function twstep!(p, X)
      r = p[end][1]
      w0 = p[end][2] === nothing ? nothing : wlogodds(p[end][2], X)
      bt, brt, bct = Inf, nothing, nothing
      bw, brw, bcw = Inf, nothing, nothing
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        dt = abs(log(c[1].odds[end]) - T)
        dt < bt && ((bt, brt, bct) = (dt, r2, c))
        if w0 !== nothing
          w1 = wlogodds(c, X)
          w1 === nothing || (dw = abs(w1 - w0); dw < bw && ((bw, brw, bcw) = (dw, r2, c)))
        end
      end
      brt === nothing && return false
      useW = brw !== nothing && bw < bt
      twnote[] = (by = useW ? :W : :T, dT = bt, dW = brw === nothing ? nothing : bw)
      if (d = DC_STATS[]) !== nothing
        k = useW ? :tw_w : :tw_t
        d[k] = get(d, k, 0.0) + 1
        brw !== nothing && !useW && (d[:tw_wlost] = get(d, :tw_wlost, 0.0) + 1)
      end
      push!(p, useW ? (brw, bcw) : (brt, bct))
      true
    end
    # accumby :bfs (user's design): all ways of backing off once from the current context, then twice, ... (up to
    # BFS_LEVELS), until some context supports W; among those, the smallest last-step ΔW = |log o(x|W,C) -
    # log o(x|W,C')| over the moves into C' from a level-(k-1) context where W has records; contexts with no such
    # parent after them, by ΔT = |log o(x|C') - T|.  Returns the chosen rungs, or nothing (then the odds walk).
    function bfsctx(X, A)
      r0 = path[1][1]
      prev = Set{Vector{Int}}([r0])
      for lev in 1:BFS_LEVELS
        cur = Set{Vector{Int}}()
        for r in prev, i in eachindex(ladders)
          r[i] <= length(ladders[i]) || continue
          r[i] == length(ladders[i]) && !maydrop(r, i) && continue
          r2 = copy(r); r2[i] += 1
          push!(cur, r2)
        end
        isempty(cur) && return nothing
        bsc, bctx = (Inf, Inf), nothing
        for r in cur
          c = build(r)
          c === nothing && continue
          xcorr(r, X, A) === nothing && continue
          w1 = wlogodds(c, X)
          dw = Inf
          for i in eachindex(r)
            r[i] > r0[i] || continue
            rp = copy(r); rp[i] -= 1
            rp in prev || continue
            cp = build(rp)
            cp === nothing && continue
            w0 = wlogodds(cp, X)
            (w0 === nothing || w1 === nothing) && continue
            dw = min(dw, abs(w1 - w0))
          end
          sc = (dw, abs(log(c[1].odds[end]) - T))
          sc < bsc && ((bsc, bctx) = (sc, r))
        end
        if bctx !== nothing
          (d = DC_STATS[]) === nothing || (k = Symbol("bfs_l", lev); d[k] = get(d, k, 0.0) + 1)
          return bctx
        end
        prev = cur
      end
      (d = DC_STATS[]) === nothing || (d[:bfs_fail] = get(d, :bfs_fail, 0.0) + 1)
      nothing
    end
    # accumby :look1red / :look1ind (user's design): greedy, one level at a time -- if some one-step backoff gives a
    # context that supports W, take the one with the smallest ΔW (ties by ΔT); otherwise back off the feature y that
    # :look1red is the most redundant, the smallest D(a → y) over the other kept features a (CI table, value-level
    # where known); :look1ind is the most independent of W, the smallest |Yule's Q|(W, y) (odds ratio nearest 1)
    # (:look1nonred: the opposite -- back off the least redundant feature first, the largest min D(a → y))
    # (:look1int: as :look1red, but among supported moves the smallest interaction I = |corr(W | C) - corr(W | C')|,
    #  corr = W's records' log odds smoothed toward the context's minus the context's -- how much the move changes
    #  what W contributes, not W's estimate; a feature W implies keeps |its own effect| and so stays)
    CL = m.opts.accumby in (:look1red, :look1nonred, :look1int) && !isempty(base) ? get(m.ci, m.recs[base[1]].typekey, nothing) : nothing
    VL = CL === nothing ? nothing : get(m.civ, m.recs[base[1]].typekey, Dict{Tuple{Int16,UInt64,Int16},Float32}())
    function redundancy(r, i)                         # min over kept a of D(a → feature i)
      CL === nothing && return 0.0
      sb = lslots[i][r[i]]
      sb == 0 && return Inf
      d = Inf
      for j in eachindex(ladders)
        (j == i || r[j] > length(ladders[j])) && continue
        sa = lslots[j][r[j]]
        (sa == 0 || sa == sb) && continue
        v = get(VL, (Int16(sa), ladders[j][r[j]], Int16(sb)), CL[sa, sb])
        isnan(v) || (d = min(d, v))
      end
      d
    end
    # W's (log odds, correction) in context c: its records smoothed toward the context's odds (one count), or with
    # lookwsmooth = k its likelihood ratio, P(W|x) / P(W|¬x), each rate smoothed by k records toward W's pooled rate
    function wlog2(c, X)
      S, e = c
      kx = get(only(extend_all(m, S, depth(S), [X]))[2], label, (0.0, 0.0))
      kx[1] + kx[2] > 0 || return nothing
      ks = m.opts.lookwsmooth
      if ks > 0 && e !== nothing
        n = get(e[2], label, (0.0, 0.0))
        pE = (kx[1] + kx[2]) / max(n[1] + n[2], 1.0)
        lr = log(max((kx[1] + ks * pE) / (n[1] + ks), 1e-12)) - log(max((kx[2] + ks * pE) / (n[2] + ks), 1e-12))
        return (log(S.odds[end]) + lr, lr, kx[1] + kx[2])
      end
      w = log(smoothOdds(kx[1], kx[2], S.odds[end]))
      (w, w - log(S.odds[end]), kx[1] + kx[2])
    end
    lookmetric = m.opts.accumby == :look1int ? :int : m.opts.lookw
    function lookstep!(p, X, A)
      r = p[end][1]
      w0 = p[end][2] === nothing ? nothing : wlog2(p[end][2], X)
      kb = m.opts.lookwback
      bsc, brr, bcc = (Inf, Inf), nothing, nothing      # supported one-step moves
      fsc, frr, fcc = Inf, nothing, nothing             # otherwise
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        if xcorr(r2, X, A) !== nothing
          w1 = wlog2(c, X)
          # lookwback: the interaction backs off (hard) or blends (soft) toward the pairwise one over all records,
          # |corr(W | a) - corr(W | a')| (a' = the next rung, or nothing when a is dropped), by W's records n in the
          # thinner context -- also where W has none in the current context
          ip = kb > 0 && lookmetric != :dw ? abs(pair_corr(m, base, X, ladders[i][r[i]], label, prior) -
            pair_corr(m, base, X, r2[i] <= length(ladders[i]) ? ladders[i][r2[i]] : UInt64(0), label, prior)) : NaN
          dw = if w1 === nothing || (w0 === nothing && isnan(ip))
            Inf
          else
            n = w0 === nothing ? 0.0 : min(w0[3], w1[3])
            dI = w0 === nothing ? ip : abs(w1[2] - w0[2])     # the interaction (W's correction)
            if !isnan(ip) && w0 !== nothing
              dI = m.opts.lookwsoft ? (n / (n + kb)) * dI + (kb / (n + kb)) * ip : (n < kb ? ip : dI)
            end
            if (ds = DC_STATS[]) !== nothing && !isnan(ip)
              kk = w0 === nothing || (!m.opts.lookwsoft && n < kb) ? :lw_pair : :lw_ctx
              ds[kk] = get(ds, kk, 0.0) + 1
            end
            dW = w0 === nothing ? Inf : abs(w1[1] - w0[1])          # ΔW (W's estimate)
            lookmetric == :dw ? dW : lookmetric == :int ? dI : lookmetric == :min ? min(dW, dI) : max(dW, dI)
          end
          sc = (dw, abs(log(c[1].odds[end]) - T))
          sc < bsc && ((bsc, brr, bcc) = (sc, r2, c))
        elseif brr === nothing
          f = m.opts.accumby in (:look1red, :look1int) ? redundancy(r, i) :
            m.opts.accumby == :look1nonred ? (d = redundancy(r, i); isinf(d) ? Inf : -d) :
            depq(m, base, X, ladders[i][r[i]], label, prior; measure = :yule)
          f < fsc && ((fsc, frr, fcc) = (f, r2, c))
        end
      end
      if (d = DC_STATS[]) !== nothing
        k = brr !== nothing ? :look_supp : frr !== nothing ? :look_dep : :look_none
        d[k] = get(d, k, 0.0) + 1
      end
      brr === nothing && ((brr, bcc) = (frr, fcc))
      brr === nothing && return false
      push!(p, (brr, bcc))
      true
    end
    # (leftwback) the starting context's log odds: the chain and the taken leftovers at their most specific (T if unsupported)
    l0 = if m.opts.leftwback > 0 || m.opts.accumby == :wsel
      c1 = build(ones(Int, length(ladders)))
      c1 === nothing || c1[2] === nothing ? T : log(c1[1].odds[end])
    else
      T
    end
    # accumby :wsel (user's design 2026-10-31): each leftover its own path; each move by the largest combination (min or
    # sum) of w_back = exp(-(l(C') - l_0)^2 / N) and w_ovl (1 - the leftover's largest overlap with a counted feature C'
    # leaves out)
    # w of context rungs r (combined as wselcomb; sum normalized to [0, 1])
    function wctx(r, c, X)
      N = m.opts.leftwback > 0 ? m.opts.leftwback : 1.0
      li = c[2] === nothing ? log(prior) : log(c[1].odds[end])
      wb = exp(-(li - l0)^2 / N)
      wo = overlap_weight(m, base, X, setdiff(union(st.members, taken), items(r)), label, prior)
      m.opts.wselcomb == :min ? min(wb, wo) : (wb + wo) / 2
    end
    function wstep!(p, X, A)
      r = p[end][1]
      bsc, brr, bcc = -Inf, nothing, nothing            # by the weights alone
      csc, crr, ccc = -Inf, nothing, nothing            # (wcorr) by |w * corr| among moves supporting X
      for i in eachindex(ladders)
        r[i] <= length(ladders[i]) || continue
        r[i] == length(ladders[i]) && !maydrop(r, i) && continue
        r2 = copy(r); r2[i] += 1
        c = build(r2)
        c === nothing && continue
        w = wctx(r2, c, X)
        w > bsc && ((bsc, brr, bcc) = (w, r2, c))
        if m.opts.wselmove == :wcorr
          res = xcorr(r2, X, A)
          res === nothing || (s = abs(w * res[1]); s > csc && ((csc, crr, ccc) = (s, r2, c)))
        end
      end
      crr === nothing || ((brr, bcc) = (crr, ccc))
      brr === nothing && return false
      push!(p, (brr, bcc))
      true
    end
    bestX, bestc, bestk, bestr = 0, 0.0, 0, Int[]
    lcands = Tuple{Int,Float64,Int,Vector{Int}}[]     # (interleave) every scored leftover this round
    # accumby :depodds: X's marginal correction |log o(x | X) - log o(x)| (0 if unsupported)
    function margsigned(X)                          # log o(x | X) - log o(x) (0 if unsupported)
      eX = only(extend_all(m, ChainState(prior, base), 0, [X]))
      y, ok = update_odds(m, eX, label, prior)
      ok ? log(y) - log(prior) : 0.0
    end
    function margimpact(X)
      eX = only(extend_all(m, ChainState(prior, base), 0, [X]))
      y, ok = update_odds(m, eX, label, prior)
      ok ? abs(log(y) - log(prior)) : 0.0
    end
    dead = Int[]                                      # (telescope) leftovers with no supported context
    tries = ACCUM_TRACE[] === nothing ? nothing : Dict{Int,Vector{Any}}()   # (trace: each X's walk along the path)
    explained = Int[]                                # (mindev + explainpool) explained-away leftovers this round
    for (pi_, sx) in enumerate(pending)
      X = Ic[sx]
      X in items(rung) && continue                  # already in the context (as a generalization): redundant
      m.opts.telescope && blocked(sx) && continue
      A = m.opts.telescope ? anchor(sx) : UInt64(0)
      tries === nothing || (tries[sx] = Any[])
      # accumby :dep: X walks its own path, each step the move losing the least of X's dependence on the context
      xpath = m.opts.accumby == :odds ? path : Any[path[1]]
      imp = m.opts.accumby == :depodds ? margimpact(X) : 0.0
      bfsr = m.opts.accumby == :bfs && xcorr(path[1][1], X, A) === nothing ? bfsctx(X, A) : nothing
      bfsr === nothing || (xpath = Any[(bfsr, build(bfsr))])     # (bfs: the chosen context, taken as is)
      xstep! = m.opts.accumby in (:odds, :bfs) ? (m.opts.accumby == :bfs && bfsr !== nothing ? (p -> false) : step!) : m.opts.accumby in (:dep, :depodds) ? (p -> depstep!(p, X, imp)) :
        m.opts.accumby == :marg ? (p -> margstep!(p, log(prior) + margsigned(X))) :
        m.opts.accumby in (:mintw, :mintwc) ? (p -> twstep!(p, X)) : m.opts.accumby in (:look1red, :look1ind, :look1nonred, :look1int) ? (p -> lookstep!(p, X, A)) :
        m.opts.accumby == :wsel ? (p -> wstep!(p, X, A)) :
        (p -> mdstep!(p, X, A))
      k = 1
      k0, xc, xk, xv = 0, 0.0, 0, 0.0                 # (accumwin: the first supported step, and the smallest |correction| since)
      while true
        res = xpath[k][2] === nothing ? nothing : xcorr(xpath[k][1], X, A)
        if xpath[k][2] !== nothing && tries !== nothing   # context, its odds and records, X's records with it, X's correction (or nothing)
          S, e = xpath[k][2]
          eX = res === nothing ? only(extend_all(m, S, depth(S), [X])) : res[4]
          push!(tries[sx], (ctx = items(xpath[k][1]), odds = e === nothing ? prior : S.odds[end],
            n = e === nothing ? nothing : get(e[2], label, (0.0, 0.0)), nX = get(eX[2], label, (0.0, 0.0)),
            corr = res === nothing ? nothing : res[1], how = k == 1 ? nothing : twnote[]))
        end
        corr = res === nothing ? nothing : res[1]
        if corr !== nothing && m.opts.accumby in (:mindev, :mindevsup) && abs(corr) < m.opts.mindevtau   # explained away
          m.opts.explainpool && push!(explained, sx)
          break
        end
        if corr !== nothing && (m.opts.driftsigma > 0 || m.opts.drifttau > 0 || m.opts.driftstep > 0)
          # drift: how far this context has moved from the running estimate, and how many ladder steps it took
          S, e = res[3]
          d = abs((e === nothing ? log(prior) : log(S.odds[end])) - T)
          steps = sum(xpath[k][1] .- 1)
          if m.opts.drifttau > 0 && d > m.opts.drifttau
            corr = nothing
          else
            m.opts.driftsigma > 0 && (corr *= exp(-d / m.opts.driftsigma))
            m.opts.driftstep > 0 && (corr *= m.opts.driftstep ^ steps)
          end
        end
        if corr !== nothing && m.opts.accumby == :mintwc
          # mintwc: from the first supported context, keep backing off by the ΔW move (W supported in C') while
          # W's estimate stays within mindevtau of its value there; W's correction comes from the last such context
          wref = log(xpath[k][2][1].odds[end]) + corr
          xc, xk, xv = corr, k, res[2]
          while true
            r = xpath[end][1]
            best, br, bc, bres = Inf, nothing, nothing, nothing
            for i in eachindex(ladders)
              r[i] <= length(ladders[i]) || continue
              r[i] == length(ladders[i]) && !maydrop(r, i) && continue
              r2 = copy(r); r2[i] += 1
              c = build(r2)
              c === nothing && continue
              res2 = xcorr(r2, X, A)
              res2 === nothing && continue
              d = abs(log(c[1].odds[end]) + res2[1] - wref)
              d < best && ((best, br, bc, bres) = (d, r2, c, res2))
            end
            (br === nothing || best >= m.opts.mindevtau) && break
            push!(xpath, (br, bc))
            xc, xk, xv = bres[1], length(xpath), bres[2]
            (d = DC_STATS[]) === nothing || (d[:tw_cont] = get(d, :tw_cont, 0.0) + 1)
            if tries !== nothing
              S, e = bc
              push!(tries[sx], (ctx = items(br), odds = e === nothing ? prior : S.odds[end], n = e === nothing ? nothing : get(e[2], label, (0.0, 0.0)),
                nX = get(bres[4][2], label, (0.0, 0.0)), corr = bres[1], how = (by = :cont, dT = nothing, dW = best)))
            end
          end
          break
        end
        if corr !== nothing
          k0 == 0 && (k0 = k)
          (xk == 0 || abs(corr) < abs(xc)) && ((xc, xk, xv) = (corr, k, res[2]))
          k - k0 >= m.opts.accumwin && break         # (accumwin 0: the first supported context, as before)
        end
        k == length(xpath) && !xstep!(xpath) && break   # nothing left to generalize
        k += 1
      end
      if xk == 0
        m.opts.telescope && push!(dead, sx)
        continue
      end
      xr = xpath[xk][1]
      # accumgreedy (user's design 2026-11-02): the context drifted too far from T -- a greedy context instead (grown from
      # the chain and the taken leftovers by dependence, as greedyby = :dep); the shared context's rungs stay as they are
      if m.opts.accumgreedy > 0
        cx = build(xr)
        lc = cx === nothing || cx[2] === nothing ? log(prior) : log(cx[1].odds[end])
        if abs(lc - T) > m.opts.accumgreedy
          g = greedy_dep_context(m, base, X, UInt64[st.members; collect(taken)], label, prior, m.opts.minctx, m.opts.maxctx)
          (d = DC_STATS[]) === nothing || (d[g === nothing ? :ag_kept : :ag_greedy] = get(d, g === nothing ? :ag_kept : :ag_greedy, 0.0) + 1)
          g === nothing || ((xc, xr) = (g[1], copy(path[1][1])))
        else
          (d = DC_STATS[]) === nothing || (d[:ag_accum] = get(d, :ag_accum, 0.0) + 1)
        end
      end
      if m.opts.branches > 1 && xk > 1                # X needed generalizing: its contexts along the other branches too
        ctxs = Any[(xr, xc, xv)]
        for bp in branchpaths()
          j = 2
          while true
            res = xcorr(bp[j][1], X, A)
            if res !== nothing
              any(c -> c[1] == bp[j][1], ctxs) || push!(ctxs, (bp[j][1], res[1], res[2]))
              break
            end
            j == length(bp) && !step!(bp) && break
            j += 1
          end
        end
        if length(ctxs) > 1
          xc = combine_contexts(m.opts.combine, ctxs, r -> xcorr(r, X, A))
          xr = reduce((a, b) -> min.(a, b), first.(ctxs))   # carried forward: the union (finest rung of each feature)
        end
      end
      # depweight: scaled by 1 - X's strongest dependence on a counted feature (chain, taken) outside its context
      m.opts.depweight && (xc *= depweight(m, base, X, setdiff(union(st.members, taken), items(xr)), label, prior))
      # (wsel + wselweight) the correction times w of its chosen context
      if m.opts.accumby == :wsel && m.opts.wselweight
        cw = build(xr)
        cw === nothing || (xc *= wctx(xr, cw, X))
      end
      # leftover weights (user's design 2026-10-31): backoff -- how far the chosen context's log odds drifted from the
      # starting context's; overlap -- the largest share of X's effect it shares with a counted feature outside its context
      if m.opts.leftwback > 0 || m.opts.leftwovl
        wb, wo = 1.0, 1.0
        if m.opts.leftwback > 0
          cx = build(xr)
          li = cx === nothing || cx[2] === nothing ? log(prior) : log(cx[1].odds[end])
          wb = exp(-(li - l0)^2 / m.opts.leftwback)
        end
        m.opts.leftwovl && (wo = overlap_weight(m, base, X, setdiff(union(st.members, taken), items(xr)), label, prior))
        xc *= m.opts.leftwback > 0 && m.opts.leftwovl ?
          (m.opts.leftwcomb == :prod ? wb * wo : m.opts.leftwcomb == :min ? min(wb, wo) : max(wb, wo)) : wb * wo
      end
      # X's correction in its chosen context; the X with the largest |correction| is taken
      # (accumpick :signed: the biggest raise instead of the largest |correction|)
      if hist !== nothing && !isempty(tries[sx])
        tr_ = something(findlast(t_ -> t_.corr !== nothing, tries[sx]), length(tries[sx]))
        push!(get!(hist, sx, Any[]), (round = rnd, T = T, ctx = tries[sx][tr_].ctx, odds = tries[sx][tr_].odds,
          nX = tries[sx][tr_].nX, corr = tries[sx][tr_].corr, score = xc))
      end
      (bestX == 0 || (m.opts.accumpick == :signed ? xc > bestc : abs(xc) > abs(bestc))) && ((bestX, bestc, bestk, bestr) = (pi_, xc, xk, xr))
      chainon && push!(lcands, (pi_, xc, xk, xr))
    end
    # explainpool: the explained-away leftovers join the context (correction 0) and leave the pending list
    function poolexplained!()
      for sx in explained
        sx in pending || continue
        (li, ls) = ladder(Ic[sx], sx)
        push!(ladders, li); push!(lslots, ls); push!(rung, 1)
        (d = DC_STATS[]) === nothing || (d[:pooled] = get(d, :pooled, 0.0) + 1)
      end
      filter!(s -> !(s in explained), pending)
    end
    if chainon                                        # interleave: the chain's step against the best leftover
      tau = m.opts.tau
      cs = pk.bs == 0 ? -Inf : pk.best
      if ss == :posneg
        lp = isempty(lcands) ? nothing : lcands[argmax([c[2] for c in lcands])]
        lsc = lp === nothing ? -Inf : lp[2]
        if max(cs, lsc) < tau                         # nothing raises the odds by tau: the largest lowering
          ln = isempty(lcands) ? nothing : lcands[argmin([c[2] for c in lcands])]
          ncs, nlsc = pk.nbs == 0 ? -Inf : pk.nbest, ln === nothing ? -Inf : -ln[2]
          if max(ncs, nlsc) < tau
            finish!(); continue
          end
          ncs >= nlsc ? (chainstep!(true); continue) : ((bestX, bestc, bestk, bestr) = ln)
        else
          cs >= lsc ? (chainstep!(false); continue) : ((bestX, bestc, bestk, bestr) = lp)
        end
      else
        lsc = bestX == 0 ? -Inf : abs(bestc)
        max(cs, lsc) < tau && (finish!(); continue)
        cs >= lsc && (chainstep!(false); continue)
      end
      (d = DC_STATS[]) === nothing || (d[:il_left] = get(d, :il_left, 0.0) + 1)
    end
    if bestX == 0
      isempty(explained) || (poolexplained!(); continue)
      isempty(dead) && break
      filter!(s -> !(s in dead), pending)             # (telescope) their finer rungs may go ahead
      continue
    end
    sx = pending[bestX]
    # accumstop (user's design 2026-11-02): the best leftover's context drifted more than theta from T -- it and every
    # remaining leftover are discarded
    if m.opts.accumstop > 0
      cx = build(bestr)
      lc = cx === nothing || cx[2] === nothing ? log(prior) : log(cx[1].odds[end])
      if abs(lc - T) > m.opts.accumstop
        (d = DC_STATS[]) === nothing || (d[:as_stop] = get(d, :as_stop, 0.0) + 1; d[:as_dropped] = get(d, :as_dropped, 0.0) + length(pending))
        break
      end
      (d = DC_STATS[]) === nothing || (d[:as_taken] = get(d, :as_taken, 0.0) + 1)
    end
    ACCUM_TRACE[] === nothing || ACCUM_TRACE[](sx, T, tries[sx], bestk, bestc, get(hist, sx, Any[]))
    # (dcstats) the taken leftover's context size: features kept (0: against the prior, its marginal)
    (d = DC_STATS[]) === nothing || (k_ = Symbol(:acc_ctx_, min(count(j -> bestr[j] <= length(ladders[j]), eachindex(ladders)), 5));
      d[k_] = get(d, k_, 0.0) + 1)
    DC_STATS[] === nothing || dc_record!(m, base, Ic[sx], setdiff(union(st.members, taken), items(bestr)), label, prior, bestc)
    LEFTOVER_TRACE[] === nothing || LEFTOVER_TRACE[](sx, true, [something(findfirst(==(f), Ic), 0) for f in items(bestr)], bestc)
    m.opts.accumreset || (rung = copy(bestr))         # (accumreset: the context stays at its most specific)
    (li, ls) = ladder(Ic[sx], sx)
    push!(ladders, li); push!(lslots, ls); push!(rung, 1)
    push!(taken, Ic[sx])
    total_ += bestc
    T += bestc
    deleteat!(pending, bestX)
    chainrem === nothing || filter!(!=(sx), chainrem)
    filter!(s -> s != sx, explained)
    poolexplained!()
    filter!(s -> !(s in dead), pending)
    filter!(s -> !(Ic[s] in items(rung)), pending)  # leftovers now in the context are redundant
  end
  total_
end

"""
A greedy context for leftover X (accumgreedy): grown from `avail` (chain features, taken leftovers) one feature at a
time -- the one X depends on most (depq, depmeasure), among those keeping S ∪ f ∪ X supported -- as greedy contexts
with greedyby = :dep are; (correction, context) once no feature can join, or nothing if under `needmin` features.
"""
function greedy_dep_context(m::ChainModel, base, X::UInt64, avail::Vector{UInt64}, label::UInt32, prior, needmin::Int, needmax::Int)
  S, SX = ChainState(prior, base), ChainState(prior, base)
  eX = only(extend_all(m, S, 0, [X]))
  yX, okX = update_odds(m, eX, label, prior)
  advance!(m, SX, X, eX, yX)
  corr = okX ? log(yX) - log(prior) : 0.0
  chosen = UInt64[]
  while length(chosen) < needmax
    fs = UInt64[f for f in avail if !(f in chosen) && f != X]
    isempty(fs) && break
    eS, eSX = extend_all(m, S, depth(S), fs), extend_all(m, SX, depth(SX), fs)
    oS = S.odds[end]
    bs, bi, by1, by2, bcorr = -Inf, 0, 0.0, 0.0, 0.0
    for (i, f) in enumerate(fs)
      y1, ok1 = update_odds(m, eS[i], label, oS)
      y2, ok2 = leftover_update(m, eS[i], eSX[i], label, y1)
      (ok1 && ok2 && ctxsupported(m, eSX[i], label)) || continue
      sc = depq(m, base, X, f, label, prior)
      sc > bs && ((bs, bi, by1, by2, bcorr) = (sc, i, y1, y2, log(y2) - log(y1)))
    end
    bi == 0 && break
    advance!(m, S, fs[bi], eS[bi], by1)
    advance!(m, SX, fs[bi], eSX[bi], by2)
    push!(chosen, fs[bi])
    corr = bcorr
  end
  length(chosen) >= needmin ? (corr, chosen) : nothing
end

const DEPQ = Dict{Tuple{UInt64,UInt64,UInt32},Float64}()
const DEP_MEASURES = (:yule, :pmi, :pmipos, :mi, :oddspmi, :moi, :logor)
# cache tags per measure (high bits of the label)
const DEP_TAGS = Dict(:yule => 0x00000000, :pmi => 0x40000000, :pmipos => 0x80000000, :mi => 0x10000000,
  :oddspmi => 0x20000000, :moi => 0x30000000, :logor => 0x50000000)

"""
ρ(E, a): how strongly leftover E and feature a go together, label-blind -- |Yule's Q| of their 2×2 table over the
decision type's records for `label` (chosen + not), with 1/2 added to each cell: 1 when one implies the other, 0 when
independent.  depmeasure :pmi -- log p(E,a) / (p(E) p(a)) (0 when independent, < 0 when they avoid each other);
:pmipos -- max(pmi, 0).  :mi, :oddspmi, :moi, :logor (user 2026-11-02) -- step_crit's one-count-smoothed scores on
the E x a table (E in the role of the choice): mutual information, |odds PMI|, mutual odds information, |log odds
ratio| (|Yule's Q| = tanh(|log OR| / 2) ranks as |log OR|, but its differences don't).  Cached (symmetric).
"""
function depq(m::ChainModel, base, E::UInt64, a::UInt64, label::UInt32, prior; measure = m.opts.depmeasure)
  (E == a || isempty(base)) && return 1.0
  tag = DEP_TAGS[measure]
  key = E < a ? (E, a, label ⊻ tag) : (a, E, label ⊻ tag)
  r = get(DEPQ, key, nothing)
  r === nothing || return r
  S0 = ChainState(prior, base)
  ea, eE = extend_all(m, S0, 0, [a, E])
  S1 = ChainState(prior, base)
  advance!(m, S1, a, ea, prior)
  eaE = only(extend_all(m, S1, 1, [E]))
  n(e) = sum(get(e[2], label, (0.0, 0.0)))
  (c, oc) = get(m.prior, lpriorkey(m.recs[base[1]].typekey, label), (0.0, 0.0))
  n11 = n(eaE)
  n10, n01 = n(eE) - n11, n(ea) - n11
  n00 = c + oc - n11 - n10 - n01
  length(DEPQ) >= MAXMEMO && empty!(DEPQ)
  if measure in (:mi, :oddspmi, :moi, :logor)     # E present / absent against a present / absent
    nE, N = max(n11, 0.0) + max(n10, 0.0), max(n11, 0.0) + max(n10, 0.0) + max(n01, 0.0) + max(n00, 0.0)
    return DEPQ[key] = step_crit(measure, nE, N - nE, max(n11, 0.0), max(n01, 0.0))
  end
  if measure != :yule                           # pointwise mutual information, 1/2 added to each cell
    N = max(n11, 0.0) + max(n10, 0.0) + max(n01, 0.0) + max(n00, 0.0) + 2.0
    pmi = log((max(n11, 0.0) + 0.5) * N / ((max(n11, 0.0) + max(n10, 0.0) + 1.0) * (max(n11, 0.0) + max(n01, 0.0) + 1.0)))
    return DEPQ[key] = measure == :pmipos ? max(pmi, 0.0) : pmi
  end
  p, q = (n11 + 0.5) * (max(n00, 0.0) + 0.5), (max(n10, 0.0) + 0.5) * (max(n01, 0.0) + 0.5)
  DEPQ[key] = abs(p - q) / (p + q)
end

"""
Double-counting counter (accum and greedy leftovers; on when DC_STATS[] is a Dict): for each leftover X taken with a
nonzero correction, q = the largest |Yule's Q|(X, a) over counted features a (chain, earlier leftovers) outside
X's context -- the evidence X may repeat.  Keys: :dc_n leftovers, :dc_hi those with q >= DC_HI, :dc_corr
Σ|corr|, :dc_mass Σ|corr|·q.
"""
const DC_STATS = Ref{Union{Nothing,Dict{Symbol,Float64}}}(nothing)
const DC_HI = 0.5
function dc_record!(m::ChainModel, base, X, counted, label, prior, corr)
  (d = DC_STATS[]) === nothing && return
  corr == 0 && return
  q = maximum((depq(m, base, X, a, label, prior; measure = :yule) for a in counted); init = 0.0)
  d[:dc_n] = get(d, :dc_n, 0.0) + 1
  d[:dc_hi] = get(d, :dc_hi, 0.0) + (q >= DC_HI)
  d[:dc_corr] = get(d, :dc_corr, 0.0) + abs(corr)
  d[:dc_mass] = get(d, :dc_mass, 0.0) + abs(corr) * q
  nothing
end

const PAIRCORR = Dict{Tuple{UInt64,UInt64,UInt32},Float64}()
"""
W's correction given only a, over all the decision type's records: log o(x | W, a) - log o(x | a) (a = 0: given
nothing, log o(x | W) - log o(x)); NaN if W isn't supported there.  (lookwback: the interaction's pairwise backoff.)
"""
function pair_corr(m::ChainModel, base, W::UInt64, a::UInt64, label::UInt32, prior)
  isempty(base) && return NaN
  key = (W, a, label)
  r = get(PAIRCORR, key, nothing)
  r === nothing || return r
  length(PAIRCORR) >= MAXMEMO && empty!(PAIRCORR)
  S = ChainState(prior, base)
  x = prior
  if a != 0 && a != W
    ea = only(extend_all(m, S, 0, [a]))
    ya, oka = update_odds(m, ea, label, prior)
    oka || return PAIRCORR[key] = NaN
    advance!(m, S, a, ea, ya)
    x = ya
  end
  eW = only(extend_all(m, S, depth(S), [W]))
  y, ok = update_odds(m, eW, label, x)
  PAIRCORR[key] = ok ? log(y) - log(x) : NaN
end

"""
leftwovl: 1 - X's largest overlap with a feature in `counted`, overlap(X, a) = (ΔX + Δa - ΔXa) / min(|ΔX|, |Δa|) with Δ
changes in log odds from the prior over all records (ΔXa = Δa + corr(X | a)), clamped to [0, 1]; 1 if none is known.
"""
function overlap_weight(m::ChainModel, base, X, counted, label, prior)
  dx = pair_corr(m, base, X, UInt64(0), label, prior)
  (isnan(dx) || dx == 0) && return 1.0
  w = 1.0
  for a in counted
    da = pair_corr(m, base, a, UInt64(0), label, prior)
    cxa = pair_corr(m, base, X, a, label, prior)
    (isnan(da) || isnan(cxa) || da == 0) && continue
    ov = (dx - cxa) / min(abs(dx), abs(da))          # (ΔX + Δa - ΔXa, with ΔXa = Δa + corr(X | a))
    w = min(w, clamp(1 - ov, 0.0, 1.0))
  end
  w
end

"depweight: 1 - the largest ρ(X, a) over features `counted` (0 if X goes with one completely)."
depweight(m::ChainModel, base, X, counted, label, prior) =
  1.0 - maximum((depq(m, base, X, a, label, prior) for a in counted); init = 0.0)

"""
One correction from a leftover's corrections in several contexts `ctxs` (rung vectors with (correction, variance)):
:avg -- inverse-variance weighted mean; :ie -- inclusion-exclusion over the contexts' meets (exact if X's
correction is additive in the context features), Σ over nonempty subsets J of (-1)^(|J|+1) corr(meet of J), the
meet of contexts being each feature's coarser rung (`xcorr(r)` scores one); :mix -- the two weighted by their
inverse variances (the inclusion-exclusion one's variance: the sum of its terms').  An unsupported meet (rare: a
meet is coarser) falls back to :avg.
"""
function combine_contexts(mode::Symbol, ctxs, xcorr)
  w = [1 / c[3] for c in ctxs]
  avg = sum(w .* [c[2] for c in ctxs]) / sum(w)
  mode == :avg && return avg
  B = length(ctxs)
  ie, vie = 0.0, 0.0
  for J in 1:(2^B - 1)
    sel = [b for b in 1:B if (J >> (b - 1)) & 1 == 1]
    if length(sel) == 1
      c, v = ctxs[sel[1]][2], ctxs[sel[1]][3]
    else
      res = xcorr(reduce((a, b) -> max.(a, b), [ctxs[b][1] for b in sel]))
      res === nothing && return avg
      c, v = res[1], res[2]
    end
    ie += (-1)^(length(sel) + 1) * c
    vie += v
  end
  mode == :ie && return ie
  vavg = 1 / sum(w)
  λ = vavg / (vavg + vie)                            # the inclusion-exclusion estimate's share
  λ * ie + (1 - λ) * avg
end

"""
Pairwise conditional independence for decision type `tk` (attachment-style records: one per candidate):
D[A, B] = how much feature slot B still changes o(x) once A's value is known -- slot_effects' split-half
measure with A as the gate, averaged over A's values weighted by their records -- from a sample of
`sample` records.  D[A, B] ≈ 0: o(x | A, B) ≈ o(x | A) for the values seen.  NaN rows: types without
attachment-style records.
"""
function ci_table(m::ChainModel, tk::UInt64; sample = 200_000)
  recs = get(m.typerecs, tk, Int32[])
  isempty(recs) && return nothing
  n = Int(m.recs[recs[1]].n)
  all(j -> m.recs[j].label == 0 && m.lsid[j] == 0 && m.recs[j].n == n, recs) || return nothing
  rs = recs[1:max(1, length(recs) ÷ sample):end]
  p0 = priorodds(m, lpriorkey(tk, UInt32(0)))
  D = fill(NaN32, n, n)
  for a in 1:n
    e, vc = slot_effects(m, rs, n, a, p0)
    num, den = zeros(n), zeros(n)
    for ((v, s), x) in e
      w = Float64(sum(vc[v]))
      w >= max(m.opts.mincount, 1) || continue
      num[s] += w * x; den[s] += w
    end
    for s in 1:n
      s == a && continue
      den[s] > 0 && (D[a, s] = Float32(num[s] / den[s]))
    end
  end
  D
end

"""
The value-level version of ci_table: (A, a, B) -> how much feature slot B still changes o(x) once A = a
(slot_effects' split-half measure with A as the gate, unaveraged), for values a with at least `vmin`
records in the sample (rarer values: ciorder :value falls back to D[A, B]).  `nothing` as ci_table.
"""
function ci_values(m::ChainModel, tk::UInt64; sample = 200_000, vmin = 200)
  recs = get(m.typerecs, tk, Int32[])
  isempty(recs) && return nothing
  n = Int(m.recs[recs[1]].n)
  all(j -> m.recs[j].label == 0 && m.lsid[j] == 0 && m.recs[j].n == n, recs) || return nothing
  rs = recs[1:max(1, length(recs) ÷ sample):end]
  p0 = priorodds(m, lpriorkey(tk, UInt32(0)))
  itemof = Dict{Int32,UInt64}()
  V = Dict{Tuple{Int16,UInt64,Int16},Float32}()
  for a in 1:n
    e, vc = slot_effects(m, rs, n, a, p0)
    for ((v, s), x) in e
      (s == a || sum(vc[v]) < vmin) && continue
      isempty(itemof) && for (it, id) in m.idof; itemof[id] = it; end
      V[(Int16(a), itemof[v], Int16(s))] = x
    end
  end
  V
end

"Trace hook (accum): for each leftover taken, (slot, running estimate before, its walk along the path, step taken, correction,
its score in every round it was scored -- see hist)."
const ACCUM_TRACE = Ref{Any}(nothing)
"greedyby=mindev trace hook: (X, path [(a, signed diff, n(B∪a), n(B∪a∪X))], status :explained | :kept | :short, correction) per evaluation."
const MINDEV_TRACE = Ref{Any}(nothing)
"Step score in force for the decision being scored (posnegonly = :attach sets :abs for labeled decisions), or nothing."
const STEPSCORE_OVERRIDE = Ref{Union{Nothing,Symbol}}(nothing)
"accumby=bfs: the deepest backoff level searched (beyond it, the odds walk)."
const BFS_LEVELS = 3

"Does a leftover context entry (context + leftover) have the records `ctxsupport` asks for?"
function ctxsupported(m::ChainModel, e, label)
  (c1, c2) = get(e[2], label, (0.0, 0.0))
  m.opts.leftmin2 > 0 && (c1 < m.opts.leftmin2 || c2 < m.opts.leftmin2) && return false   # two-sided support
  m.opts.ctxsupport <= 0 && return true
  c1 + c2 >= m.opts.ctxsupport
end

function _greedy_leftovers(m::ChainModel, st::ChainState, Ic, label, prior, remaining)
  base = st.views[1].base                            # the decision type's records
  avail = copy(st.members)                           # chain features, (redundant ones,) then included leftovers
  total_ = 0.0
  needmin, needmax = m.opts.minctx, m.opts.maxctx
  if m.opts.ctxchain                                 # contexts as large as the chain (none if it's empty)
    depth(st) == 0 && return 0.0
    needmin = needmax = depth(st)
  end
  nred = 0
  poolw = Dict{UInt64,Float64}()                     # poolsigma: pool feature -> weight of conditioning on it
  tk0 = isempty(base) ? nothing : m.recs[base[1]].typekey
  if interleaves(m, tk0)                             # the chain hasn't run yet: it runs with the leftovers
    G0 = get(m.generalize, tk0, Dict{Int,Vector{Int}}())
    skip = isempty(m.noleft) ? Set{Int}() : Set{Int}(get(m.noleft, tk0, Int[]))
    return accum_leftovers(m, st, Ic, G0, Int[], label, prior, base; chainrem = remaining, skip)
  end
  if m.opts.redundant                                # redundant features: context only, no correction of their own
    weak = Int[]
    rd = redundant_divergence(m, st, Ic, label, remaining; weak)
    red = first.(rd)
    append!(avail, UInt64[Ic[s] for s in red])
    gone = m.opts.poolmarginal == :drop ? union(red, weak) : red   # :drop: weak non-redundant features are discarded
    remaining = [s for s in remaining if !(s in gone)]
    nred = length(red)
    m.opts.poolsigma > 0 && generalization_weights!(poolw, m, st, Ic, label, prior, base, Set(UInt64[Ic[s] for s in red]))
  end
  # adaptive: short chains (plus pool) need only as much context as they have, but at least 1
  m.opts.adaptctx && (needmin = max(1, min(m.opts.minctx, depth(st) + nred)))
  tk = isempty(base) ? UInt64(0) : m.recs[base[1]].typekey
  mode = leftmode(m, tk)
  G = mode == :greedy ? nothing : get(m.generalize, tk, Dict{Int,Vector{Int}}())
  if G !== nothing && m.opts.distback && tk == typekey(:token, 1)   # attachment records: slot 1 distance, slot 2 direction
    G = copy(G)
    G[1] = [2]
  end
  # :cover: the chain forms' generalizations may stand in for them in contexts
  if mode == :cover
    for w in st.members
      sw = findfirst(==(w), Ic)
      sw === nothing && continue
      for sg in get(G, sw, Int[])
        g = Ic[sg]
        (g == ABSENT || g in avail) || push!(avail, g)
      end
    end
  end
  if mode == :accum
    rel = Int[s for s in remaining if Ic[s] != ABSENT &&
      (m.opts.relevance <= 0 || leftover_relevance(m, st, Ic[s], label, prior, base) >= m.opts.relevance)]
    return accum_leftovers(m, st, Ic, G, rel, label, prior, base)
  end
  pending = Int[]
  incl = Set{UInt64}()                               # included leftovers (depweight: counted evidence)
  # X's greedy context (from `avail`): (correction, context) or nothing if it never reaches needmin features;
  # greedymin: the context, among the growth's prefixes with at least needmin features, with the smallest |correction|
  function greedy_context(X)
    S, SX = ChainState(prior, base), ChainState(prior, base)
    eX = only(extend_all(m, S, 0, [X]))
    yX, okX = update_odds(m, eX, label, prior)
    advance!(m, SX, X, eX, yX)
    corr = okX ? log(yX) - log(prior) : 0.0
    chosen = UInt64[]
    mpath = Any[]                                    # (mindev trace) the path: (a, diff, n(B∪a), n(B∪a∪X)) per step
    best = nothing                                   # (greedymin) the smallest |correction| so far
    note!() = length(chosen) >= needmin && (best === nothing || abs(corr) < abs(best[1])) && (best = (corr, copy(chosen)))
    if m.opts.ctxfromchain                            # the chain first, in its order, as far as S∪X stays supported
      for f in st.members
        e1, e2 = only(extend_all(m, S, depth(S), [f])), only(extend_all(m, SX, depth(SX), [f]))
        y1, ok1 = update_odds(m, e1, label, S.odds[end])
        y2, ok2 = leftover_update(m, e1, e2, label, y1)
        (ok1 && ok2 && ctxsupported(m, e2, label)) || break
        advance!(m, S, f, e1, y1); advance!(m, SX, f, e2, y2)
        push!(chosen, f)
        corr = log(y2) - log(y1)
        note!()
      end
    end
    while length(chosen) < needmax
      fs = UInt64[f for f in avail if !(f in chosen) && f != X]    # (:cover: X may be among the stand-ins)
      isempty(fs) && break
      eS = extend_all(m, S, depth(S), fs)
      eSX = extend_all(m, SX, depth(SX), fs)
      oS = S.odds[end]
      bs, bi, by1, by2, bcorr = -Inf, 0, 0.0, 0.0, 0.0
      for (i, f) in enumerate(fs)
        y1, ok1 = update_odds(m, eS[i], label, oS)          # o(S∪f)
        y2, ok2 = leftover_update(m, eS[i], eSX[i], label, y1)   # o(S∪f∪X), from o(S∪f)
        (ok1 && ok2 && ctxsupported(m, eSX[i], label)) || continue   # S∪f∪X must be supported
        c = log(y2) - log(y1)
        m.opts.greedyby == :mindev && (sc = -abs(c); sc > bs && ((bs, bi, by1, by2, bcorr) = (sc, i, y1, y2, c)); continue)
        sc = m.opts.greedyby == :correction ? abs(c - corr) : m.opts.greedyby == :odds ? abs(log(y1) - log(oS)) :
          m.opts.greedyby == :dep ? depq(m, base, X, f, label, prior) : depq(m, base, X, f, label, prior) * abs(c - corr)
        sc > bs && ((bs, bi, by1, by2, bcorr) = (sc, i, y1, y2, c))
      end
      bi == 0 && break
      # mindev (user's design): B grows by the a with the smallest diff = |log o(x|e,B∪a) - log o(x|B∪a)|; e is
      # explained away (dropped) once diff < mindevtau.  mindevstop :localmin -- stop (after minctx features) when
      # every a would raise diff; :exhaust -- grow until no a is supported, e keeps the last diff; :pathmin -- grow
      # as :exhaust, e keeps the smallest diff on the path (from minctx features on)
      if m.opts.greedyby == :mindev
        m.opts.mindevstop == :localmin && length(chosen) >= needmin && abs(bcorr) > abs(corr) && break
        MINDEV_TRACE[] === nothing || push!(mpath, (fs[bi], bcorr, get(eS[bi][2], label, (0.0, 0.0)), get(eSX[bi][2], label, (0.0, 0.0))))
        if abs(bcorr) < m.opts.mindevtau
          MINDEV_TRACE[] === nothing || MINDEV_TRACE[](X, mpath, :explained, bcorr)
          return m.opts.explainpool ? :explained : nothing
        end
      end
      advance!(m, S, fs[bi], eS[bi], by1)
      advance!(m, SX, fs[bi], eSX[bi], by2)
      push!(chosen, fs[bi])
      corr = bcorr
      note!()
    end
    (m.opts.greedymin || (m.opts.greedyby == :mindev && m.opts.mindevstop == :pathmin)) && best !== nothing && ((corr, chosen) = best)
    MINDEV_TRACE[] === nothing || m.opts.greedyby != :mindev || MINDEV_TRACE[](X, mpath, length(chosen) >= needmin ? :kept : :short, corr)
    length(chosen) >= needmin || return nothing
    mode == :cover && !covers(st, Ic, G, chosen) && return nothing
    W = isempty(poolw) ? 1.0 : prod(get(poolw, f, 1.0) for f in chosen; init = 1.0)
    if W < 1.0 && m.opts.poolmode == :local          # scale the correction by the generalizations' weights
      corr *= W
    elseif W < 1.0                                   # blend toward conditioning without the pool features
      corr = W * corr + (1 - W) * context_correction(m, X, [f for f in chosen if !haskey(poolw, f)], label, prior, base)
    end
    m.opts.depweight && (corr *= depweight(m, base, X, setdiff(union(st.members, incl), chosen), label, prior))
    (corr, chosen)
  end
  for s in remaining
    X = Ic[s]
    X == ABSENT && continue
    m.opts.relevance > 0 && leftover_relevance(m, st, X, label, prior, base) < m.opts.relevance && continue
    if mode in (:backoff, :limited)
      c = backoff_correction(m, st, Ic, G, X, label, prior, base; mode)
      LEFTOVER_TRACE[] === nothing || LEFTOVER_TRACE[](s, c !== nothing,
        c === nothing ? Int[] : [something(findfirst(==(f), Ic), 0) for f in c[2]], c === nothing ? 0.0 : c[1])
      c === nothing || (total_ += c[1]; X in avail || push!(avail, X))
      continue
    end
    if m.opts.greedysel                              # (rounds below: the largest correction first)
      push!(pending, s)
      continue
    end
    c = greedy_context(X)
    c === nothing && continue
    if c === :explained                              # explainpool: X joins the context pool, no correction
      X in avail || push!(avail, X)
      (d = DC_STATS[]) === nothing || (d[:pooled] = get(d, :pooled, 0.0) + 1)
      continue
    end
    LEFTOVER_TRACE[] === nothing || LEFTOVER_TRACE[](s, true, [something(findfirst(==(f), Ic), 0) for f in c[2]], c[1])
    DC_STATS[] === nothing || dc_record!(m, base, X, setdiff(union(st.members, incl), c[2]), label, prior, c[1])
    total_ += c[1]
    X in avail || push!(avail, X)
    push!(incl, X)
  end
  # greedysel: in rounds, every remaining leftover's context against the current pool, the one with the
  # largest |correction| taken (it joins the pool), until none has a supported context
  while !isempty(pending)
    bi, bc = 0, nothing
    expl = Int[]                                     # (explainpool) explained away this round
    for (i, s) in enumerate(pending)
      c = greedy_context(Ic[s])
      c === nothing && continue
      c === :explained && (push!(expl, s); continue)
      (bc === nothing || abs(c[1]) > abs(bc[1])) && ((bi, bc) = (i, c))
    end
    if !isempty(expl)                                # explainpool: they join the context pool, no correction
      for s in expl
        Ic[s] in avail || push!(avail, Ic[s])
        (d = DC_STATS[]) === nothing || (d[:pooled] = get(d, :pooled, 0.0) + 1)
      end
      bi == 0 && (filter!(s -> !(s in expl), pending); continue)
    end
    bi == 0 && break
    s = pending[bi]
    LEFTOVER_TRACE[] === nothing || LEFTOVER_TRACE[](s, true, [something(findfirst(==(f), Ic), 0) for f in bc[2]], bc[1])
    DC_STATS[] === nothing || dc_record!(m, base, Ic[s], setdiff(union(st.members, incl), bc[2]), label, prior, bc[1])
    total_ += bc[1]
    Ic[s] in avail || push!(avail, Ic[s])
    push!(incl, Ic[s])
    deleteat!(pending, bi)
    filter!(x -> !(x in expl), pending)
  end
  total_
end

function decide(m::ChainModel, st::State, key::AgendaKey)
  d, i = decisiondef(st, key), key[3]
  d.ready(st, i) || return (NONE, -Inf)
  r = chain_scores(m, st, key)
  r === nothing && return (NONE, -Inf)
  cands, scores = r
  # soft head properties: each candidate head also scored by the (uncommitted) head properties' log
  # probabilities of the values it implies -- the product of the ceiling's domains=product
  SOFTHEADS[] && st.infer && d.name == :attach && (scores = scores .+ soft_head_terms(m, st, i, cands))
  wait = d.open ? waitscore(m, st, key) : nothing
  # displacement: a filtered-out head that beats the best allowed one by more than δ is chosen (commit! reopens
  # the arcs in its way)
  if d.name == :attach && !isnan(DISPLACE[]) && st.infer && !STRICT[]
    wide = displace_candidates(st, i, cands)
    if !isempty(wide)
      _, ws = chain_scores(m, st, key; cands = wide)
      SOFTHEADS[] && (ws = ws .+ soft_head_terms(m, st, i, wide))
      maximum(ws) > maximum(scores) + DISPLACE[] && return choose([cands; wide], [scores; ws], wait)
    end
  end
  choose(cands, scores, wait)
end

"""
For soft head properties (SOFTHEADS): per attachment candidate of token t, the sum over t's uncommitted
head properties of log P(property = the value the candidate implies), the properties scored as their
decisions would be (whether or not they're ready).
"""
function soft_head_terms(m::ChainModel, st::State, t, cands)
  toks = st.layers[:token]
  add = zeros(length(cands))
  for (k, dd) in enumerate(toks.spec.decisions)
    haskey(ALLHEADSETS, dd.name) && toks[dd.name, t] == PENDING || continue
    r = chain_scores(m, st, (:token, k, t))
    r === nothing && continue
    vals, sc = r
    mx = maximum(sc)
    lz = mx + log(sum(x -> exp(x - mx), sc))
    for (j, c) in enumerate(cands)
      v = head_value(st, t, c, dd.name)
      v == PENDING && continue
      q = findfirst(==(v), vals)
      add[j] += q === nothing ? log(1e-12) : sc[q] - lz
    end
  end
  add
end

"A decision's candidates (or the given ones) and their chain log odds (with leftovers), or nothing if none; readiness not checked."
function chain_scores(m::ChainModel, st::State, key::AgendaKey; cands = nothing)
  d, i = decisiondef(st, key), key[3]
  cands = cands === nothing ? d.candidates(st, i) : cands
  isempty(cands) && return nothing
  tk = typekey(key[1], key[2])
  CHAIN_TRACE[] === nothing || (TRACE_KEY[] = key)
  labels = [d.labeled ? c : UInt32(0) for c in cands]
  I = [d.items(st, i, c) for c in cands]
  isempty(m.screened) || (I = [[x in m.screened ? ABSENT : x for x in Ic] for Ic in I])
  isempty(m.climb) || (I = [[haskey(m.climb, x) ? climbkey!(m, x, labels[c]) : x for x in I[c]] for c in eachindex(I)])
  memokey = hash((tk, labels, I))
  scores = get(m.memo, memokey, nothing)
  (d_ = DC_STATS[]) === nothing || (k_ = scores === nothing ? Symbol(:memo_miss_, d.name) : Symbol(:memo_hit_, d.name); d_[k_] = get(d_, k_, 0.0) + 1)
  scores === nothing || return (cands, copy(scores))
  if m.opts.posnegonly != :all && d.labeled && m.opts.stepscore != :abs &&    # labeled decisions (but head properties with :heads):
     !(m.opts.posnegonly == :heads && haskey(ALLHEADSETS, d.name))            # plain :abs chains, no restart
    STEPSCORE_OVERRIDE[] = :abs
    try
      return chain_scores_inner(m, st, key, d, i, cands, tk, labels, I, memokey, false)
    finally
      STEPSCORE_OVERRIDE[] = nothing
    end
  end
  chain_scores_inner(m, st, key, d, i, cands, tk, labels, I, memokey, m.opts.posnegrestart)
end

function chain_scores_inner(m::ChainModel, st::State, key::AgendaKey, d, i, cands, tk, labels, I, memokey, restart::Bool)
  priors = [priorodds(m, lpriorkey(tk, l)) for l in labels]
  tr = get(m.typerecs, tk, NOLIST)
  # leftovers are sampled by the ranking the chains end with
  logodds(states, rems) = (cap = samplecap(m, [s.odds[end] for s in states]; nlabels = d.labeled ? length(cands) : 1);
    [chain_logodds(m, states[c], I[c], labels[c], priors[c], rems isa Vector{Int} ? rems : rems[c]; cap)
     for c in eachindex(cands)])
  scores = if m.opts.select == :maxdev && m.opts.top2
    logodds(run_top2(m, I, labels, priors, tr; tk)...)
  elseif m.opts.select == :maxdev && m.opts.target
    logodds(run_maxdev_target(m, I, labels, priors, tr; tk)...)
  elseif m.opts.select == :maxdev && m.opts.sample > 0
    logodds(run_maxdev_rounds(m, I, labels, priors, tr; tk)...)
  elseif m.opts.select == :maxdev
    fstates = Vector{Any}(nothing, length(cands))     # (stickbreak = :cond) each candidate's final chain state
    if m.opts.dualchain != :off
      # two chains per candidate; leftovers (dualleft) in the context of both chains' features, on the combined score
      dscores = map(eachindex(cands)) do c
        P, N, dP, dN, used = run_dual(m, I[c], labels[c], priors[c], tr; tk)
        sc = dual_score(m, P, N, dP, dN, priors[c])
        m.opts.dualleft || return sc
        U = ChainState(priors[c], tr)
        for x in unique([P.members; N.members])
          e = only(extend_all(m, U, depth(U), [x]))
          y, ok = update_odds(m, e, labels[c], U.odds[end])
          ok && advance!(m, U, x, e, y)
        end
        rem = [s for s in (tk === nothing ? collect(eachindex(I[c])) : admitted(m, tk, I[c])) if !(s in used) && !(I[c][s] in U.members)]
        sc + chain_logodds(m, U, I[c], labels[c], priors[c], rem) - log(U.odds[end])
      end
      m.opts.stickbreak != :off && (dscores = stickbreak_scores(m, dscores, labels, nothing, tk, d.labeled))
      length(m.memo) >= MAXMEMO && empty!(m.memo)
      m.memo[memokey] = copy(dscores)
      return (cands, dscores)
    end
    if m.opts.candmemo && CHAIN_TRACE[] === nothing && LEFTOVER_TRACE[] === nothing && PARTS_HOOK[] === nothing &&
       m.opts.sample <= 0 && !m.opts.contest && m.opts.stickbreak == :off && !(restart && !m.opts.posnegafter && length(cands) > 1)
      cs_ = cand_scores(m, cands, I, labels, priors, tr, tk, restart)
      length(m.memo) >= MAXMEMO && empty!(m.memo)
      m.memo[memokey] = copy(cs_)
      return (cands, cs_)
    end
    if m.opts.forkcap > 0                            # forked chains: each candidate's score from its chains
      fscores = [fork_score(m, I[c], labels[c], priors[c], tr; tk) for c in eachindex(cands)]
      m.opts.stickbreak != :off && (fscores = stickbreak_scores(m, fscores, labels, nothing, tk, d.labeled))
      length(m.memo) >= MAXMEMO && empty!(m.memo)
      m.memo[memokey] = copy(fscores)
      return (cands, fscores)
    end
    chains = [run_single(m, I[c], labels[c], priors[c], tr; tk) for c in eachindex(cands)]
    if restart && m.opts.posnegafter && length(cands) > 1
      # restart decided after the leftovers: rank by the full score, restart the top candidate's chain seeded with its
      # negative steps (posnegleftneg: then its negative leftovers), recompute its leftovers, re-rank -- until the top
      # is stable (up to 3 restarts)
      leftneg = [UInt64[] for _ in cands]
      leftall = [Tuple{UInt64,Float64}[] for _ in cands]   # (reorder) each taken leftover and its correction
      function full!(c)
        if CHAIN_TRACE[] !== nothing                  # (tracing: this candidate's leftovers start afresh)
          CHAIN_TRACE_CAND[] = cands[c]
          CHAIN_TRACE[](:reset, (cand = cands[c],))
        end
        old = LEFTOVER_TRACE[]
        empty!(leftneg[c]); empty!(leftall[c])
        LEFTOVER_TRACE[] = (s, inc, ctxs, corr) -> (inc && corr < 0 && push!(leftneg[c], I[c][s]); inc && push!(leftall[c], (I[c][s], corr));
          old === nothing || old(s, inc, ctxs, corr))
        try
          chain_logodds(m, chains[c][1], I[c], labels[c], priors[c], chains[c][2])
        finally
          LEFTOVER_TRACE[] = old
        end
      end
      fs = [full!(c) for c in eachindex(cands)]
      done_ = Set{Int}()
      for _ in 1:3
        top = argmax(fs)
        top in done_ && break
        push!(done_, top)
        cs = chains[top][1]
        negs = UInt64[cs.members[j] for j in eachindex(cs.members) if cs.odds[j+1] < cs.odds[j]]
        m.opts.posnegleftneg && append!(negs, leftneg[top])
        if m.opts.posnegseed in (:reorder, :reorderabs)
          # reorder: every feature the candidate used, chain steps and taken leftovers, by what it added in its previous
          # position (signed, or by size), appended in that order and re-scored
          used = [[(cs.members[j], log(cs.odds[j+1]) - log(cs.odds[j])) for j in eachindex(cs.members)]; leftall[top]]
          sort!(used; by = u -> m.opts.posnegseed == :reorder ? -u[2] : -abs(u[2]))
          negs = UInt64[u[1] for u in used]
        end
        isempty(negs) && break
        chains[top] = run_single(m, I[top], labels[top], priors[top], tr; tk, seed = negs)
        fs[top] = full!(top)
        (d = DC_STATS[]) === nothing || (d[:pn_restart] = get(d, :pn_restart, 0.0) + 1)
        argmax(fs) == top && break
      end
      scores = fs
      if CHAIN_TRACE[] !== nothing
        for c in eachindex(cands)
          cs = chains[c][1]
          CHAIN_TRACE[](:final, (cand = cands[c], logodds = fs[c], members = copy(cs.members), odds = copy(cs.odds)))
        end
      end
      m.opts.stickbreak != :off && (scores = stickbreak_scores(m, scores, labels, [c[1] for c in chains], tk, d.labeled))
      length(m.memo) >= MAXMEMO && empty!(m.memo)
      m.memo[memokey] = copy(scores)
      return (cands, scores)
    end
    if restart && length(cands) > 1
      # restart the top candidate's chain seeded with its own negative steps (then posneg again), until the top
      # candidate is stable or every new top has been restarted (up to 3 restarts)
      negsof(cs) = UInt64[cs.members[j] for j in eachindex(cs.members) if cs.odds[j+1] < cs.odds[j]]
      # :oldtop -- the seed comes from the candidate ranked first after its positive stage (before any negative step)
      posscore(cs) = (j = findfirst(j -> cs.odds[j+1] < cs.odds[j], eachindex(cs.members)); log(cs.odds[j === nothing ? end : j]))
      prev = argmax([posscore(cs) for (cs, _) in chains])
      done_ = Set{Int}()
      for _ in 1:3
        top = argmax([log(cs.odds[end]) for (cs, _) in chains])
        top in done_ && break
        cs = chains[top][1]
        negs = m.opts.posnegseed == :own ? negsof(cs) : negsof(chains[prev][1])
        prev = top
        push!(done_, top)
        isempty(negs) && break
        chains[top] = run_single(m, I[top], labels[top], priors[top], tr; tk, seed = negs)
        (d = DC_STATS[]) === nothing || (d[:pn_restart] = get(d, :pn_restart, 0.0) + 1)
        argmax([log(c[1].odds[end]) for c in chains]) == top && break
      end
    end
    [begin
      CHAIN_TRACE[] === nothing || (CHAIN_TRACE_CAND[] = cands[c])
      cs, rem = chains[c]
      fstates[c] = cs
      lo = chain_logodds(m, cs, I[c], labels[c], priors[c], rem)
      CHAIN_TRACE[] === nothing || CHAIN_TRACE[](:final, (cand = cands[c], logodds = lo, members = copy(cs.members), odds = copy(cs.odds)))
      lo
    end for c in eachindex(cands)]
  else
    logodds(run_chain(m, I, labels, priors, tr; tk)...)
  end
  m.opts.contest && (scores = contest!(m, scores, I, labels, priors, tr, tk))
  m.opts.stickbreak != :off && (scores = stickbreak_scores(m, scores, labels, @isdefined(fstates) ? fstates : nothing, tk, d.labeled))
  length(m.memo) >= MAXMEMO && empty!(m.memo)
  m.memo[memokey] = copy(scores)
  (cands, scores)
end

"""
Per-candidate scores (candmemo; 2026-11-03): each candidate's chain + leftovers score depends only on its decision type,
label and items, so it is memoized under (type, label, items, step score) -- a decision whose other candidates changed
re-scores only the changed ones.  Exact: the same scores as the decision-level path.  With posneg restarts after the
leftovers (posnegafter), the top-ranked candidate's restarted score is memoized too (NaN: no negative step to seed
from), and the ranking / restart loop replayed from the memo.  With forkcap, the forked score.
"""
function cand_scores(m::ChainModel, cands, I, labels, priors, tr, tk, restart::Bool)
  K = length(cands)
  ss = STEPSCORE_OVERRIDE[] === nothing ? m.opts.stepscore : STEPSCORE_OVERRIDE[]
  ck = [hash((tk, labels[c], I[c], ss)) for c in 1:K]
  length(m.cmemo) >= MAXMEMO && empty!(m.cmemo)
  hit!(part) = (d = DC_STATS[]) === nothing || (k = Symbol(:cmemo_, part); d[k] = get(d, k, 0.0) + 1)
  if m.opts.forkcap > 0
    return map(1:K) do c
      v = get(m.cmemo, (ck[c], 3), nothing)
      v === nothing || (hit!(:hit); return v)
      hit!(:miss)
      m.cmemo[(ck[c], 3)] = fork_score(m, I[c], labels[c], priors[c], tr; tk)
    end
  end
  chains = Vector{Any}(nothing, K)
  leftneg = [UInt64[] for _ in 1:K]
  leftall = [Tuple{UInt64,Float64}[] for _ in 1:K]
  # chain + leftovers of candidate c's chain chains[c], noting its negative and all taken leftovers (as full! does)
  function full!(c)
    empty!(leftneg[c]); empty!(leftall[c])
    LEFTOVER_TRACE[] = (s, inc, ctxs, corr) -> (inc && corr < 0 && push!(leftneg[c], I[c][s]); inc && push!(leftall[c], (I[c][s], corr)))
    try
      chain_logodds(m, chains[c][1], I[c], labels[c], priors[c], chains[c][2])
    finally
      LEFTOVER_TRACE[] = nothing
    end
  end
  function first!(c)                                 # candidate c's own chain and its score
    chains[c] = run_single(m, I[c], labels[c], priors[c], tr; tk)
    full!(c)
  end
  fs = map(1:K) do c
    v = get(m.cmemo, (ck[c], 1), nothing)
    v === nothing || (hit!(:hit); return v)
    hit!(:miss)
    m.cmemo[(ck[c], 1)] = first!(c)
  end
  (restart && m.opts.posnegafter && K > 1) || return fs
  done_ = Set{Int}()
  for _ in 1:3
    top = argmax(fs)
    top in done_ && break
    push!(done_, top)
    r = get(m.cmemo, (ck[top], 2), nothing)
    if r === nothing
      chains[top] === nothing && first!(top)        # (its first score was memoized: rebuild the chain for the seed)
      cs = chains[top][1]
      negs = UInt64[cs.members[j] for j in eachindex(cs.members) if cs.odds[j+1] < cs.odds[j]]
      m.opts.posnegleftneg && append!(negs, leftneg[top])
      if m.opts.posnegseed in (:reorder, :reorderabs)
        used = [[(cs.members[j], log(cs.odds[j+1]) - log(cs.odds[j])) for j in eachindex(cs.members)]; leftall[top]]
        sort!(used; by = u -> m.opts.posnegseed == :reorder ? -u[2] : -abs(u[2]))
        negs = UInt64[u[1] for u in used]
      end
      if isempty(negs)
        r = NaN
      else
        chains[top] = run_single(m, I[top], labels[top], priors[top], tr; tk, seed = negs)
        r = full!(top)
      end
      m.cmemo[(ck[top], 2)] = r
    end
    isnan(r) && break
    fs[top] = r
    (d = DC_STATS[]) === nothing || (d[:pn_restart] = get(d, :pn_restart, 0.0) + 1)
    argmax(fs) == top && break
  end
  fs
end

"log(1 + exp(x)), without overflow."
log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

"Label l's (chosen, not chosen) records in prefix counts `cnt` (the decision type's prior counts at the root)."
labcount(m::ChainModel, cnt, tk, l) = isempty(cnt) ? get(m.prior, lpriorkey(tk, l), (0.0, 0.0)) : get(cnt, l, (0.0, 0.0))

"""
Stick-breaking (stickbreak): the candidates, in the order of their log odds s, get p(r_1) = q_1, p(r_2) = q_2 (1 - q_1),
p(r_3) = q_3 (1 - q_1)(1 - q_2), ..., q = s's probability -- each later value fits in what the earlier leave -- then
renormalized; returned as log probabilities.  :cond (labeled decisions, final chain states known): each later value's
odds first conditioned on not the earlier values, its not-chosen records less the earlier values' chosen records in its
own final prefix (one count added to each).
"""
function stickbreak_scores(m::ChainModel, scores, labels, states, tk, labeled)
  K = length(scores)
  K <= 1 && return scores
  s = copy(scores)
  ord = sortperm(scores; rev = true)
  if m.opts.stickbreak == :cond && labeled && states !== nothing
    for (k, ck) in enumerate(ord)
      k == 1 && continue
      cs = states[ck]
      cs === nothing && continue
      cnt = cs.cnt[end]
      c = labcount(m, cnt, tk, labels[ck])
      removed = sum(labcount(m, cnt, tk, labels[cj])[1] for cj in ord[1:k-1])
      s[ck] += log(c[2] + 1) - log(max(c[2] - removed, 0.0) + 1)
    end
  end
  out = similar(s)
  rem = 0.0                                          # log of the mass the earlier values leave
  for ck in ord
    out[ck] = rem - log1pexp(-s[ck])                 # log q + log remaining
    rem -= log1pexp(s[ck])                           # += log(1 - q)
  end
  lz = rem < -1e-12 ? log(-expm1(rem)) : log(1e-12)  # the mass given out: 1 - remaining
  out .- lz
end

"""
Counts packed for storage: keys sorted; for key i, label entries start[i]:start[i+1]-1 of the
label / chosen / not-chosen arrays.  Memory-mapped from disk by load_counts.
"""
struct PackedCounts
  keys::Vector{UInt64}
  start::Vector{UInt32}
  lab::Vector{UInt16}
  c1::Vector{UInt32}
  c2::Vector{UInt32}
end
Base.length(p::PackedCounts) = length(p.keys)

"Counts for `key`, or nothing (packed: binary search on the sorted keys)."
function pcget(p::PackedCounts, key::UInt64)
  i = searchsortedfirst(p.keys, key)
  (i > length(p.keys) || p.keys[i] != key) && return nothing
  LabelCounts(UInt32(p.lab[j]) => (Float64(p.c1[j]), Float64(p.c2[j])) for j in p.start[i]:p.start[i+1]-1)
end
pcget(d::AbstractDict, key::UInt64) = get(d, key, nothing)

"Pack a Dict of counts (keys sorted)."
function packcounts(d::AbstractDict{UInt64,LabelCounts})
  ks = sort!(collect(keys(d)))
  start, lab, c1, c2 = UInt32[1], UInt16[], UInt32[], UInt32[]
  for k in ks
    for (l, (a, b)) in d[k]
      push!(lab, UInt16(l)); push!(c1, UInt32(a)); push!(c2, UInt32(b))
    end
    push!(start, UInt32(length(lab) + 1))
  end
  PackedCounts(ks, start, lab, c1, c2)
end

"Unpack into a Dict (tests, small caches)."
unpackcounts(p::PackedCounts) = Dict(k => pcget(p, k) for k in p.keys)

"Merge two packed caches with disjoint keys into one, in key order, without unpacking either."
function mergecounts(a::PackedCounts, b::PackedCounts)
  n = length(a) + length(b)
  ks, start = Vector{UInt64}(undef, n), Vector{UInt32}(undef, n + 1)
  lab, c1, c2 = UInt16[], UInt32[], UInt32[]
  sizehint!(lab, length(a.lab) + length(b.lab)); sizehint!(c1, length(lab)); sizehint!(c2, length(lab))
  i = j = 1
  start[1] = 1
  for k in 1:n
    froma = j > length(b) || (i <= length(a) && a.keys[i] <= b.keys[j])
    p, r = froma ? (a, i) : (b, j)
    ks[k] = p.keys[r]
    for q in p.start[r]:p.start[r+1]-1
      push!(lab, p.lab[q]); push!(c1, p.c1[q]); push!(c2, p.c2[q])
    end
    start[k+1] = UInt32(length(lab) + 1)
    froma ? (i += 1) : (j += 1)
  end
  PackedCounts(ks, start, lab, c1, c2)
end

const COUNT_FILES = ("counts_keys.bin", "counts_start.bin", "counts_lab.bin", "counts_c1.bin", "counts_c2.bin")
const COUNT_SIG = "counts_sig.txt"

"""
Save packed counts in `dir` with signature `sig` (a string identifying the events): each array to
a temporary file, renamed into place, then the signature file (with the array lengths) last.
"""
function save_counts(dir::AbstractString, sig::AbstractString, p::PackedCounts)
  arrays = (p.keys, p.start, p.lab, p.c1, p.c2)
  rm(joinpath(dir, COUNT_SIG); force = true)        # invalid until every array is written
  for (f, a) in zip(COUNT_FILES, arrays)
    tmp = joinpath(dir, f * ".tmp")
    open(io -> write(io, a), tmp, "w")
    mv(tmp, joinpath(dir, f); force = true)
  end
  open(joinpath(dir, COUNT_SIG), "w") do io
    println(io, sig)
    println(io, join(length.(arrays), ' '))
  end
end

"Memory-map the counts saved in `dir`, or nothing if missing or saved for other events (`sig`)."
function load_counts(dir::AbstractString, sig::AbstractString)
  sf = joinpath(dir, COUNT_SIG)
  isfile(sf) || return nothing
  lines = readlines(sf)
  (length(lines) >= 2 && lines[1] == sig) || return nothing
  lens = parse.(Int, split(lines[2]))
  types = (UInt64, UInt32, UInt16, UInt32, UInt32)
  arrays = map(zip(COUNT_FILES, types, lens)) do (f, T, n)
    path = joinpath(dir, f)
    (isfile(path) && filesize(path) == n * sizeof(T)) || return nothing
    n == 0 ? T[] : Mmap.mmap(path, Vector{T}, n)
  end
  any(isnothing, arrays) && return nothing
  PackedCounts(arrays...)
end
