using SeqLayers, Test

const SAMPLE = """
# newdoc id = d1
# sent_id = 1
# text = I don't like rain.
1	I	I	PRON	PRP	_	4	nsubj	_	_
2-3	don't	_	_	_	_	_	_	_	_
2	do	do	AUX	VBP	_	4	aux	_	_
3	n't	not	PART	RB	_	4	advmod	_	_
4	like	like	VERB	VB	_	0	root	_	_
5	rain	rain	NOUN	NN	_	4	obj	_	SpaceAfter=No
6	.	.	PUNCT	.	_	4	punct	_	_

# sent_id = 2
# text = The U.S. economy grew, slowly.
1	The	the	DET	DT	_	3	det	_	_
2	U.S.	U.S.	PROPN	NNP	_	3	compound	_	_
3	economy	economy	NOUN	NN	_	4	nsubj	_	_
4	grew	grow	VERB	VBD	_	0	root	_	SpaceAfter=No
5	,	,	PUNCT	,	_	6	punct	_	_
6	slowly	slowly	ADV	RB	_	4	advmod	_	SpaceAfter=No
7	.	.	PUNCT	.	_	4	punct	_	_

# sent_id = 3
# text = Rain fell.
1	Rain	rain	NOUN	NN	_	2	nsubj	_	_
2	fell	fall	VERB	VBD	_	0	root	_	SpaceAfter=No
3	.	.	PUNCT	.	_	2	punct	_	_

"""

docs = mktemp() do path, io
  write(io, SAMPLE)
  close(io)
  read_conllu(path)
end

@testset "SeqLayers" begin
  @testset "reader" begin
    @test length(docs) == 1
    d = only(docs)
    @test String(d.text) == "I don't like rain. The U.S. economy grew, slowly. Rain fell."
    @test length(d.gold.head) == 16
    @test haskey(d.gold.head, 3:4) && haskey(d.gold.head, 5:7)      # do | n't
  end

  specs = default_specs()

  @testset "oracle reproduces gold (segment words = $w)" for w in (128, 3)
    r = evaluate(docs, specs, OracleModel(), Lexicon(); words = w, edge = 4)
    @test r.token_f1 == 1.0
    @test r.uas == 1.0
    @test r.las == 1.0
    @test r.sentence_f1 == 1.0
  end

  # Follows gold but never chooses E, so every sentence (except at document end) must close from the parse side
  struct OracleNoE <: AbstractModel end
  function SeqLayers.decide(::OracleNoE, st::SeqLayers.State, key::SeqLayers.AgendaKey)
    choice, margin = decide(OracleModel(), st, key)
    key[1] == :char && choice == SeqLayers.GAP_E && SeqLayers.GAP_S in SeqLayers.gap_candidates(st, key[3]) &&
      return (SeqLayers.GAP_S, margin)
    (choice, margin)
  end

  @testset "parse-side closure sets E (segment words = $w)" for w in (128, 3)
    r = evaluate(docs, specs, OracleNoE(), Lexicon(); words = w, edge = 4)
    @test r.token_f1 == 1.0 && r.uas == 1.0 && r.sentence_f1 == 1.0
  end

  @testset "both sides agree on E" begin
    d = only(docs)
    for model in (OracleModel(), OracleNoE())
      st = SeqLayers.State(d, 1:length(d.text), true, 0, specs, Lexicon())
      solve!(st, model)
      ends = Set(j for j in 1:length(d.text) if st.layers[:char][:gap, j] == SeqLayers.GAP_E)
      @test ends == d.gold.sentends
    end
  end

  @testset "oracle output" begin
    out = process(only(docs), specs, OracleModel(), Lexicon())
    @test [s.tokens for s in out] == [["I", "do", "n't", "like", "rain", "."],
      ["The", "U.S.", "economy", "grew", ",", "slowly", "."], ["Rain", "fell", "."]]
    @test out[1].heads == [4, 4, 4, 0, 4, 4]
    @test out[1].deprels == ["nsubj", "aux", "advmod", "root", "obj", "punct"]
  end

  @testset "suffix arrays" begin
    for s in ("banana", "abracadabra", "aaaaaaa", "mississippi", "a", "")
      t = UInt32.(codepoint.(collect(s)))
      @test SeqLayers.suffixarray(t) == sortperm([t[i:end] for i in 1:length(t)])
    end
    t = UInt32.(rand(1:3, 500))
    @test SeqLayers.suffixarray(t) == sortperm([t[i:end] for i in 1:length(t)])
  end

  @testset "context counts" begin
    d = only(docs)
    idx = CharIndex(docs; maxlen = 4, mincount = 1, beta = 0.0)
    # brute-force label counts for the left context ending at position p, length k
    function brute(p, k, exclude_own)
      c = zeros(Int, 4)
      ctx = d.text[p-k+1:p]
      for q in k:length(d.text)
        d.text[q-k+1:q] == ctx || continue
        exclude_own && q == p && continue
        g = d.gold.gap[q]
        1 <= g <= 4 && (c[g] += 1)
      end
      c
    end
    text = String(d.text)
    p = findfirst("rain", text)[end]            # "rain" occurs once; "ain" also occurs in "Rain"
    # in-index doc, leave-one-out: "rain" has no other occurrence, so the context stops at "ain"
    probs, depth = SeqLayers.context_probs(idx, d, p, :left)
    @test depth == 3
    c = brute(p, 3, true)
    @test probs ≈ c ./ sum(c)
    # same text, not in the index: its "rain" counts, so the context reaches length 4
    probs2, depth2 = SeqLayers.context_probs(idx, doc_from_text(text), p, :left)
    @test depth2 == 4
    c2 = brute(p, 4, false)
    @test probs2 ≈ c2 ./ sum(c2)
  end

  @testset "oracle with suffix-array features" begin
    sspecs = default_specs(charindex = CharIndex(docs))
    r = evaluate(docs, sspecs, OracleModel(), Lexicon())
    @test r.token_f1 == 1.0 && r.uas == 1.0 && r.sentence_f1 == 1.0
  end

  @testset "smoothOdds" begin
    so = SeqLayers.smoothOdds
    @test so(0, 0, 0.2) == 0.2                       # no observations: e
    @test so(3, 10, 0.3) == 0.3                      # r == e
    @test so(6, 4, 3.0) ≈ 6 / 3                      # r < e, c1 > c2, c2 > 1: c1/(c2-1)
    @test so(6, 4, 2.0) == 2.0                       # ...never past e
    @test so(5, 1, 10.0) == 10.0                     # r < e, c1 > c2, c2 <= 1: e
    @test so(1, 10, 0.5) ≈ 2 / 10                    # r < e, c1 <= c2: (c1+1)/c2
    @test so(4, 5, 1.0) == 1.0                       # ...never past e
    @test so(3, 10, 0.1) ≈ 2 / 10                    # r > e, c1 <= c2: (c1-1)/c2
    @test so(1, 10, 0.05) == 0.05                    # ...never past e
    @test so(10, 2, 1.0) ≈ 10 / 3                    # r > e, c1 > c2: c1/(c2+1)
    @test so(10, 0, 1.0) ≈ 10.0                      # ...including c2 == 0
    @test so(1, 0, 2.0) == 2.0                       # ...never past e
  end

  @testset "lattice expansion matches brute force" begin
    # synthetic events: (typekey, label, chosen, sorted items)
    rng_items = UInt64.(101:106)
    events = [(UInt64(7), UInt32(rand(1:2)), rand() < 0.3, sort!(unique(rand(rng_items, rand(1:5))))) for _ in 1:300]
    dir = mktempdir()
    open(joinpath(dir, SeqLayers.CANDS_FILE), "w") do cio
      open(joinpath(dir, SeqLayers.ITEMS_FILE), "w") do iio
        n = 0
        for (tk, label, chosen, I) in events
          write(cio, [SeqLayers.CandRec(tk, label, chosen, n + 1, length(I))])
          write(iio, I)
          n += length(I)
        end
      end
    end
    so = SeqLayers.smoothOdds
    function reference(tk, label, S, maxorder)
      sel = filter(e -> e[1] == tk && e[2] == label, events)
      p = count(e -> e[3], sel)
      lp = log((p + 0.5) / (length(sel) - p + 0.5))
      memo = Dict{Vector{UInt64},Float64}()
      subsets(T) = [T[[(mask >> (j - 1)) & 1 == 1 for j in 1:length(T)]] for mask in 1:(1<<length(T))-1]
      function λ(U)
        get!(memo, U) do
          logE = lp + sum((λ(V) for V in subsets(U) if length(V) < length(U)); init = 0.0)
          c1 = count(e -> e[3] && issubset(U, e[4]), sel)
          c2 = count(e -> !e[3] && issubset(U, e[4]), sel)
          log(so(c1, c2, exp(logE))) - logE
        end
      end
      lp + sum((λ(U) for U in subsets(S) if length(U) <= maxorder); init = 0.0)
    end
    for maxorder in (2, 5)
      m = LatticeModel(dir; maxorder, minsupport = 1, diffthreshold = 0.0, wmothreshold = 0.0, verbose = false)
      for S in ([UInt64(101)], UInt64[101, 103], UInt64[102, 104, 105], UInt64[101, 102, 103, 104, 105])
        @test SeqLayers.lattice_logodds(m, UInt64(7), UInt32(1), S) ≈ reference(UInt64(7), UInt32(1), S, maxorder) atol = 1e-12
      end
    end
    # pruning: with high support nothing survives, so the score falls back to the prior
    m = LatticeModel(dir; maxorder = 3, minsupport = 10_000, verbose = false)
    @test isempty(m.lambda)
    @test SeqLayers.lattice_logodds(m, UInt64(7), UInt32(1), UInt64[101, 102]) ≈ reference(UInt64(7), UInt32(1), UInt64[], 0)
  end

  @testset "lattice model on the sample" begin
    lex = Lexicon()
    dir = record_events(docs, specs, lex, mktempdir())
    m = LatticeModel(dir; maxorder = 3, verbose = false)
    @test !isempty(m.prior) && !isempty(m.supported)
    r = evaluate(docs, specs, m, lex)
    @test 0 <= r.token_f1 <= 1
  end

  @testset "chain model" begin
    for (a, b) in ((Int32[1, 3, 5, 7], Int32[3, 4, 5]), (Int32[2], Int32.(1:100)), (Int32[], Int32[1]), (Int32.(1:3:300), Int32.(2:2:300)))
      @test SeqLayers.intersect_sorted(a, b) == intersect(a, b)
    end
    # with 2 candidates the two clr criteria coincide
    x, y = [0.3, 2.0], [0.9, 1.1]
    @test SeqLayers.criterion(:clr_mean_abs, x, y) ≈ SeqLayers.criterion(:clr_rms, x, y)
    @test SeqLayers.criterion(:clr_rms, [1.0, 2.0, 4.0], [2.0, 4.0, 8.0]) ≈ 0 atol = 1e-12   # scale-free

    lex = Lexicon()
    dir = record_events(docs, specs, lex, mktempdir())
    m = ChainModel(dir; mincount = 1)
    # counts for a set match a brute-force scan of the records
    recs, items = SeqLayers.load_events(dir)
    present(r) = filter(!=(SeqLayers.ABSENT), items[r.start:r.start+r.n-1])
    # (for several record-storage settings: lists kept, views filtered, everything scanned)
    for kf in (1.0, 0.1, 0.001)
      mk = ChainModel(dir; mincount = 1, keepfrac = kf)
      for r in recs[[findfirst(r -> length(present(r)) >= 3, recs); findlast(r -> length(present(r)) >= 4, recs)]]
        S = present(r)[1:min(end, 4)]
        c, cst = SeqLayers.setcounts(mk, r.typekey, S)
        brute = [j for (j, q) in enumerate(recs) if q.typekey == r.typekey && issubset(S, items[q.start:q.start+q.n-1])]
        @test SeqLayers.total(c) == length(brute)
        @test c == SeqLayers.labelcounts(mk, brute)
        @test SeqLayers.buildlist(mk, cst.views[end]) == brute
      end
    end

    for sel in (:maxdev, :avglogodds, :clr_mean_abs, :clr_rms), lo in (:none, :marginal, :backoff, :backoff1)
      r = evaluate(docs, specs, ChainModel(dir; select = sel, leftovers = lo, mincount = 1), lex)
      @test 0 <= r.token_f1 <= 1
    end
    r = evaluate(docs, specs, ChainModel(dir; select = :clr_rms, target = true, mincount = 1), lex)
    @test 0 <= r.token_f1 <= 1
    # limited-discrepancy search: k = 0 is the greedy decoder; branches only ever win on score
    mg = ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1)
    g = evaluate_counts(docs, specs, mg, lex)
    @test evaluate_counts(docs, specs, mg, lex; search = LDS(0)) == g
    for sc in (:sum, :mean)
      empty!(SeqLayers.LDS_STATS); merge!(SeqLayers.LDS_STATS, Dict(:segments => 0, :branches => 0, :switched => 0))
      r = eval_scores(evaluate_counts(docs, specs, mg, lex; search = LDS(3; score = sc)))
      @test 0 <= r.uas <= 1
      @test SeqLayers.LDS_STATS[:branches] > 0
    end
    # DFS with failures: budget 0 is the greedy decoder; undoing a commit restores the grids exactly
    @test evaluate_counts(docs, specs, mg, lex; search = DFS(budget = 0)) == g
    rd = eval_scores(evaluate_counts(docs, specs, mg, lex; search = DFS(eps = 0.01, budget = 50)))
    @test 0 <= rd.uas <= 1
    SeqLayers.SOFTHEADS[] = true                     # soft head properties: the decoder still decodes, greedy and DFS
    try
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mg, lex)).uas <= 1
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mg, lex; search = DFS(eps = 0.01, budget = 50, jump = true, epshead = 0.001))).uas <= 1
    finally
      SeqLayers.SOFTHEADS[] = false
    end
    @test evaluate_counts(docs, specs, mg, lex) == g    # (and switched off again: the greedy decoder as before)
    SeqLayers.FWDCHECK[] = true                      # forward checking: greedy and DFS still decode
    try
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mg, lex)).uas <= 1
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mg, lex; search = DFS(eps = 0.01, budget = 50, jump = true))).uas <= 1
    finally
      SeqLayers.FWDCHECK[] = false
    end
    for o in (DFS(eps = 0.01, budget = 50, jump = true), DFS(eps = 0.01, budget = 50, order = :product),
              DFS(eps = 0.01, budget = 50, jump = true, order = :product))
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mg, lex; search = o)).uas <= 1
    end
    sd2 = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    SeqLayers.activate!(sd2); SeqLayers.rescore!(sd2, mg)
    for _ in 1:30                                   # some greedy steps first
      isempty(sd2.agenda) && break
      k_, mrg = first(sd2.agenda); mrg == -Inf && break
      SeqLayers.commit_choice!(sd2, k_, sd2.best[k_]); SeqLayers.activate!(sd2); SeqLayers.rescore!(sd2, mg)
    end
    before = Dict(l => (copy(sd2.layers[l].grid[:, 1:sd2.layers[l].ncols]), sd2.layers[l].ncols) for l in sd2.order)
    SeqLayers.TRAIL[] = Tuple{Any,Int,Int,UInt32}[]
    try
      k_, _ = first(sd2.agenda)
      cp_ = SeqLayers.ChoicePoint(0, SeqLayers.snapshot(sd2), k_, [sd2.best[k_]], 1)
      for _ in 1:20                                 # commit and propagate a while, then undo all of it
        isempty(sd2.agenda) && break
        k2, m2 = first(sd2.agenda); m2 == -Inf && break
        SeqLayers.commit_choice!(sd2, k2, sd2.best[k2]); SeqLayers.activate!(sd2); SeqLayers.rescore!(sd2, mg)
      end
      SeqLayers.restore!(sd2, cp_)
    finally
      SeqLayers.TRAIL[] = nothing
    end
    @test all(sd2.layers[l].ncols == before[l][2] && sd2.layers[l].grid[:, 1:before[l][2]] == before[l][1] for l in sd2.order)
    # every option lands in its own ChainOpts field (positional constructor)
    mo = ChainModel(dir; select = :maxdev, leftovers = :greedy, mincount = 1, ctxmode = :accum, greedymin = true, greedysel = false,
      driftsigma = 0.25, drifttau = 0.5, driftstep = 0.75, distback = true, shrink = 7.0, accumwin = 3, accumreset = true, cidrop = 0.125,
      shrinkleft = 9.0, leftmin2 = 4, depmeasure = :pmi, mindevtau = 0.3, mindevstop = :pathmin, explainpool = true, chainshrink = 20.0, chainshrinkto = :gen, chainsel = 2.5, lookw = :max, lookwsmooth = 5.0, lookwback = 5.0, lookwsoft = true, stickbreak = :cond, posnegrestart = true, posnegseed = :oldtop, dualchain = :inchain, dualcombine = :sumdedup, dualw = 0.25, dualleft = false, dualturn = :refine, posnegafter = true, posnegleftneg = true, posnegonly = :attach, leftwback = 2.0, leftwovl = true, leftwcomb = :min, wselcomb = :min, wselmove = :wcorr, wselweight = true, tuplek = 6, tuplemin = 1.5, interleave = false)
    o = mo.opts
    @test (o.greedymin, o.greedysel, o.driftsigma, o.drifttau, o.driftstep, o.distback, o.shrink, o.accumwin, o.accumreset, o.cidrop,
      o.shrinkleft, o.leftmin2, o.depmeasure, o.mindevtau, o.mindevstop, o.explainpool, o.chainshrink, o.chainshrinkto, o.chainsel, o.lookw, o.lookwsmooth, o.lookwback, o.lookwsoft, o.stickbreak, o.posnegrestart, o.posnegseed, o.dualchain, o.dualcombine, o.dualw, o.dualleft, o.dualturn, o.posnegafter, o.posnegleftneg, o.posnegonly, o.leftwback, o.leftwovl, o.leftwcomb, o.wselcomb, o.wselmove, o.wselweight, o.tuplek, o.tuplemin, o.interleave) == (true, false, 0.25, 0.5, 0.75, true, 7.0, 3, true, 0.125, 9.0, 4, :pmi, 0.3, :pathmin, true, 20.0, :gen, 2.5, :max, 5.0, 5.0, true, :cond, true, :oldtop, :inchain, :sumdedup, 0.25, false, :refine, true, true, :attach, 2.0, true, :min, :min, :wcorr, true, 6, 1.5, false)
    for (cs, to, zs) in ((20.0, :prev, 0.0), (20.0, :gen, 0.0), (20.0, :prior, 0.0), (0.0, :prev, 2.6))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum,
        chainshrink = cs, chainshrinkto = to, chainsel = zs)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for (cm, kw) in ((:accum, (accumby = :mindev,)), (:greedy, (greedyby = :mindev, greedysel = true)), (:greedy, (greedyby = :mindev,)))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = cm,
        explainpool = true, kw...)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for (lw, ls, lb, lsoft) in ((:min, 0.0, 0.0, false), (:max, 5.0, 0.0, false), (:int, 5.0, 0.0, false), (:int, 0.0, 5.0, false), (:int, 0.0, 5.0, true))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true,
        ctxmode = :accum, accumby = :look1red, lookw = lw, lookwsmooth = ls, lookwback = lb, lookwsoft = lsoft)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for el in (:marginal, :inchain), cb in (:sum, :maxabs, :weighted, :sumdedup), dl in (true, false), dt in (:both, :refine)
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, dualchain = el, dualcombine = cb, dualleft = dl, dualturn = dt)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for (ss, pr, ps, pa, pl) in ((:posneg, false, :own, false, false), (:posnegprior, false, :own, false, false), (:posneg, true, :own, false, false),
        (:posneg, true, :oldtop, false, false), (:posneg, true, :own, true, false), (:posneg, true, :own, true, true),
        (:negpos, false, :own, false, false), (:posneg, true, :reorder, true, false), (:posneg, true, :reorderabs, true, false))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, stepscore = ss, posnegrestart = pr, posnegseed = ps,
        posnegafter = pa, posnegleftneg = pl)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, stepscore = ss, posnegrestart = pr, posnegseed = ps,
        posnegafter = pa, posnegleftneg = pl, posnegonly = :attach)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, stepscore = ss, posnegrestart = pr, posnegseed = ps,
        posnegafter = pa, posnegleftneg = pl, posnegonly = :heads)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    mp = ChainModel(dir; select = :maxpos, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    @test SeqLayers.criterion(:maxpos, [1.0, 2.0], [3.0, 1.0]) ≈ log(3.0)
    @test SeqLayers.criterion(:maxabsscore, [0.5, 2.0], [0.1, 2.0]) ≈ log(10.0) - log(2.0)   # a negative score made more negative counts
    mp = ChainModel(dir; select = :maxposneg, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    mp = ChainModel(dir; select = :maxabsscore, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    for sb in (:renorm, :cond)
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, stickbreak = sb)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    # stick-breaking: a distribution, the top value keeps its rank
    let sc = [2.0, 0.5, -1.0, -3.0]
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, stickbreak = :renorm)
      lp = SeqLayers.stickbreak_scores(mp, sc, UInt32[0, 0, 0, 0], nothing, UInt64(0), false)
      @test sum(exp.(lp)) ≈ 1
      @test argmax(lp) == 1
    end
    # association step scores: independence scores 0, association scores more; MI is non-negative
    for ss in SeqLayers.CRIT_STEPS
      @test SeqLayers.step_crit(ss, 400.0, 600.0, 40.0, 60.0) < 0.02                  # x independent of C
      @test SeqLayers.step_crit(ss, 400.0, 600.0, 90.0, 10.0) > SeqLayers.step_crit(ss, 400.0, 600.0, 55.0, 45.0)
      @test SeqLayers.step_crit(ss, 400.0, 600.0, 5.0, 95.0) > 0.01                   # (negative association counts too)
      @test isfinite(SeqLayers.step_crit(ss, 0.0, 10.0, 0.0, 10.0)) && isfinite(SeqLayers.step_crit(ss, 10.0, 0.0, 10.0, 0.0))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum, stepscore = ss)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    # the association measures as leftover-context dependence (greedy and accum)
    for dmeas in (:mi, :oddspmi, :moi, :logor), (cm, kw) in ((:greedy, (greedyby = :dep,)), (:accum, (accumby = :depodds,)))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = cm,
        depmeasure = dmeas, kw...)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    # interleave: chain steps and leftovers in one loop (abs and posneg steps, with the leftover weightings)
    @test_throws ErrorException ChainModel(dir; select = :maxdev, leftovers = :greedy, ctxmode = :greedy, interleave = true)
    @test_throws ErrorException ChainModel(dir; select = :maxdev, leftovers = :greedy, ctxmode = :accum, interleave = true, stepscore = :signed)
    for (ss, kw) in ((:abs, (;)), (:posneg, (;)), (:posneg, (posnegrestart = true, posnegafter = true, posnegonly = :heads)),
                     (:abs, (leftwback = 1.0, leftwovl = true, leftwcomb = :max)), (:abs, (accumby = :depodds,)), (:abs, (poolsigma = 0.1, poolmarginal = :drop, poolkeepgen = true)))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 3, redundant = true, ctxmode = :accum,
        interleave = true, stepscore = ss, kw...)
      mp.generalize = SeqLayers.generalization_slots(specs)
      SeqLayers.DC_STATS[] = Dict{Symbol,Float64}()
      try
        @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
        @test get(SeqLayers.DC_STATS[], :il_chain, 0) > 0
      finally
        SeqLayers.DC_STATS[] = nothing
      end
    end
    for (tk_, tm) in ((6, 0.0), (0, 1.0), (4, 0.5), (1, 0.0))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum,
        tuplek = tk_, tuplemin = tm)
      mp.generalize = SeqLayers.generalization_slots(specs)
      SeqLayers.DC_STATS[] = Dict{Symbol,Float64}()
      try
        @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
        dc = SeqLayers.DC_STATS[]
        @test !haskey(dc, :tuple0)                     # a tuple step that runs adds 1-3 members
        @test tk_ == 1 ? isempty(keys(dc) ∩ (:tuple2, :tuple3)) && get(dc, :tuple1, 0) > 0 : get(dc, :tuple3, 0) > 0
      finally
        SeqLayers.DC_STATS[] = nothing
      end
    end
    # the tuple step adds 1-3 members at depth 0 only, all from the candidate's slots
    let mp = ChainModel(dir; select = :maxdev, mincount = 1, tuplek = 6)
      tk = first(k for (k, v) in mp.typerecs if length(v) > 20)
      r = mp.recs[mp.typerecs[tk][1]]
      st = SeqLayers.ChainState(SeqLayers.priorodds(mp, SeqLayers.lpriorkey(tk, r.label)), mp.typerecs[tk])
      Ic = mp.items[r.start:r.start+r.n-1]
      rem = collect(eachindex(Ic))
      added = SeqLayers.tuple_step!(mp, st, Ic, r.label, rem, tk)
      @test added == (SeqLayers.depth(st) > 0)
      @test SeqLayers.depth(st) <= 3 && all(x -> x in Ic, st.members) && length(rem) == length(Ic) - SeqLayers.depth(st)
      @test !SeqLayers.tuple_step!(mp, st, Ic, r.label, rem, tk) || SeqLayers.depth(st) == 0
    end
    for wcomb in (:min, :sum), wm in (:weights, :wcorr), ww in (false, true)
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum,
        accumby = :wsel, wselcomb = wcomb, wselmove = wm, wselweight = ww)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for (wb, wo, wc) in ((1.0, false, :prod), (0.0, true, :prod), (1.0, true, :prod), (1.0, true, :min), (1.0, true, :max))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum,
        leftwback = wb, leftwovl = wo, leftwcomb = wc)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for ab in (:marg, :mindev, :mindevsup, :mintw, :mintwc, :bfs, :look1ind, :look1red, :look1nonred, :look1int)
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true,
        ctxmode = :accum, accumby = ab)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for (gs, ms) in ((false, :localmin), (true, :localmin), (true, :exhaust), (true, :pathmin))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true,
        greedyby = :mindev, greedysel = gs, mindevstop = ms)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    for dm in (:pmi, :pmipos), (cm, gb, ab) in ((:greedy, :dep, :odds), (:accum, :odds, :dep))
      mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = cm,
        greedyby = gb, accumby = ab, depmeasure = dm)
      mp.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mp, lex)).uas <= 1
    end
    # Beta shrinkage: n records move the estimate n/(n+k) of the way from the parent's
    @test SeqLayers.shrinkOdds(7, 1, 0.0146, 20.0) ≈ (p = (7 + 20 * 0.0146 / 1.0146) / 28; p / (1 - p))
    @test SeqLayers.shrinkOdds(0, 0, 0.3, 10.0) ≈ 0.3
    @test isfinite(SeqLayers.shrinkOdds(10, -3, 0.5, 20.0)) && SeqLayers.shrinkOdds(10, -3, 0.5, 20.0) > 0   # (inconsistent sampled counts)
    ms = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, shrink = 20.0)
    @test 0 <= eval_scores(evaluate_counts(docs, specs, ms, lex)).uas <= 1
    # evidence-side leftover smoothing: Bayes' rule with E's smoothed rates among chosen / not chosen
    me = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, leftsmooth = :evidence)
    v0 = SeqLayers.RecView(Int32[])
    y2, ok = SeqLayers.leftover_update(me, (v0, Dict(UInt32(0) => (100.0, 900.0))), (v0, Dict(UInt32(0) => (10.0, 10.0))), UInt32(0), 0.2)
    oE = 20 / 980
    px, pnx = SeqLayers.smoothOdds(10, 90, oE), SeqLayers.smoothOdds(10, 890, oE)
    @test ok && y2 ≈ 0.2 * (px / (1 + px)) / (pnx / (1 + pnx))
    @test 0.2 < y2 < 0.2 * 9                          # toward 1 from the unsmoothed ratio (10/100)/(10/900)
    @test 0 <= eval_scores(evaluate_counts(docs, specs, me, lex)).uas <= 1
    # leftover contexts from the generalized chain (backoff, limited) or covering it (cover)
    for cm_ in (:backoff, :limited, :cover, :accum)
      mb = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = cm_)
      mb.generalize = SeqLayers.generalization_slots(specs)
      @test 0 <= eval_scores(evaluate_counts(docs, specs, mb, lex)).uas <= 1
    end
    # pairwise conditional-independence table, and accum dropping only redundant features
    Dt = SeqLayers.ci_table(mg, SeqLayers.typekey(:token, 1); sample = 1000)
    @test Dt isa Matrix{Float32} && size(Dt, 1) == size(Dt, 2) && all(x -> isnan(x) || x >= 0, Dt)
    @test SeqLayers.ci_table(mg, SeqLayers.typekey(:token, 2)) === nothing     # labeled records: not attachment-style
    mc = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1, redundant = true, ctxmode = :accum, cidrop = 0.1)
    mc.generalize = SeqLayers.generalization_slots(specs)
    mc.ci[SeqLayers.typekey(:token, 1)] = Dt
    @test 0 <= eval_scores(evaluate_counts(docs, specs, mc, lex)).uas <= 1
    # relation variants: bounded left/right siblings never cross the head; with everything attached
    # (oracle) and sentences closed, a missing relation is NONE, never PENDING
    so = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    solve!(so, OracleModel())
    to = so.layers[:token]
    pos_(u) = to[:first_char, u]
    fb, fu = SeqLayers.RelFlags(true, true), SeqLayers.RelFlags(false, true)
    for t in 1:length(to)
      (to[:sentence, t] == PENDING || to[:attach, t] == PENDING) && continue
      h = to[:attach, t]
      for kind in SeqLayers.SIB_KINDS
        vb = SeqLayers.sibling_value(so, t, h, kind, fb)
        @test vb != PENDING
        vu = SeqLayers.sibling_value(so, t, h, kind, fu)
        old = h == NONE ? PENDING : SeqLayers.sibling_of(so, t, h, kind)
        @test vu == (old == PENDING ? NONE : old)                 # unbounded = the old relation, none made explicit
        if kind == :left_sibling && vb != NONE && h != NONE && pos_(Int(h)) < pos_(t)
          @test pos_(Int(vb)) > pos_(Int(h))
        end
        if kind == :right_sibling && vb != NONE && h != NONE && pos_(Int(h)) > pos_(t)
          @test pos_(Int(vb)) < pos_(Int(h))
        end
      end
      for kind in SeqLayers.CHILD_KINDS
        @test SeqLayers.child_value(so, t, kind, fb) != PENDING
      end
    end
    # tree rank: t's rank under its gold head counts the head's other dependents between them
    for t in 1:length(to)
      h = to[:attach, t]
      (h == PENDING || h == NONE) && continue
      lo, hi = minmax(pos_(t), pos_(Int(h)))
      @test SeqLayers.tree_rank(so, t, h) == 1 + count(u -> u != t && to[:attach, u] == h && lo < pos_(u) < hi, 1:length(to))
    end
    for t in 1:length(to)                            # the gold head_trank label is the rank under the gold head
      h = to[:attach, t]
      (h == PENDING || h == NONE) && continue
      r = SeqLayers.tree_rank(so, t, h)
      @test SeqLayers.head_value(so, t, h, :head_trank) == SeqLayers.headcode(:head_trank, SeqLayers.tranklabel(r))
    end
    # index-equivalence items: under the gold tree, "the token's right neighbour is the candidate's nearest left
    # dependent" etc. agree with the attachments
    let n1 = 0
      for t in 1:length(to)
        h = to[:attach, t]
        (h == PENDING || h == NONE) && continue
        it = SeqLayers.idxeq_items(so, t, h)
        @test length(it) == 6
        rn = SeqLayers.adjacent_token(so, t, 1)
        if rn != PENDING && rn != NONE && to[:attach, Int(rn)] == h
          @test it[4] == SeqLayers.item(UInt32(1), (SeqLayers.IDXEQ_TAG, 4, UInt32(1)))
          n1 += 1
        end
      end
      @test n1 > 0
    end
    ispecs = default_specs(idxeq = true)
    @test evaluate(docs, ispecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    idir = record_events(docs, ispecs, Lexicon(), mktempdir())
    mi = ChainModel(idir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, ispecs, mi, Lexicon())).uas <= 1
    # sibling head properties: the oracle commits the gold labels; each equals the value its gold head implies
    sspecs = default_specs(headsibs = [:head_sibin, :head_sibout, :head_sibleft, :head_sibright])
    ss = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, sspecs, Lexicon())
    solve!(ss, OracleModel())
    ts = ss.layers[:token]
    nsib = 0
    for t in 1:length(ts), name in (:head_sibin, :head_sibout, :head_sibleft, :head_sibright)
      h = ts[:attach, t]
      h == PENDING && continue
      @test SeqLayers.head_value(ss, t, h, name) == ts[name, t]
      nsib += SeqLayers.ALLHEADSETS[name][ts[name, t]] in ("1", "2", "3-5", "6-10", "11+")
    end
    @test nsib > 0                                    # (some token has a sibling)
    @test evaluate(docs, sspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    sdir = record_events(docs, sspecs, Lexicon(), mktempdir())
    msb = ChainModel(sdir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, sspecs, msb, Lexicon())).uas <= 1
    # kinship to a neighbour: a hand tree (heads 1->2, 2 root, 3->2, 4->3, 5->4, 6->3, 7->6, 8->1)
    let H = [2, 0, 2, 3, 4, 3, 6, 1], P = Set{Int}()
      hd(u) = u in P ? PENDING : H[u] == 0 ? NONE : UInt32(H[u])
      kin(t, n) = (v = SeqLayers.kin_between(hd, t, n, hd(t), 8); v == PENDING ? "?" : SeqLayers.KIN_LABELS[v])
      @test kin(1, 2) == "parent" && kin(2, 1) == "dep" && kin(1, 3) == "sibling" && kin(4, 1) == "aunt/uncle"
      @test kin(1, 4) == "niece/nephew" && kin(8, 4) == "1stcousin" && kin(4, 2) == "grandparent" && kin(5, 2) == "other_ancestor"
      @test kin(2, 5) == "other_descendant" && kin(5, 1) == "other_relative" && kin(5, 6) == "aunt/uncle"
      @test kin(2, 4) == "grandchild" && kin(3, 5) == "grandchild" && kin(3, 7) == "grandchild" && kin(2, 7) == "other_descendant"
      @test SeqLayers.KIN_LABELS[SeqLayers.KIN_NONE] == "none" && SeqLayers.KIN_LABELS[SeqLayers.KIN_OTHER] == "other_relative"
      @test SeqLayers.ALLHEADSETS.head_kinright == [SeqLayers.KIN_LABELS; "root"]
      @test SeqLayers.KIN_LABELS[SeqLayers.kincode(3, 3)] == "2ndcousin" && SeqLayers.KIN_LABELS[SeqLayers.kincode(3, 2)] == "other_relative"
      @test SeqLayers.kin_between(hd, 1, 3, NONE, 8) == SeqLayers.KIN_OTHER   # t a second root: no common ancestor
      @test SeqLayers.kin_between(hd, 2, 1, UInt32(4), 8) == PENDING          # 4 is under 2: a cycle
      push!(P, 3)
      @test kin(1, 4) == "?" && kin(1, 3) == "?" && kin(4, 3) == "parent" && kin(5, 4) == "parent"   # (3's head unknown)
    end
    # kinship head properties: the oracle commits the gold labels; each equals the value its gold head implies
    kspecs = default_specs(headkin = [:head_kinleft, :head_kinright])
    ks = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, kspecs, Lexicon())
    solve!(ks, OracleModel())
    tk_ = ks.layers[:token]
    kinseen = Set{String}()
    for t in 1:length(tk_), name in (:head_kinleft, :head_kinright)
      h = tk_[:attach, t]
      h == PENDING && continue
      @test SeqLayers.head_value(ks, t, h, name) == tk_[name, t]
      push!(kinseen, SeqLayers.ALLHEADSETS[name][tk_[name, t]])
      h == NONE && continue
      @test SeqLayers.kin_items(ks, t, h)[1] == SeqLayers.item(tk_[:head_kinleft, t], (SeqLayers.KIN_TAG, 1, tk_[:head_kinleft, t]))
    end
    @test length(kinseen) >= 5                         # (parent, dep, sibling, none, root, ...)
    @test evaluate(docs, kspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    for ksp in (kspecs, default_specs(kinitems = true), default_specs(kinitems = true, idxeq = true))
      @test evaluate(docs, ksp, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
      kdir = record_events(docs, ksp, Lexicon(), mktempdir())
      mk = ChainModel(kdir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
      @test 0 <= eval_scores(evaluate_counts(docs, ksp, mk, Lexicon())).uas <= 1
    end
    # neighbor2: the tokens two over (form, upos, head direction) as context and attachment paths
    nspecs2 = default_specs(neighbor2 = true, headsibs = [:head_sibin])
    ctx2 = nspecs2[2].context
    @test (ctx2[end].rels, ctx2[end].feat) == ([:right_neighbor, :right_neighbor], :head_dir) && (ctx2[end-5].rels, ctx2[end-5].feat) == ([:left_neighbor, :left_neighbor], :form) && length(ctx2) == length(default_specs(headsibs = [:head_sibin])[2].context) + 6
    @test SeqLayers.LINKPATHS[nspecs2[2]][end-5:end] == collect(length(ctx2)-5:length(ctx2))
    @test evaluate(docs, nspecs2, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    ndir2 = record_events(docs, nspecs2, Lexicon(), mktempdir())
    mn2 = ChainModel(ndir2; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    mn2.generalize = SeqLayers.generalization_slots(nspecs2)
    @test 0 <= eval_scores(evaluate_counts(docs, nspecs2, mn2, Lexicon())).uas <= 1
    hspecs = default_specs(headtrank = true)
    @test evaluate(docs, hspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    hdir = record_events(docs, hspecs, Lexicon(), mktempdir())
    mh = ChainModel(hdir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, hspecs, mh, Lexicon())).uas <= 1
    # head word classes as head properties (head_wc30): gold from the gold head's class; the oracle stays exact
    wdir = mktempdir()
    write(joinpath(wdir, "wordclasses.tsv"), "#sizes\t30\n" * join(["the\t90\t3", "rain\t40\t7", "fell\t30\t12", "like\t20\t12"], '\n') * "\n")
    wc30 = load_wordclasses(wdir)
    hdocs = deepcopy(docs)
    SeqLayers.add_head_wclass_gold!(hdocs, wc30, [30])
    wspecs = default_specs(headwc = [30], headwcclasses = wc30)
    @test evaluate(hdocs, wspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    wdir2 = record_events(hdocs, wspecs, Lexicon(), mktempdir())
    mw = ChainModel(wdir2; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(hdocs, wspecs, mw, Lexicon())).uas <= 1
    tspecs = default_specs(treerank = true)
    @test evaluate(docs, tspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    rspecs = default_specs(relations = true, relbounded = true, relnone = true)
    @test evaluate(docs, rspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    rdir = record_events(docs, rspecs, Lexicon(), mktempdir())
    mr = ChainModel(rdir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 1)
    @test 0 <= eval_scores(evaluate_counts(docs, rspecs, mr, Lexicon())).uas <= 1
    # domain variables: a pruned head direction removes the candidates on that side
    sd = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    solve!(sd, OracleModel())
    tk_ = sd.layers[:token]
    t_ = findfirst(t -> tk_[:attach, t] != NONE && tk_[:attach, t] != PENDING &&
      tk_[:first_char, Int(tk_[:attach, t])] < tk_[:first_char, t], 1:length(tk_))     # its head on the left
    for n in [:attach; SeqLayers.headnames(tk_)]
      tk_[n, t_] = PENDING
    end
    SeqLayers.refresh!(sd)
    cs = SeqLayers.attach_candidates(sd, t_)
    L, R, Rt = (SeqLayers.headcode(:head_dir, x) for x in ("left", "right", "root"))
    props = Dict(:head_dir => (UInt32[L, R, Rt], [0.9995, 0.0001, 0.0004]))
    alive, D = SeqLayers.propagate_domains(sd, t_, cs, fill(1 / length(cs), length(cs)), props; eps = 1e-3)
    @test D[:head_dir] == Set([L])
    @test all(SeqLayers.head_value(sd, t_, cs[j], :head_dir) == L for j in alive)
    @test length(alive) == count(h -> SeqLayers.head_value(sd, t_, h, :head_dir) == L, cs)
    c1, n1 = SeqLayers.domain_choice(sd, t_, cs, fill(1 / length(cs), length(cs)), props; mode = :product)
    @test SeqLayers.head_value(sd, t_, c1, :head_dir) == L && n1 == length(alive)
    st0 = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    st1 = SeqLayers.copy_state(st0)
    solve!(st1, mg)
    @test all(st0.layers[l].grid == SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex).layers[l].grid for l in st0.order)
    # maxdev target mode: with no restarts allowed, chains in step = independent chains
    st = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    solve!(st, OracleModel())
    plain = ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1)
    nores = ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1, target = true, maxrestarts = 0)
    keys_ = [(l, k, i) for l in (:char, :token) for (k, dd) in enumerate(st.layers[l].spec.decisions)
             for i in 1:st.layers[l].ncols if !isempty(dd.candidates(st, i))]
    restore = [SeqLayers.default_mask(st, key) for key in keys_[1:min(end, 40)]]   # make them pending again
    @test all(decide(plain, st, key) == decide(nores, st, key) for key in keys_[1:min(end, 40)] if
              SeqLayers.decisiondef(st, key).ready(st, key[3]))
    foreach(f -> f(), reverse(restore))
    for rs in (:immediate, :exhaust)
      r = evaluate(docs, specs, ChainModel(dir; select = :maxdev, leftovers = :backoff1, target = true,
        restart = rs, mincount = 1), lex)
      @test 0 <= r.token_f1 <= 1
    end
    @test_throws ErrorException ChainModel(dir; select = :clr_rms, target = true, restart = :exhaust)

    # joint tables and record storage are exact: the same decisions with and without joint tables
    # (all-frequent and mixed) and for any keepfrac, for per-candidate chains, decision-level
    # chains and target mode
    restore = [SeqLayers.default_mask(st, key) for key in keys_[1:min(end, 60)]]
    ready = [key for key in keys_[1:min(end, 60)] if SeqLayers.decisiondef(st, key).ready(st, key[3])]
    for kw in ((select = :maxdev, leftovers = :backoff1), (select = :clr_rms, leftovers = :backoff),
               (select = :maxdev, leftovers = :backoff1, target = true, restart = :exhaust))
      base = ChainModel(dir; mincount = 1, kw...)
      for (jm, kf) in ((1, 0.1), (3, 0.1), (0, 1.0), (0, 0.001), (3, 0.001))
        jt = ChainModel(dir; mincount = 1, jointmin = jm, keepfrac = kf, kw...)
        jm > 0 && @test !isempty(jt.pairs)
        @test all(decide(base, st, key) == decide(jt, st, key) for key in ready)
      end
    end
    # sampling: with a huge c nothing is sampled, so the round-by-round chains (and leftovers) must
    # give exactly what independent chains give; with c = 30 decoding still works
    for kw in ((select = :maxdev, leftovers = :backoff1), (select = :clr_rms, leftovers = :backoff))
      base = ChainModel(dir; mincount = 1, kw...)
      huge = ChainModel(dir; mincount = 1, sample = 1e30, kw...)
      @test all(decide(base, st, key) == decide(huge, st, key) for key in ready)
    end
    s30 = ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1, jointmin = 3, sample = 30.0)
    @test all(isfinite(decide(s30, st, key)[2]) || decide(s30, st, key)[2] == Inf for key in ready)
    @test SeqLayers.samplecap(s30, [0.01, 0.5, 0.04]) == ceil(Int, 30 / (0.04 / 1.04))
    @test SeqLayers.samplecap(m, [0.01, 0.5]) == SeqLayers.EXACT
    @test collect(SeqLayers.sample_positions(10, 5)) == [1, 3, 5, 7, 9]
    foreach(f -> f(), reverse(restore))

    # climbing: an item climbs to the finest ring that supports the label (support counted for that
    # label); its records and counts match brute force
    tk, lab = recs[1].typekey, recs[1].label
    q = 4                                                  # one slot, by position
    val(r) = r.typekey == tk && r.n >= q ? items[r.start+q-1] : SeqLayers.ABSENT
    vals = Dict{UInt64,Int}()                              # records of label `lab` per value
    for r in recs
      x = val(r)
      x != SeqLayers.ABSENT && r.label == lab && (vals[x] = get(vals, x, 0) + 1)
    end
    xs3 = first.(sort(collect(vals); by = p -> -p[2]))[1:3]
    brute(S) = [j for (j, r) in enumerate(recs) if val(r) in S]
    for (mc, expect) in ((1, xs3[1:1]), (vals[xs3[1]] + 1, xs3[1:2]), (vals[xs3[1]] + vals[xs3[2]] + 1, xs3))
      mc_ = ChainModel(dir; mincount = mc)
      mc_.climb[xs3[1]] = SeqLayers.Climb(xs3, [1, 2, 3], Vector{UInt64}[])
      cst = SeqLayers.ChainState(1.0, mc_.typerecs[tk])
      e = only(SeqLayers.extend_all(mc_, cst, 0, [SeqLayers.climbkey!(mc_, xs3[1], lab)]))
      @test e[1].base == brute(expect)
      @test e[2] == SeqLayers.labelcounts(mc_, brute(expect))
    end
    # outer rings count too, and decoding with climbing runs
    mo = ChainModel(dir; mincount = vals[xs3[1]] + vals[xs3[2]] + 1)
    mo.climb[xs3[1]] = SeqLayers.Climb(xs3[1:1], [1], [xs3[1:2], xs3])
    cst = SeqLayers.ChainState(1.0, mo.typerecs[tk])
    @test only(SeqLayers.extend_all(mo, cst, 0, [SeqLayers.climbkey!(mo, xs3[1], lab)]))[1].base == brute(xs3)
    mo.climb[xs3[2]] = SeqLayers.Climb(xs3[[2, 1]], [1, 2], Vector{UInt64}[])
    @test 0 <= evaluate(docs, specs, mo, lex).token_f1 <= 1
    # per-level counts for leftovers (climbleft = :specific) match brute force at every level
    ml = ChainModel(dir; mincount = 1, climbleft = :specific)
    ml.climb[xs3[1]] = SeqLayers.Climb(xs3, [1, 2, 3], Vector{UInt64}[])
    vk = SeqLayers.climbkey!(ml, xs3[1], lab)
    per = SeqLayers.climblevels(ml, SeqLayers.ChainState(1.0, ml.typerecs[tk]), 0, vk)
    @test all(per[L] == SeqLayers.labelcounts(ml, brute(xs3[1:L])) for L in 1:3)
    for cl in (:specific, :none)
      mc2 = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :backoff1, climbleft = cl)
      mc2.climb[xs3[1]] = SeqLayers.Climb(xs3, [1, 2, 3], Vector{UInt64}[])
      @test 0 <= evaluate(docs, specs, mc2, lex).token_f1 <= 1
    end
    # greedy leftovers: with an impossible minimum context nothing is added, so it equals no leftovers;
    # with minctx 1 and 2 (and both criteria) decoding works
    gnone = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :none)
    gbig = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, minctx = 1000)
    st1 = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    solve!(st1, OracleModel())
    k1 = [(l, k, i) for l in (:char, :token) for (k, dd) in enumerate(st1.layers[l].spec.decisions)
          for i in 1:st1.layers[l].ncols if !isempty(dd.candidates(st1, i))][1:40]
    r1 = [SeqLayers.default_mask(st1, key) for key in k1]
    rd1 = [key for key in k1 if SeqLayers.decisiondef(st1, key).ready(st1, key[3])]
    @test all(decide(gnone, st1, key) == decide(gbig, st1, key) for key in rd1)
    foreach(f -> f(), reverse(r1))
    for mc in (1, 2), gb in (:correction, :odds)
      g = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, minctx = mc, greedyby = gb)
      @test 0 <= evaluate(docs, specs, g, lex).token_f1 <= 1
    end
    @test_throws ErrorException ChainModel(dir; leftovers = :greedy, greedyby = :maxdev)
    # greedy speed-ups: an unreachable relevance threshold skips every leftover (= no leftovers);
    # a cap of 0 leaves every context empty, so with minctx 1 nothing is included either
    for kw in ((relevance = 1e9,), (maxctx = 0,))
      g = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, kw...)
      @test all(decide(gnone, st1, key) == decide(g, st1, key) for key in rd1)
    end
    for kw in ((maxctx = 2,), (relevance = 0.1,), (maxctx = 2, relevance = 0.1))
      g = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, kw...)
      @test 0 <= evaluate(docs, specs, g, lex).token_f1 <= 1
    end
    # position-based passes are exact: the same decisions as reading every item
    pos_ = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :backoff1, keepfrac = 1.0)
    all_ = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :backoff1, keepfrac = 1.0, usepos = false)
    @test any(>(0), pos_.slotof)
    @test all(decide(pos_, st1, key) == decide(all_, st1, key) for key in rd1)
    gpos = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, minctx = 2, keepfrac = 1.0)
    gall = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :greedy, minctx = 2, keepfrac = 1.0, usepos = false)
    @test all(decide(gpos, st1, key) == decide(gall, st1, key) for key in rd1)
    # grouped decisions are exact: fewer records, the same counts for feature sets, and the same
    # decisions, for several chain settings
    mu0 = ChainModel(dir; mincount = 1)
    mg0 = ChainModel(dir; mincount = 1, group = true)
    @test length(mg0.recs) < length(mu0.recs) && any(!iszero, mg0.lsid)
    for r in recs[1:37:end]
      S = present(r)[1:min(end, 3)]
      isempty(S) && continue
      @test first(SeqLayers.setcounts(mu0, r.typekey, S)) == first(SeqLayers.setcounts(mg0, r.typekey, S))
    end
    for kw in ((select = :maxdev, leftovers = :backoff1), (select = :maxdev, leftovers = :none, jointmin = 3),
               (select = :clr_rms, leftovers = :backoff), (select = :maxdev, leftovers = :greedy, minctx = 1, keepfrac = 1.0))
      mu = ChainModel(dir; mincount = 1, kw...)
      mg = ChainModel(dir; mincount = 1, group = true, kw...)
      @test all(decide(mu, st1, key) == decide(mg, st1, key) for key in rd1)
    end
    @test_throws ErrorException ChainModel(dir; group = true, poolm = 3)
    # persistent counts: a model given the counts another one collected decides exactly the same,
    # and collects nothing new for those decisions
    for kw in ((select = :maxdev, leftovers = :backoff1), (select = :maxdev, leftovers = :greedy, minctx = 1, keepfrac = 1.0))
      ma = ChainModel(dir; mincount = 1, group = true, kw...)
      ma.keepcounts = true; ma.keepmin = 0
      da = [decide(ma, st1, key) for key in rd1]
      @test !isempty(ma.newcounts)
      mb = ChainModel(dir; mincount = 1, group = true, kw...)
      mb.pcounts = copy(ma.newcounts); mb.keepcounts = true
      @test [decide(mb, st1, key) for key in rd1] == da
      @test isempty(mb.newcounts)
      @test SeqLayers.unpackcounts(SeqLayers.packcounts(ma.newcounts)) == ma.newcounts
      # saved and memory-mapped: the same counts, and the same decisions
      cdir = mktempdir()
      SeqLayers.save_counts(cdir, "sig-1", SeqLayers.packcounts(ma.newcounts))
      pc = SeqLayers.load_counts(cdir, "sig-1")
      @test pc !== nothing && length(pc) == length(ma.newcounts)
      @test all(SeqLayers.pcget(pc, k) == c for (k, c) in ma.newcounts)
      @test SeqLayers.pcget(pc, UInt64(12345)) === nothing
      @test SeqLayers.load_counts(cdir, "other-events") === nothing          # saved for other events
      mc = ChainModel(dir; mincount = 1, group = true, kw...)
      mc.pcounts = pc
      @test [decide(mc, st1, key) for key in rd1] == da
      # merging two disjoint packed caches
      ks = collect(keys(ma.newcounts))
      h1 = Dict(k => ma.newcounts[k] for k in ks[1:2:end]); h2 = Dict(k => ma.newcounts[k] for k in ks[2:2:end])
      @test SeqLayers.unpackcounts(SeqLayers.mergecounts(SeqLayers.packcounts(h1), SeqLayers.packcounts(h2))) == ma.newcounts
      # a truncated array file is refused
      f = joinpath(cdir, "counts_c2.bin")
      write(f, read(f)[1:max(end - 4, 0)])
      @test SeqLayers.load_counts(cdir, "sig-1") === nothing
    end
    # candidate pool: orderings exist and hold slots and pairs; a pool as large as every entry,
    # without pairs, chooses what the plain chain chooses; pooled chains decode
    pm = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :none, poolm = 3, poollam = 0.0, poolgain = -Inf)
    @test !isempty(pm.pool) && all(e -> 1 <= e[1] && 0 <= e[2], first(values(pm.pool)))
    @test any(e -> e[2] != 0, reduce(vcat, collect(values(pm.pool))))
    singles_only = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :none, poolm = 1000, poolgain = 1e9)
    @test all(e -> e[2] == 0, reduce(vcat, collect(values(singles_only.pool))))
    plain_ = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :none)
    @test all(decide(plain_, st1, key) == decide(singles_only, st1, key) for key in rd1)
    # pair counts match brute force: at the empty prefix and after one feature, through the short-list
    # route (keepfrac 1: every value has a list) and through the shared pass (keepfrac tiny: none do)
    tk2 = recs[1].typekey
    prows = [j for (j, r) in enumerate(recs) if r.typekey == tk2]
    itemsat(j) = items[recs[j].start:recs[j].start+recs[j].n-1]
    r0 = itemsat(prows[1])
    for kf in (1.0, 1e-9), (qa, qb, qc) in ((2, 3, 1), (4, 5, 2))
      mp = ChainModel(dir; mincount = 1, keepfrac = kf)
      x, y, z = r0[qa], r0[qb], r0[qc]
      st0_ = SeqLayers.ChainState(1.0, mp.typerecs[tk2])
      e0 = only(SeqLayers.extend_pairs(mp, st0_, 0, [(x, y)]))
      want0 = [j for j in prows if x in itemsat(j) && y in itemsat(j)]
      @test e0[2] == SeqLayers.labelcounts(mp, want0)
      ez = only(SeqLayers.extend_all(mp, st0_, 0, [z]))
      SeqLayers.advance!(mp, st0_, z, ez, 1.0)
      e1 = only(SeqLayers.extend_pairs(mp, st0_, 1, [(x, y)]))
      want1 = [j for j in want0 if z in itemsat(j)]
      @test e1[2] == SeqLayers.labelcounts(mp, want1)
    end
    for pmm in (3, 5)
      g = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :none, poolm = pmm)
      @test 0 <= evaluate(docs, specs, g, lex).token_f1 <= 1
    end
    # without climbing items, climbleft changes nothing
    st0 = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    solve!(st0, OracleModel())
    ks = [(l, k, i) for l in (:char, :token) for (k, dd) in enumerate(st0.layers[l].spec.decisions)
          for i in 1:st0.layers[l].ncols if !isempty(dd.candidates(st0, i))][1:40]
    rs = [SeqLayers.default_mask(st0, key) for key in ks]
    rdy = [key for key in ks if SeqLayers.decisiondef(st0, key).ready(st0, key[3])]
    a0 = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :backoff1)
    for cl in (:specific, :none)
      a1 = ChainModel(dir; mincount = 1, select = :maxdev, leftovers = :backoff1, climbleft = cl)
      @test all(decide(a0, st0, key) == decide(a1, st0, key) for key in rdy)
    end
    foreach(f -> f(), reverse(rs))

    # screening: higher thresholds screen more; clr_rms (centered RMS) never exceeds maxdev (largest change)
    lo_ = ChainModel(dir; mincount = 1, jointmin = 1, screen = 0.05).screened
    hi_ = ChainModel(dir; mincount = 1, jointmin = 1, screen = 0.5).screened
    @test issubset(lo_, hi_) && length(hi_) > length(lo_)
    @test issubset(lo_, ChainModel(dir; mincount = 1, jointmin = 1, screen = 0.05, screenby = :clr_rms).screened)
    @test isempty(m.screened)
    # rare items (fewer than mincount records) are screened whatever the threshold
    m5 = ChainModel(dir; mincount = 5, jointmin = 1, screen = 1e-9)
    @test all(x in m5.screened for (x, k) in m5.itemcount if k < 5)
    all_ = ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1, jointmin = 1, screen = Inf)
    @test length(all_.screened) == length(all_.itemcount)
    for sm in (ChainModel(dir; select = :maxdev, leftovers = :backoff1, mincount = 1, jointmin = 1, screen = 0.05), all_)
      r = evaluate(docs, specs, sm, lex)
      @test 0 <= r.token_f1 <= 1
    end
    @test_throws ErrorException ChainModel(dir; jointmin = 1, screen = 0.1, screenby = :avglogodds)
    @test_throws ErrorException ChainModel(dir; screen = 0.1)                # screening needs joint tables
  end

  @testset "gold-state recording" begin
    d = only(docs)
    st = SeqLayers.State(d, 1:length(d.text), true, 0, specs, Lexicon())
    solve!(st, OracleModel())
    snapshot() = Dict(l => copy(st.layers[l].grid) for l in st.order)
    before = snapshot()
    ndecisions = 0
    for l in st.order, (k, dd) in enumerate(st.layers[l].spec.decisions), i in 1:st.layers[l].ncols
      isempty(dd.allcandidates(st, i)) && continue
      ndecisions += 1
      restore = dd.mask(st, (l, k, i))
      @test st.layers[l][dd.name, i] == PENDING
      restore()
    end
    @test snapshot() == before && isempty(st.hidden)     # masks undo cleanly
    # under a gap's mask, the token that gap is inside reads as pending (no leak of the answer)
    chars = st.layers[:char]
    i = findfirst(j -> j > 1 && chars[:gap, j-1] == SeqLayers.GAP_I && chars[:gap, j] != NONE, 1:length(chars))
    path = FeatPath([:left_neighbor, :token_of], :form)
    @test follow(st, :char, i, path) != PENDING
    restore = SeqLayers.gap_mask(st, (:char, 1, i))
    @test follow(st, :char, i, path) == PENDING
    restore()
    # the recorder writes one record per candidate of every decision in the closed region
    dir = record_events(docs, specs, Lexicon(), mktempdir())
    recs, _ = SeqLayers.load_events(dir)
    @test count(r -> r.chosen == 1, recs) == ndecisions
  end

  @testset "head-property decisions" begin
    S = SeqLayers
    specs = default_specs()
    d = only(docs)
    st = S.State(d, 1:length(d.text), true, 0, specs, Lexicon())
    solve!(st, OracleModel())
    toks = st.layers[:token]
    n = toks.ncols
    @test all(toks[name, t] != PENDING for t in 1:n for name in keys(S.HEADSETS))
    # gold heads agree with the gold head properties, and word distances match the tree
    @test all(S.head_consistent(st, t, toks[:attach, t]) for t in 1:n)
    like = findfirst(t -> S.tokstr(st, t) == "like", 1:n)
    rain = findfirst(t -> S.tokstr(st, t) == "rain" && toks[:attach, t] == like, 1:n)
    I = findfirst(t -> S.tokstr(st, t) == "I", 1:n)
    @test S.word_distance(st, I, like) == 3 && S.word_distance(st, rain, like) == 1
    @test S.HEADSETS.head_upos[toks[:head_upos, rain]] == "VERB" && S.HEADSETS.head_dir[toks[:head_dir, rain]] == "left"
    @test S.HEADSETS.head_dist[toks[:head_dist, I]] == "3-5" && S.isrootlabel(:head_dir, toks[:head_dir, like])
    # a wrong head property excludes the gold head (strict narrowing)
    old = toks[:head_dir, rain]
    toks[:head_dir, rain] = S.headcode(:head_dir, "right")
    @test !(like in S.attach_candidates(st, rain))
    toks[:head_dir, rain] = old
    # inference: with an attachment hidden, a single remaining candidate is the gold head
    st.infer = true
    for t in 1:n
      gold = toks[:attach, t]
      toks[:attach, t] = PENDING
      S.refresh!(st)
      single = length(S.attach_candidates(st, t)) == 1
      S.infer_attach!(st, t)
      @test single ? toks[:attach, t] == gold : toks[:attach, t] == PENDING
      toks[:attach, t] = gold
      S.refresh!(st)
    end
    # backward step: an attachment determines the head properties (gold here, since the oracle's tags are gold)
    gold = Dict((t, name) => toks[name, t] for t in 1:n for name in keys(S.HEADSETS))
    kh = Dict(dd.name => k for (k, dd) in enumerate(toks.spec.decisions))
    for t in 1:n, name in keys(S.HEADSETS)
      toks[name, t] = PENDING
    end
    st.agenda[(:token, kh[:head_dist], rain)] = 0.0              # a derived decision leaves the agenda
    S.derive_heads!(st, 1)
    @test all(toks[name, t] == gold[(t, name)] for t in 1:n for name in keys(S.HEADSETS))
    @test !haskey(st.agenda, (:token, kh[:head_dist], rain))
    @test all(S.isrootlabel(name, toks[name, like]) for name in keys(S.HEADSETS))
    # a head whose tag is still pending: head_upos waits, and is filled once the tag is decided
    likeupos = toks[:upos, like]
    toks[:upos, like] = PENDING; toks[:head_upos, rain] = PENDING
    S.derive_heads!(st, 1)
    @test toks[:head_upos, rain] == PENDING
    toks[:upos, like] = likeupos
    S.derive_heads!(st, 1)
    @test toks[:head_upos, rain] == gold[(rain, :head_upos)]
    # properties committed before the attachment are left alone
    toks[:head_dir, rain] = S.headcode(:head_dir, "right")
    S.derive_heads!(st, 1)
    @test S.HEADSETS.head_dir[toks[:head_dir, rain]] == "right"
    toks[:head_dir, rain] = gold[(rain, :head_dir)]
    # with the attachment committed, head properties wait to be derived (but the oracle decides them)
    @test !toks.spec.decisions[kh[:head_upos]].ready(st, rain)
    toks[:attach, rain] = PENDING
    @test toks.spec.decisions[kh[:head_upos]].ready(st, rain)
    toks[:attach, rain] = like
    st.infer = false
    @test toks.spec.decisions[kh[:head_upos]].ready(st, rain)
    st.infer = true
    # and the oracle (infer = false) derives nothing
    st.infer = false
    toks[:head_dist, rain] = PENDING
    S.derive_heads!(st, 1)
    @test toks[:head_dist, rain] == PENDING
    toks[:head_dist, rain] = gold[(rain, :head_dist)]
    st.infer = true
    # masks: an attachment hides the head properties, a head property hides the attachment
    k_att = findfirst(dd -> dd.name == :attach, toks.spec.decisions)
    k_hu = findfirst(dd -> dd.name == :head_upos, toks.spec.decisions)
    restore = S.attach_mask(st, (:token, k_att, rain))
    @test toks[:attach, rain] == PENDING && all(toks[name, rain] == PENDING for name in keys(S.HEADSETS))
    restore()
    restore = S.head_mask(st, (:token, k_hu, rain))
    @test toks[:attach, rain] == PENDING && toks[:head_upos, rain] == PENDING && toks[:head_dir, rain] != PENDING
    restore()
    @test toks[:attach, rain] == like && S.head_consistent(st, rain, like)
  end

  @testset "word classes" begin
    dir = mktempdir()
    # level sizes 4 and 2; "fell" and "slowly" aren't in the table ("lowly" shares slowly's suffix)
    write(joinpath(dir, "wordclasses.tsv"), "#sizes\t4\t2\n" * join(["the\t100\t1\t1", "rain\t50\t2\t1",
      "economy\t40\t2\t1", "grew\t30\t3\t2", "like\t30\t3\t2", "i\t90\t4\t2", "do\t60\t3\t2",
      "n't\t20\t4\t2", ".\t200\t4\t2", ",\t150\t4\t2", "u.s.\t10\t2\t1", "lowly\t5\t1\t1"], '\n') * "\n")
    wc = load_wordclasses(dir)
    @test wc.sizes == [4, 2]
    @test SeqLayers.wclass_candidates(wc, 1, "rain") == UInt32[2]
    @test SeqLayers.wclass_candidates(wc, 2, "grew") == UInt32[2]
    @test SeqLayers.wclass_candidates(wc, 1, "slowly") == UInt32[5, 1]       # unknown, then lowly's class
    @test SeqLayers.wclass_candidates(wc, 1, "fell") == UInt32[5]
    @test SeqLayers.wclass_gold(wc, 2, "fell") == 3 && SeqLayers.wclass_gold(wc, 2, "the") == 1
    wspecs = default_specs(wordclasses = wc)
    toks = wspecs[2]
    @test [d.name for d in toks.decisions][end-1:end] == [:wclass4, :wclass2]
    @test length(toks.context) == length(specs[2].context) + 2
    r = evaluate(docs, wspecs, OracleModel(), Lexicon(); words = 128, edge = 4)
    @test r.uas == 1.0 && r.las == 1.0
    # recorded, the classes reach the attachment items (2 more paths: dependent and candidate)
    edir = mktempdir()
    record_events(docs, wspecs, Lexicon(), edir)
    ev = first(SeqLayers.load_events(edir))
    edir0 = mktempdir()
    record_events(docs, specs, Lexicon(), edir0)
    ev0 = first(SeqLayers.load_events(edir0))
    tk = SeqLayers.typekey(:token, 1)
    n(e) = [Int(c.n) for c in e if c.typekey == tk]
    @test n(ev) == n(ev0) .+ 4
    # class slots, for keeping classes out of the leftovers: the last 2 x 2 attachment slots, context paths 23:24
    ws = SeqLayers.wclass_slots(wspecs)
    na = 2 + 2 * (length(SeqLayers.LINK_PATHS) + 2)
    @test ws[SeqLayers.typekey(:token, 1)] == Set(na-3:na) && ws[SeqLayers.typekey(:token, 2)] == Set(23:24)
    @test isempty(SeqLayers.wclass_slots(specs))
    mw = ChainModel(edir; select = :maxdev, leftovers = :greedy, minctx = 1, mincount = 1)
    mw.noleft = ws
    rw = evaluate(docs, wspecs, mw, Lexicon(); words = 128, edge = 4)
    @test 0 <= rw.uas <= 1
    # gold values: "fell" and "slowly" are unknown at both levels, so their chosen label is the unknown code
    for (k, unk) in ((length(toks.decisions) - 1, 5), (length(toks.decisions), 3))
      ch = [c.label for c in ev if c.typekey == SeqLayers.typekey(:token, k) && c.chosen == 1]
      @test length(ch) == 16 && count(==(unk), ch) == 2
    end
  end

  @testset "suffix features" begin
    sspecs = default_specs(suffixes = [3])
    @test length(sspecs[2].context) == length(specs[2].context) + 1
    @test sspecs[2].context[end].feat == :suffix3
    r = evaluate(docs, sspecs, OracleModel(), Lexicon(); words = 128, edge = 4)
    @test r.uas == 1.0 && r.las == 1.0
    edir = mktempdir()
    record_events(docs, sspecs, Lexicon(), edir)
    ev = first(SeqLayers.load_events(edir))
    edir0 = mktempdir()
    record_events(docs, specs, Lexicon(), edir0)
    ev0 = first(SeqLayers.load_events(edir0))
    tk = SeqLayers.typekey(:token, 1)
    @test [Int(c.n) for c in ev if c.typekey == tk] == [Int(c.n) for c in ev0 if c.typekey == tk] .+ 2
  end

  @testset "relevance gates" begin
    nspecs = default_specs(noise = 7)
    @test nspecs[2].context[end].feat == :noise
    @test evaluate(docs, nspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    edir = mktempdir()
    record_events(docs, nspecs, Lexicon(), edir)
    m0 = ChainModel(edir; select = :maxdev, leftovers = :greedy, minctx = 1, mincount = 1)
    @test isempty(m0.gates)
    mg = ChainModel(edir; select = :maxdev, leftovers = :greedy, minctx = 1, mincount = 1, gate = 0.05)
    tk = SeqLayers.typekey(:token, 1)
    @test haskey(mg.gates, tk) && !haskey(mg.gates, SeqLayers.typekey(:token, 2))   # attach only (unlabeled)
    g = mg.gates[tk]
    @test g.nslots == 2 + 2 * (length(SeqLayers.LINK_PATHS) + 1)
    @test 1 in g.gateslots && 2 in g.gateslots                  # distance and direction have few values
    @test all(e -> 0 <= e < Inf, g.effect)
    # a record admits its always-slots plus those opened by its gate values, and nothing else
    Ic = [hash(k) for k in 1:g.nslots]                          # unknown items: only the always-slots
    @test SeqLayers.admitted(mg, tk, Ic) == findall(g.always)
    @test SeqLayers.admitted(mg, UInt64(1), Ic) == collect(1:g.nslots)   # no gates for the type: all
    mh = ChainModel(edir; select = :maxdev, leftovers = :greedy, minctx = 1, mincount = 1, gate = 1e9)
    @test !any(mh.gates[tk].always) && isempty(mh.gates[tk].byvalue)
    r = evaluate(docs, nspecs, mg, Lexicon(); words = 128, edge = 4)
    @test 0 <= r.uas <= 1
  end

  @testset "root candidate items absent" begin
    rspecs = default_specs(rootabsent = true)
    @test evaluate(docs, rspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    edir, edir0 = mktempdir(), mktempdir()
    record_events(docs, rspecs, Lexicon(), edir); record_events(docs, specs, Lexicon(), edir0)
    (rr, ri), (r0, i0) = SeqLayers.load_events(edir), SeqLayers.load_events(edir0)
    tk = SeqLayers.typekey(:token, 1)
    a = [r for r in rr if r.typekey == tk]; b = [r for r in r0 if r.typekey == tk]
    @test length(a) == length(b)
    cand(I, r) = [I[r.start+s-1] for s in 4:2:r.n]             # candidate-side slots
    # some candidate (the root) has all its candidate-side items absent with rootabsent, none without
    @test any(r -> all(==(SeqLayers.ABSENT), cand(ri, r)), a)
    @test !any(r -> all(==(SeqLayers.ABSENT), cand(i0, r)), b)
  end

  @testset "greedy leftovers: redundant pool, chain-sized contexts" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    kw = (select = :maxdev, leftovers = :greedy, minctx = 1, mincount = 1)
    for (red, cs) in ((false, :min), (true, :min), (false, :chain), (true, :chain), (false, :adaptive), (true, :adaptive))
      m = ChainModel(edir; kw..., redundant = red, ctxsize = cs)
      @test m.opts.redundant == red && m.opts.ctxchain == (cs == :chain) && m.opts.adaptctx == (cs == :adaptive)
      r = evaluate(docs, specs, m, Lexicon(); words = 128, edge = 4)
      @test 0 <= r.uas <= 1
    end
    @test_throws ErrorException ChainModel(edir; kw..., ctxsize = :other)
    # soft pool: generalizations of chain forms weighted by how well they stand in for the form; a huge
    # sigma gives them weight ~1, so results match the hard pool
    gen = SeqLayers.generalization_slots(specs)
    @test haskey(gen[SeqLayers.typekey(:token, 1)], 3) && haskey(gen[SeqLayers.typekey(:token, 2)], 1)   # dep form; form
    mh = ChainModel(edir; kw..., redundant = true)
    ms = ChainModel(edir; kw..., redundant = true, poolsigma = 1e9); ms.generalize = gen
    @test evaluate(docs, specs, mh, Lexicon(); words = 128, edge = 4).uas ≈ evaluate(docs, specs, ms, Lexicon(); words = 128, edge = 4).uas
    m2 = ChainModel(edir; kw..., redundant = true, poolsigma = 0.1); m2.generalize = gen
    @test 0 <= evaluate(docs, specs, m2, Lexicon(); words = 128, edge = 4).uas <= 1
    ml = ChainModel(edir; kw..., redundant = true, poolsigma = 1e9, poolmode = :local); ml.generalize = gen
    @test evaluate(docs, specs, mh, Lexicon(); words = 128, edge = 4).uas ≈ evaluate(docs, specs, ml, Lexicon(); words = 128, edge = 4).uas
    ml2 = ChainModel(edir; kw..., redundant = true, poolsigma = 0.1, poolmode = :local); ml2.generalize = gen
    @test 0 <= evaluate(docs, specs, ml2, Lexicon(); words = 128, edge = 4).uas <= 1
    # local mode records each step's alternatives
    tr1 = ml2.typerecs[SeqLayers.typekey(:token, 1)]; r1 = ml2.recs[tr1[1]]
    I1 = [x for x in ml2.items[r1.start:r1.start+r1.n-1]]
    st1, _ = SeqLayers.run_single(ml2, I1, UInt32(0), 0.05, tr1)
    @test length(st1.alts) == SeqLayers.depth(st1)
  end

  @testset "maxdev step scores" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    for ss in (:abs, :signed, :maxmin, :maxmin2)
      m = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1, stepscore = ss)
      @test m.opts.stepscore == ss
      @test 0 <= evaluate(docs, specs, m, Lexicon(); words = 128, edge = 4).uas <= 1
    end
    @test_throws ErrorException ChainModel(edir; select = :maxdev, stepscore = :other)
  end

  @testset "top-two lockstep chains" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    m1 = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1)
    m2 = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1, lockstep = :top2)
    @test m2.opts.top2 && !m1.opts.top2
    # chains are cut where they leave the top two: never longer than the independent chain
    tr = m1.typerecs[SeqLayers.typekey(:token, 1)]
    I = [[x for x in m1.items[r.start:r.start+r.n-1]] for r in m1.recs[tr[1:4]]]
    labs, pri = fill(UInt32(0), 4), fill(0.05, 4)
    st2, _ = SeqLayers.run_top2(m2, I, labs, pri, tr)
    for c in 1:4
      st1, _ = SeqLayers.run_single(m1, I[c], labs[c], pri[c], tr)
      @test SeqLayers.depth(st2[c]) <= SeqLayers.depth(st1)
      @test st2[c].members == st1.members[1:SeqLayers.depth(st2[c])]   # a prefix of its own chain
    end
    @test 0 <= evaluate(docs, specs, m2, Lexicon(); words = 128, edge = 4).uas <= 1
  end

  @testset "contest re-ranking" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    m = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1, contest = true)
    @test m.opts.contest
    tk = SeqLayers.typekey(:token, 1)
    tr = m.typerecs[tk]
    I = [[x for x in m.items[r.start:r.start+r.n-1]] for r in m.recs[tr[1:3]]]
    la, lb = SeqLayers.contest_chain(m, I[1], I[2], UInt32(0), UInt32(0), 0.05, 0.05, tr, tk)
    @test isfinite(la) && isfinite(lb)
    # a candidate against itself: no slot changes the ratio, so both stay at the prior
    @test SeqLayers.contest_chain(m, I[1], I[1], UInt32(0), UInt32(0), 0.05, 0.05, tr, tk) == (log(0.05), log(0.05))
    sc = SeqLayers.contest!(m, [log(0.2), log(0.1), log(0.01)], I, fill(UInt32(0), 3), fill(0.05, 3), tr, tk)
    @test length(sc) == 3 && sc[3] == log(0.01)                     # a third candidate below both keeps its score
    @test 0 <= evaluate(docs, specs, m, Lexicon(); words = 128, edge = 4).uas <= 1
  end

  @testset "pick the fewest-records feature among the top k" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    m0 = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1)
    m1 = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1, pickfewest = 1)   # k = 1: same as off
    m3 = ChainModel(edir; select = :maxdev, leftovers = :none, mincount = 1, pickfewest = 3)
    tk = SeqLayers.typekey(:token, 1)
    tr = m0.typerecs[tk]
    for r in m0.recs[tr[1:5]]
      I = [x for x in m0.items[r.start:r.start+r.n-1]]
      a, _ = SeqLayers.run_single(m0, I, UInt32(0), 0.05, tr)
      b, _ = SeqLayers.run_single(m1, I, UInt32(0), 0.05, tr)
      @test a.members == b.members
    end
    @test 0 <= evaluate(docs, specs, m3, Lexicon(); words = 128, edge = 4).uas <= 1
  end

  @testset "projectivity" begin
    d = only(docs)
    st = SeqLayers.State(d, 1:length(d.text), true, 0, specs, Lexicon())
    solve!(st, OracleModel())
    toks = st.layers[:token]
    tok(w) = findfirst(t -> SeqLayers.tokstr(st, t) == w, 1:toks.ncols)    # first token with this text
    # the gold trees are projective: no gold arc crosses another
    @test all(t -> !SeqLayers.crosses_arc(st, t, toks[:attach, t]), 1:toks.ncols)
    # "I don't like rain.": rain -> do would cross I -> like
    @test SeqLayers.crosses_arc(st, tok("rain"), UInt32(tok("do")))
    @test !SeqLayers.crosses_arc(st, tok("rain"), UInt32(tok("like")))
    # a root arc (from position 0) can't cover another word's arc end: n't as a root crosses I -> like
    @test SeqLayers.crosses_arc(st, tok("n't"), NONE)
    pspecs = default_specs(projective = true)
    @test evaluate(docs, pspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
  end

  @testset "sibling and child relations" begin
    rspecs = default_specs(relations = true)
    @test length(rspecs[2].context) == length(specs[2].context) + 23
    d = only(docs)
    st = SeqLayers.State(d, 1:length(d.text), true, 0, rspecs, Lexicon())
    solve!(st, OracleModel())
    toks = st.layers[:token]
    tok(w) = findfirst(t -> SeqLayers.tokstr(st, t) == w, 1:toks.ncols)
    r(w, k) = (u = toks[k, tok(w)]; u == PENDING ? "" : SeqLayers.tokstr(st, Int(u)))
    # "I don't like rain.": I, do, n't, rain and . all attach to like
    @test r("like", :dep_left1) == "n't" && r("like", :dep_right1) == "rain" && r("like", :dep_right2) == "."
    @test r("I", :right_sibling) == "do" && r("I", :inner_sibling) == "do" && r("I", :left_sibling) == "" && r("I", :outer_sibling) == ""
    @test r("n't", :left_sibling) == "do" && r("n't", :outer_sibling) == "do" && r("n't", :inner_sibling) == ""
    @test toks[:dep_right1_dist, tok("like")] == 1 && toks[:dep_right1_dist, tok("grew")] == 2   # ", slowly": 2 words
    @test toks[:inner_sibling_dir, tok("I")] == 2                                                    # do is to I's right
    # hypothetical siblings: n't's siblings under candidate like (with n't unattached) are do and rain
    toks[:attach, tok("n't")] = PENDING
    items = SeqLayers.hypothetical_sibling_items(st, tok("n't"), UInt32(tok("like")))
    @test length(items) == 14 && count(!=(SeqLayers.ABSENT), items) == 10       # left (do), right (rain): 3 each; outer (do): 4; no inner
    @test all(==(SeqLayers.ABSENT), SeqLayers.hypothetical_sibling_items(st, tok("n't"), NONE))
    @test evaluate(docs, rspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    # recorded: attachment records carry 2 x 9 children slots and 14 hypothetical-sibling slots more
    edir, edir0 = mktempdir(), mktempdir()
    record_events(docs, rspecs, Lexicon(), edir); record_events(docs, specs, Lexicon(), edir0)
    tk = SeqLayers.typekey(:token, 1)
    n(e) = [Int(c.n) for c in first(SeqLayers.load_events(e)) if c.typekey == tk]
    @test n(edir) == n(edir0) .+ (18 + 14)
  end

  @testset "signed head distance" begin
    sspecs = default_specs(headsdist = true)
    @test any(d -> d.name == :head_sdist, sspecs[2].decisions)
    @test !any(d -> d.name == :head_sdist, specs[2].decisions)          # off by default
    @test evaluate(docs, sspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    d = only(docs)
    st = SeqLayers.State(d, 1:length(d.text), true, 0, sspecs, Lexicon())
    solve!(st, OracleModel())
    toks = st.layers[:token]
    tok(w) = findfirst(t -> SeqLayers.tokstr(st, t) == w, 1:toks.ncols)
    lab(w) = SeqLayers.ALLHEADSETS.head_sdist[toks[:head_sdist, tok(w)]]
    @test lab("I") == "R3-5" && lab("rain") == "L1" && lab("like") == "root" && lab("n't") == "R1"
    # it narrows: with n't's head_sdist = R1, only like (one word to the right) is consistent
    t = tok("n't")
    @test SeqLayers.head_consistent(st, t, UInt32(tok("like")))
    @test !SeqLayers.head_consistent(st, t, UInt32(tok("do"))) && !SeqLayers.head_consistent(st, t, NONE)
    # and it's derived back from a committed attachment
    k = findfirst(dd -> dd.name == :head_sdist, toks.spec.decisions)
    toks[:head_sdist, t] = PENDING
    st.infer = true
    SeqLayers.derive_heads!(st, 0)
    @test lab("n't") == "R1"
  end

  @testset "pair steps in maxdev chains" begin
    edir = mktempdir()
    record_events(docs, specs, Lexicon(), edir)
    kw = (select = :maxdev, leftovers = :none, mincount = 1, jointmin = 2)
    m0 = ChainModel(edir; kw...)
    m1 = ChainModel(edir; kw..., pairsteps = true)
    @test m1.opts.pairsteps && !isempty(m1.pairs)
    tk = SeqLayers.typekey(:token, 1)
    tr = m0.typerecs[tk]
    for r in m0.recs[tr[1:6]]
      I = [x for x in m0.items[r.start:r.start+r.n-1]]
      a, ra = SeqLayers.run_single(m0, I, UInt32(0), 0.05, tr)
      b, rb = SeqLayers.run_single(m1, I, UInt32(0), 0.05, tr)
      @test allunique(b.members) && length(b.members) + length(rb) == length(I)   # each slot used at most once
    end
    for ps in (:total, :min, :increment, :synergy)
      m = ChainModel(edir; kw..., pairsteps = true, pairscore = ps)
      @test 0 <= evaluate(docs, specs, m, Lexicon(); words = 128, edge = 4).uas <= 1
    end
    # pairs first: a pair-covered slot never enters a chain alone (members come in twos)
    mp = ChainModel(edir; kw..., pairsteps = true, pairsfirst = true)
    for r in m0.recs[tr[1:6]]
      I = [x for x in m0.items[r.start:r.start+r.n-1]]
      cov = SeqLayers.pair_covered(mp, I, collect(eachindex(I)))
      st, _ = SeqLayers.run_single(mp, I, UInt32(0), 0.05, tr)
      @test iseven(count(x -> findfirst(==(x), I) in cov, st.members))
    end
    # when every present feature is pair-covered, a chain still grows by pairs
    grew = false
    for r in m0.recs[tr]
      I = [x for x in m0.items[r.start:r.start+r.n-1]]
      cov = SeqLayers.pair_covered(mp, I, collect(eachindex(I)))
      isempty(cov) && continue
      J = [s in cov ? x : SeqLayers.ABSENT for (s, x) in enumerate(I)]     # keep only covered slots
      st, _ = SeqLayers.run_single(mp, J, UInt32(0), 0.05, tr)
      SeqLayers.depth(st) > 0 && (grew = true; break)
    end
    @test grew
  end

  @testset "PP-attachment table items" begin
    dir = mktempdir()
    write(joinpath(dir, "wordclasses.tsv"), "#sizes\t2\n" * join(["like\t50\t1", "rain\t40\t2", "grew\t30\t1", "economy\t20\t2"], '\n') * "\n")
    write(joinpath(dir, "pp.tsv"), "#total\t10000\nbase\tin\t100\nclass\t1\t1000\tin\t80\nclass\t2\t1000\tin\t5\nform\tlike\t500\tin\t60\n")
    tab = load_pptable(joinpath(dir, "pp.tsv"), dir)
    @test SeqLayers.pp_bucket(80, 1000, 0.01) == 4            # 7.9 per 100 after smoothing: 4-8
    @test SeqLayers.pp_bucket(100, 1000, 0.01) == 5           # 9.8 per 100: 8-16
    @test SeqLayers.pp_bucket(0, 1000, 0.01) == 1
    @test SeqLayers.pp_relbucket(2.5, 0.81) == 6               # 3.1x the best rival: 2-4x
    @test SeqLayers.pp_relbucket(0.81, 2.5) == 3               # 0.32x: 1/4-1/2
    @test SeqLayers.pp_relbucket(1.0, 1.0) == 5
    @test SeqLayers.pp_relbucket(0.01, 1.0) == 1
    @test SeqLayers.pp_relbucket(1.0, nothing) == 9
    @test SeqLayers.pp_rate(tab, "like", "in", 2) ≈ 100 * (60 + 20 * 0.01) / 520
    @test SeqLayers.pp_rate(tab, "nosuch", "in", 1) === nothing
    @test evaluate(docs, default_specs(pptable = tab, pprel = true), OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    pspecs = default_specs(pptable = tab)
    @test evaluate(docs, pspecs, OracleModel(), Lexicon(); words = 128, edge = 4).uas == 1.0
    edir, edir0 = mktempdir(), mktempdir()
    record_events(docs, pspecs, Lexicon(), edir); record_events(docs, specs, Lexicon(), edir0)
    tk = SeqLayers.typekey(:token, 1)
    n(e) = [Int(c.n) for c in first(SeqLayers.load_events(e)) if c.typekey == tk]
    @test n(edir) == n(edir0) .+ 2                              # two PP items per attachment record
  end

  @testset "train and decode" begin
    lex = Lexicon()
    m = train!(CountModel(), docs, specs, lex)
    @test !isempty(m.chosen) && !isempty(m.unchosen) && !isempty(m.prior)
    # a candidate whose features were never seen scores exactly its prior log-odds
    empty_m = SeqLayers.CountModel(Dict{UInt64,Float64}(), Dict{UInt64,Float64}(), m.prior)
    st = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
    key = (:char, 1, 1)
    po = SeqLayers.priorodds(m, SeqLayers.labelkey(:char, 1, SeqLayers.GAP_I))
    @test SeqLayers.score(empty_m, st, key, SeqLayers.GAP_I) ≈ log(po)
    r = evaluate(docs, specs, m, lex)
    @test 0 <= r.token_f1 <= 1
    out = process(doc_from_text("Rain fell. I don't like rain."), specs, m, lex)
    @test !isempty(out)
    @test join(join.(getfield.(out, :tokens)), "") == "Rainfell.Idon'tlikerain."   # every char ends up in a token
  end
end
