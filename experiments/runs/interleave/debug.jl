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
specs = default_specs()
lex = Lexicon()
dir = record_events(docs, specs, lex, mktempdir())
for mc in (1, 3), (il, ss) in ((false, :posneg), (true, :posneg), (true, :abs))
  SeqLayers.DC_STATS[] = Dict{Symbol,Float64}()
  mp = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = mc, redundant = true, ctxmode = :accum,
    interleave = il, stepscore = ss, relevance = 0.0)
  mp.generalize = SeqLayers.generalization_slots(specs)
  r = eval_scores(evaluate_counts(docs, specs, mp, lex))
  d = SeqLayers.DC_STATS[]
  println("mincount=$mc interleave=$il $ss uas ", round(r.uas; digits = 3), " chain steps ", get(d, :il_chain, 0), " leftover wins ", get(d, :il_left, 0), " dc_n ", get(d, :dc_n, 0))
end
