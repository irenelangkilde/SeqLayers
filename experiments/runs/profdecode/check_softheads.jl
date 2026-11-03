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
mg = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 3, redundant = true, ctxmode = :accum)
mg.generalize = SeqLayers.generalization_slots(specs)
for sh in (false, true)
  SeqLayers.SOFTHEADS[] = sh
  st_ = SeqLayers.State(only(docs), 1:length(only(docs).text), true, 0, specs, lex)
  st_.infer = true
  solve!(st_, GoldModel(mg, Set([:gap, :tags, :heads])))
  tl = st_.layers[:token]
  ok = all(t -> tl[:head_dir, t] == only(docs).gold.tags[:head_dir][SeqLayers.tokspan(st_, t)], 1:length(tl))
  r = eval_scores(evaluate_counts(docs, specs, GoldModel(mg, Set([:gap, :tags, :heads])), lex))
  r0 = eval_scores(evaluate_counts(docs, specs, GoldModel(mg, Set([:gap])), lex))
  println("softheads=$sh: head_dir all gold $ok; uas gaps+tags+heads $(round(r.uas; digits=3)) vs gaps only $(round(r0.uas; digits=3))")
end
