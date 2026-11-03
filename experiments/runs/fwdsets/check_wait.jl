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
mw = ChainModel(dir; select = :maxdev, leftovers = :greedy, minctx = 2, mincount = 3, redundant = true, ctxmode = :accum, candmemo = true)
mw.generalize = SeqLayers.generalization_slots(specs)
for (aw, n) in ((false, 1), (true, 1), (false, 4), (true, 4), (true, 16))
  SeqLayers.ATTACH_WAIT[] = aw; SeqLayers.RESCORE_EVERY[] = n
  r = evaluate_counts(docs, specs, mw, lex)
  println("wait=$aw n=$n: ", r, " ", eval_scores(r))
end
