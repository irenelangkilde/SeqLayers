# Record the same documents with relations=true into a directory (to compare two versions' events).
using SeqLayers
f = "/Users/irenelangkildegeary2015/Projects/data/unidep/unideps-English/UD_English-GUM/en_gum-ud-train.conllu"
docs = read_conllu(f)[1:3]
record_events(docs, default_specs(relations = true), Lexicon(), ARGS[1])
