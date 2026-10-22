# SeqLayers.jl
#
# A stack of sequence layers (chars -> tokens -> sentences), each a grid of
# rows (features, relations, decisions) x columns (positions).  One agenda holds
# pending decisions from every layer; the most confident one is committed first
# (easy-first greedy search).  Constraints turn committed decisions into new
# columns in the layer above, so tokenization and parsing are interleaved.
#
# Everything task-specific lives in a LayerSpec (see specs.jl); the engine,
# models and driver are generic.

module SeqLayers

using DataStructures: PriorityQueue
import FunctionWrappers: FunctionWrapper

include("types.jl")
include("engine.jl")
include("models.jl")
include("suffixarray.jl")
include("wordclass.jl")
include("specs.jl")
include("conllu.jl")
include("driver.jl")
include("lattice.jl")
include("chain.jl")
include("climb.jl")

export NONE, PENDING,
  Layer, CharLayer, TokenLayer, SentenceLayer,
  FeatDef, RelDef, FeatPath, DecisionDef, Constraint, LayerSpec,
  Sequence, CharSeq, TokenSeq, SentenceSeq, Lexicon, Gold, Doc, State,
  addcol!, follow, context, label_features, link_features, solve!,
  AbstractModel, CountModel, OracleModel, decide, score,
  default_specs, read_conllu, doc_from_text, CharIndex,
  ParsedSentence, process, train!, evaluate, evaluate_counts, eval_scores,
  LatticeModel, record_events, ChainModel, climb_table, load_wordclasses

end
