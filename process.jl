import Base: start, next, done

const EOS = "</s>"
const UNK = "<unk>"

type Data
  tok2int::Dict{AbstractString, Int}
  int2tok::Vector{AbstractString}
  batchsize::Int
  seqlen::Int
  sentences::Vector{Vector{Int}}
end

function Data(file; batchsize=10, tok2int=nothing, int2tok=nothing, seqlen=70)
  vocab_exists = (tok2int != nothing)
  if !vocab_exists
    #tok2int["</s>"]=1 # We use "</s>"=>1 as the EOS token
    tok2int = Dict{AbstractString, Int}(EOS=>1, UNK=>2)
    vocab = vocab_from_file(file)
  end
  f = open(file)
  sentences = Vector{Vector{Int}}()
  for line in eachline(f)
    words = Vector{Int}()
    for word in split(line)
      if !vocab_exists && !(word in vocab)
        word = UNK
      end
      if vocab_exists
        ind = get(tok2int, word, tok2int[UNK])
      else
        ind = get!(tok2int, word, 1+length(tok2int))
      end
      push!(words, ind)
    end
    push!(words, tok2int[EOS])
    skey = length(words)
    push!(sentences, words)
  end
  close(f)
  vocabsize = length(tok2int)
  # create int2tok vector from tok2int dict
  int2tok = Vector{AbstractString}()
  for (tok, int) in tok2int
    push!(int2tok, tok) #int2tok[int] = tok
  end
  Data(tok2int, int2tok, batchsize, seqlen, sentences)
end

function sentenbatch(sentences::Vector{Vector{Int}}, from::Int, batchsize::Int, seqlen::Int, vocabsize::Int)
  total = length(sentences)
  to = (from + batchsize -1 < total) ? (from + batchsize -1) : total

  # not to work with surplus sentences ?
  if (to-from + 1 < batchsize)
    return (nothing, 1)
  end

  new_from = (to == total) ? 1 : (to + 1)
  batchsent = sentences[from:to]

  batch = ones(batchsize, seqlen+1)
  mask = ones(batchsize, seqlen+1) 
  batchvector = Vector{Vector{Int}}()
  maskvector = Vector{Vector{Int}}()
  sind = 1
  for s in batchsent
    batch[sind, 1:length(s)] = s
    mask[sind, 1:length(s)] = zeros(1, length(s))
    sind = sind + 1
    if (sind > batchsize)
      for i=1:size(batch, 2)
        push!(batchvector, batch[:,i])
        push!(maskvector, mask[:,i])
      end
    end
  end
  data = (batchvector, maskvector)
  return (data, new_from)
end  

function start(s::Data)
  sdict = deepcopy(s.sentences)
  @assert (!isempty(sdict)) "There is not enough data with that batchsize $(s.batchsize)"
  from = nothing
  vocabsize = length(s.tok2int)
  state = (sdict, from, vocabsize)
  return state
end

function next(s::Data, state)
  (sdict, from, vocabsize) = state
  if from==nothing
    (item, new_from) = sentenbatch(sdict, 1, s.batchsize, s.seqlen, vocabsize)
  else
    (item, new_from) = sentenbatch(sdict, from, s.batchsize, s.seqlen, vocabsize)
  end
  from = new_from
  state = (sdict, from, vocabsize)
  return (item, state)
end

function done(s::Data, state)
  (sdict, from, vocabsize) = state
  return from == 1
end

function vocab_from_file(file)
  vocab = Set{AbstractString}()
  open(file) do f
    for line in eachline(f)
      line = split(line)
      if !isempty(line)
        for word in line
          push!(vocab, word)
        end
      end
    end
  end
  return vocab
end

# function readdata(file; batchsize=10 , tok2int=nothing, int2tok=nothing)
#   global strings = readlines(file)
#   if (tok2int==nothing && int2tok==nothing)
#     tok2int = Dict{AbstractString, Int}()
#   end
#   sentences = Vector{Vector{Vector{Int}}}()
#   maxlength = 60
#   batch = ones(batchsize, maxlength)
#   batchvector = Vector{Vector{Int}}()
#   sind=1
#   for line in strings
#     s = Vector{Int}()
#     for word in split(line)
#       if !haskey(tok2int, word)
#         push!(int2tok, word)
#         tok2int[word] = length(int2tok)
#       end
#       push!(s, tok2int[word])
#     end
#     batch[sind, 1:length(s)] = s
#     sind = sind + 1
#     if (sind>batchsize)
#       for i=1:size(batch,2);
#         push!(batchvector, batch[:,i])
#       end
#       push!(sentences, batchvector)
#       batch = ones(batchsize, maxlength)
#       batchvector = Vector{Vector{Int}}()
#       sind=1
#     end
#   end
#   return sentences, tok2int, int2tok
# end
