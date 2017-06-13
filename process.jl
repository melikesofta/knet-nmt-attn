import Base: start, next, done

const EOS = "</s>"
const UNK = "<unk>"

type Data
  tok2int::Dict{AbstractString, Int}
  int2tok::Vector{AbstractString}
  batchsize::Int
  seqlen::Int
  sentences::Vector{Vector{Int}}
  atype::DataType
end

function Data(file; batchsize=10, tok2int=nothing, int2tok=nothing, seqlen=70, atype=Array{Float32})
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
  Data(tok2int, int2tok, batchsize, seqlen, sentences, atype)
end

function sentenbatch(sentences::Vector{Vector{Int}}, from::Int, batchsize::Int, seqlen::Int, vocabsize::Int, atype::DataType)
  total = length(sentences)
  to = (from + batchsize -1 < total) ? (from + batchsize -1) : total

  if (to-from + 1 < batchsize)
    return (nothing, 1)
  end

  new_from = (to == total) ? 1 : (to + 1)
  batchsent = sentences[from:to]
  critic = findmax(map(length, batchsent))[1]

  data = Vector{Vector{Int32}}(critic+1)
  data[1] = fill!(zeros(Int32, batchsize), 1)
  masks = Vector{Vector{Int32}}(critic+1)
  masks[1] = fill!(zeros(Int32, batchsize), 1)

  for cursor=1:critic+1 # to pad EOW to the end of the word
        d = Vector{Int32}(batchsize)
        mask = ones(Int32, batchsize)
        @inbounds for i=1:batchsize
            sent = batchsent[i]
            if length(sent) < critic
                if length(sent) >= cursor
                    d[i] = sent[cursor]
                else
                    d[i] = 1
                    mask[i] = 0
                end
            else
                if cursor>critic
                    d[i] = 1
                else
                    d[i] = sent[cursor]
                end
            end
            data[cursor + 1] = d
            masks[cursor + 1] = mask
        end 
    end
    data = (data, masks)
    return data, new_from
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
    (item, new_from) = sentenbatch(sdict, 1, s.batchsize, s.seqlen, vocabsize, s.atype)
  else
    (item, new_from) = sentenbatch(sdict, from, s.batchsize, s.seqlen, vocabsize, s.atype)
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
