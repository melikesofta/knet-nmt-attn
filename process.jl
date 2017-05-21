function readdata(file; batchsize=10 , tok2int=nothing, int2tok=nothing)
  global strings = readlines(file)
  if (tok2int==nothing && int2tok==nothing)
    tok2int = Dict{AbstractString, Int}()
    int2tok = Vector{AbstractString}()
    push!(int2tok, "</s>");
  end
  #tok2int["</s>"]=1 # We use "</s>"=>1 as the EOS token
  sentences = Vector{Vector{Vector{Int}}}()
  maxlength = 60
  batch = ones(batchsize, maxlength)
  batchvector = Vector{Vector{Int}}()
  sind=1
  for line in strings
    s = Vector{Int}()
    for word in split(line)
      if !haskey(tok2int, word)
        push!(int2tok, word)
        tok2int[word] = length(int2tok)
      end
      push!(s, tok2int[word])
    end
    batch[sind, 1:length(s)] = s
    sind = sind + 1
    if (sind>batchsize)
      for i=1:size(batch,2);
        push!(batchvector, batch[:,i])
      end
      push!(sentences, batchvector)
      batch = ones(batchsize, maxlength)
      batchvector = Vector{Vector{Int}}()
      sind=1
    end
  end
  return sentences, tok2int, int2tok
end
