import Base: start, next, done

const EOS = "</s>"
const UNK = "<unk>"

type Data
    tok2int::Dict{AbstractString, Int}
    int2tok::Vector{AbstractString}
    batchsize::Int
    sentences::Vector{Vector{Int}}
end

function Data(file::AbstractString; batchsize=10, tok2int=nothing, int2tok=nothing)
    dict_ready = (tok2int != nothing)
    if !dict_ready
        tok2int = Dict{AbstractString, Int}(EOS=>1, UNK=>2)
        int2tok = Vector{AbstractString}()
        push!(int2tok, "</s>")
    end
    tok2int["</s>"] = 1
    sentences = Vector{Vector{Int}}()
    f = open(file)
    for line in eachline(f)
        words = Vector{Int}()
        for word in split(line)
            if !dict_ready
                if !haskey(tok2int, word)
                    push!(int2tok, word)
                    tok2int[word] = length(int2tok)
                end
            end
            ind = get(tok2int, word, tok2int[UNK])
            push!(words, ind)
        end
        push!(words, tok2int[EOS])
        push!(sentences, words)
    end
    close(f)
    Data(tok2int, int2tok, batchsize, sentences)
end

function sentenbatch(sentences::Vector{Vector{Int}}, batchsize::Int)
    critic = findmax(map(length, sentences))[1] # length of the longest sentence in the batch
    data = Vector{Vector{Int32}}(critic)
    data[1] = fill!(zeros(Int32, batchsize), 1)
    masks = Vector{Vector{Int32}}(critic)
    masks[1] = fill!(zeros(Int32, batchsize), 1)

    for cursor=1:critic
        word_batch = Vector{Int32}(batchsize)
        mask = ones(Int32, batchsize)
        for i=1:batchsize
            sentence = sentences[i]
            if length(sentence) >= cursor
                word_batch[i] = sentence[cursor]
            else
                word_batch[i] = 1
                mask[i] = 0
            end
            data[cursor] = word_batch
            masks[cursor] = mask
        end
    end
    return (data, masks)
end

function start(s::Data)
    return nothing
end

function next(s::Data, state)
    from = state
    from = (from == nothing) ? 1 : from

    total = length(s.sentences)
    to = (from + s.batchsize - 1 < total) ? (from + s.batchsize - 1) : total
    if to == total
        (item, new_from) = (nothing, 1)
    else
        new_from = to + 1
    end
    
    batch = s.sentences[from:to]
    item = sentenbatch(batch, s.batchsize)
    state = new_from
    return (item, state)
end

function done(s::Data, state)
    return state == 1
end