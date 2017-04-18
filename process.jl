function readdata(file)
    global strings = readlines(file)
    global tok2int = Dict{AbstractString, Int}()
    global int2tok = Vector{AbstractString}()
    push!(int2tok, "</s>");
    #tok2int["</s>"]=1 # We use "</s>"=>1 as the EOS token
    sentences = Vector{Vector{Int}}()
    for line in strings
        s = Vector{Int}()
        for word in split(line)
            if !haskey(tok2int, word)
                push!(int2tok, word)
                tok2int[word] = length(int2tok)
            end
            push!(s, tok2int[word])
		    end
		#push!(s, tok2int["</s>"])
		push!(sentences, s)
    end
    return sentences, tok2int, int2tok
end

function randseq(V,B,T)
    s = Vector{Vector{Int}}()
    for t in 1:T
        push!(s, rand(2:V,B))
    end
    return s
end
