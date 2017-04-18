for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module CopySeq
using Knet,AutoGrad,ArgParse,Compat
#include(Pkg.dir("Knet/src/distributions.jl"))
include("process.jl");
include("model.jl");

function main(args=ARGS)
  (sentences, tok2int, int2tok) = readdata("cc10.en");
  for sentence in sentences
    println("\n", sentence);
    for word in sentence
      print(int2tok[word], " ")
    end
  end

  vocab = length(int2tok);
  hidden = 100;
  atype=Array{Float32};
  otype=Adam;
  model=initmodel(hidden, vocab, atype);
  opts=oparams(model,otype; lr=0.01);

  for epoch=1:50
    loss = 0; sentence_count=0;
    for sentence in sentences
      loss += s2s(model, sentence, sentence);
      sentence_count+=1;
      grads = s2sgrad(model, sentence, sentence)
      update!(model, grads, opts)
    end
    println(loss/sentence_count)
  end

  for sentence in sentences
    s2s_generate(model, sentence)
  end

end #main

function s2s_generate(model, inputs)
    state = initstate(inputs[1], model[:state0])
    for input in reverse(inputs)
      input = lstm_input(model[:embed1], input)
      input = reshape(input, 1, size(input, 1))
      state = lstm(model[:encode], state, input)
    end
    EOS = ones(Int, length(inputs[1]))
    input = lstm_input(model[:embed2], EOS)
    preds = []
    while (length(preds)<50)
      state = lstm(model[:decode], state, input)
      pred = state[1] * model[:output][1] .+ model[:output][2]
      #print(indmax(pred), " ")
      word = int2tok[indmax(pred)]
      word == "</s>" && break
      push!(preds, word)
      input = lstm_input(model[:embed2],indmax(pred))
      input = reshape(input, 1, size(input, 1))
    end
    for pred in preds
      print(pred, " ")
    end
    println()
end

oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module
