for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module SimpleEncDec
using Knet,AutoGrad,ArgParse,Compat
#include(Pkg.dir("Knet/src/distributions.jl"))
include("process.jl");
include("model.jl");

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--sourcefiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
    ("--targetfiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
		("--generate"; help="Generates a translation of the provided file")
		("--hidden"; arg_type=Int; default=100; help="Sizes of one or more LSTM layers.")
		("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
		("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
		("--lr"; arg_type=Float64; default=0.01; help="Initial learning rate.")
		("--seed"; arg_type=Int; default=42; help="Random number seed.")
		("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	end
	println(s.description)
	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true)
	println("opts=",[(k,v) for (k,v) in o]...)
	o[:seed] > 0 && srand(o[:seed])
	o[:atype] = eval(parse(o[:atype]))

  (source_data, source_tok2int, source_int2tok) = readdata(o[:sourcefiles][1])
  (target_data, target_tok2int, target_int2tok) = readdata(o[:targetfiles][1])

  source_vocab = length(source_int2tok);
  target_vocab = length(target_int2tok);
  model=initmodel(o[:hidden], source_vocab, target_vocab, o[:atype]);

  opts=oparams(model,Adam; lr=o[:lr]);
  for epoch=1:o[:epochs]
    trn_loss = s2s_train(model, source_data, target_data, opts, o)
    println(:epoch, '\t', epoch, '\t', :trn_loss, '\t', trn_loss)
  end

  for sentence in source_data
    s2s_generate(model, sentence, target_int2tok)
  end

end #main

function s2s_train(model, source_data, target_data, opts, o)
  loss = 0; sentence_count=0;
  for (source_sentence, target_sentence) in zip(source_data, target_data)
    loss += s2s(model, source_sentence, target_sentence);
    sentence_count+=1;
    grads = s2sgrad(model, source_sentence, target_sentence)
    update!(model, grads, opts)
  end
  return loss/sentence_count
end

function s2s_generate(model, inputs, target_int2tok)
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
      word = target_int2tok[indmax(pred)]
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
