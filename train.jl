# Usage: julia train.jl --sourcefiles path/to/sourcefile --targetfiles path/to/targetfile

for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module Attention
using Knet,AutoGrad,ArgParse,Compat
#include(Pkg.dir("Knet/src/distributions.jl"))
include("process.jl");
include("model.jl");

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to translate between languages"
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
  source_data = Data(o[:sourcefiles][1]; batchsize=o[:batchsize], atype=o[:atype])
  source_tok2int = source_data.tok2int
  source_int2tok = source_data.int2tok
  target_data = Data(o[:targetfiles][1]; batchsize=o[:batchsize], atype=o[:atype])
  target_tok2int = target_data.tok2int
  target_int2tok = target_data.int2tok

  if (length(o[:sourcefiles]) > 1 && length(o[:targetfiles]) > 1)
    (source_test_data,) = Data(o[:sourcefiles][2]; batchsize=o[:batchsize], tok2int=source_tok2int, int2tok=source_int2tok, atype=o[:atype])
    (target_test_data,) = Data(o[:targetfiles][2]; batchsize=o[:batchsize], tok2int=target_tok2int, int2tok=target_int2tok, atype=o[:atype])
  end

  source_vocab = length(source_int2tok);
  target_vocab = length(target_int2tok);
  model=initmodel(o[:hidden], o[:batchsize], source_vocab, target_vocab, o[:atype]);

  opts=oparams(model,Adam; lr=o[:lr]);
  for epoch=1:o[:epochs]
    trn_loss = s2s_train(model, source_data, target_data, opts, o)
    if (length(o[:sourcefiles]) > 1 && length(o[:targetfiles]) > 1)
      tst_loss = s2s_test(model, source_test_data, target_test_data)
      println(:epoch, '\t', epoch, '\t', :trn_loss, '\t', trn_loss, '\t', :tst_loss, '\t', tst_loss)
    else
      println(:epoch, '\t', epoch, '\t', :trn_loss, '\t', trn_loss)
    end
  end

  if (o[:generate] != nothing)
    generate_data = Data(o[:generate]; batchsize=1, tok2int=source_tok2int, int2tok=source_int2tok, atype=o[:atype])
    for sentence in generate_data
      s2s_generate(model, sentence, target_int2tok, o[:hidden], o[:atype])
    end
  end
end #main

function s2s_train(model, source_data, target_data, opts, o)
  loss = 0; sentence_count=0;
  for (source_sentence, target_sentence) in zip(source_data, target_data)
    (source_sentence == nothing) && break;
    loss += s2s(model, source_sentence, target_sentence);
    sentence_count+=1;
    grads = s2sgrad(model, source_sentence, target_sentence)
    update!(model, grads, opts)
  end
  return loss/sentence_count
end

function s2s_test(model, source_data, target_data)
  loss = 0; sentence_count=0;
  for (source_sentence, target_sentence) in zip(source_data, target_data)
    loss += s2s(model, source_sentence, target_sentence);
    sentence_count+=1;
  end
  return loss/sentence_count
end

function s2s_generate(model, inputs, target_int2tok, hidden, atype)
  init(d...)=atype(xavier(d...))
  model[:state1] = init(1,hidden)
  model[:state2] = init(1,hidden)

  (final_forw_state, states) = s2s_encode(inputs, model)
  EOS = ones(Int, length(inputs[1][1]))
  input = gru_input(model[:embed2], EOS)
  preds = []
  state = final_forw_state * model[:sinit]
  while (length(preds)<50)
    state, context = s2s_decode(model, state, states, input)
    pred = predict(model[:output], state, input, context)
    ind = indmax(convert(Array{Float32}, pred))
    word = target_int2tok[ind]
    word == "</s>" && break
    push!(preds, word)
    input = gru_input(model[:embed2],ind)
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
