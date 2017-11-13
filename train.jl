# Usage: julia train.jl --sourcefiles path/to/sourcefile --targetfiles path/to/targetfile

for p in ("Knet","AutoGrad","ArgParse","Compat", "JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module Attention
using Knet,AutoGrad,ArgParse,Compat,JLD
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
    ("--generatedfile"; default="foo.txt";help="Path for the file for the translation to be written")
    ("--loadfile"; help="Initialize model from file")
    ("--savefile"; help="Save final model to file")
    ("--bestfile"; help="Save best model to file")
    ("--hidden"; arg_type=Int; default=1000; help="Sizes of one or more LSTM layers.")
    ("--embedding"; arg_type=Int; default=620; help="Sizes of one or more LSTM layers.")
		("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
		("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
		("--lr"; arg_type=Float64; default=0.01; help="Initial learning rate.")
    ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
		("--seed"; arg_type=Int; default=42; help="Random number seed.")
		("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	end
	println(s.description)
	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true)
	println("opts=",[(k,v) for (k,v) in o]...)
	o[:seed] > 0 && srand(o[:seed])
  o[:atype] = eval(parse(o[:atype]))
  
  # if model is loaded from a file, dictionaries also need to be saved/loaded (embeddings are also learned)
  source_tok2int = nothing; source_int2tok = nothing;
  target_tok2int = nothing; target_int2tok = nothing;

  if o[:loadfile]!=nothing
    loadfile = o[:loadfile]
    info("Loading model from $loadfile")
    source_tok2int = load(loadfile, "source_tok2int")
    source_int2tok = load(loadfile, "source_int2tok")
    target_tok2int = load(loadfile, "target_tok2int")
    target_int2tok = load(loadfile, "target_int2tok")
    model = load(loadfile, "model")
  end

  if length(o[:sourcefiles]) > 0
    source_data = Data(o[:sourcefiles][1]; batchsize=o[:batchsize], tok2int=source_tok2int, int2tok=source_int2tok)
    target_data = Data(o[:targetfiles][1]; batchsize=o[:batchsize], tok2int=target_tok2int, int2tok=target_int2tok)
    source_tok2int = source_data.tok2int
    source_int2tok = source_data.int2tok
    target_tok2int = target_data.tok2int
    target_int2tok = target_data.int2tok
  end

  #only works for one set of test files for now
  if (length(o[:sourcefiles]) > 1 && length(o[:targetfiles]) > 1)
    source_test_data = Data(o[:sourcefiles][2]; batchsize=o[:batchsize], tok2int=source_tok2int, int2tok=source_int2tok)
    target_test_data = Data(o[:targetfiles][2]; batchsize=o[:batchsize], tok2int=target_tok2int, int2tok=target_int2tok)
  end

  source_vocab = length(source_int2tok);
  target_vocab = length(target_int2tok);

  if o[:loadfile] == nothing
    model=initmodel(o[:hidden], o[:batchsize], o[:embedding], source_vocab, target_vocab, o[:atype]);
  end

  init_trn_loss = s2s_test(model, source_data, target_data, o);
  if (length(o[:sourcefiles]) > 1 && length(o[:targetfiles]) > 1)
    init_tst_loss = s2s_test(model, source_test_data, target_test_data, o);
    devbest = init_tst_loss
    println("epoch\t0\ttrn_loss\t", init_trn_loss, "\ttst_loss\t", init_tst_loss)
  else
    println("epoch\t0\ttrn_loss\t", init_trn_loss)
  end

  opts=oparams(model,Adagrad;);
  for epoch=1:o[:epochs]
    trn_loss = s2s_train(model, source_data, target_data, opts, o)
    if (length(o[:sourcefiles]) > 1 && length(o[:targetfiles]) > 1)
      tst_loss = s2s_test(model, source_test_data, target_test_data, o)
      println(:epoch, '\t', epoch, '\t', :trn_loss, '\t', trn_loss, '\t', :tst_loss, '\t', tst_loss)
      if tst_loss < devbest
        devbest = tst_loss
        bestfile = o[:bestfile]
        if bestfile != nothing
            info("Saving best model to $bestfile")
            save(savefile, "model", model, "source_tok2int", source_tok2int, "source_int2tok",
            source_int2tok, "target_tok2int", target_tok2int, "target_int2tok", target_int2tok)
        end
      end
    else
      println(:epoch, '\t', epoch, '\t', :trn_loss, '\t', trn_loss)
    end
  end

  if (o[:generate] != nothing)
    generate_data = Data(o[:generate]; batchsize=1, tok2int=source_tok2int, int2tok=source_int2tok)
    first_sentence = true
    file = open(o[:generatedfile], "a") # append mode
    for sentence in generate_data
      if !first_sentence
        write(file, "\n")
      end
      s2s_generate(model, sentence, target_int2tok, o[:hidden], o[:atype], file)
      first_sentence = false
    end
    close(file)
  end

  # save functionality
  if o[:savefile]!=nothing
    savefile = o[:savefile]
    info("Saving final model to $savefile")
    save(savefile, "model", model, "source_tok2int", source_tok2int, "source_int2tok",
     source_int2tok, "target_tok2int", target_tok2int, "target_int2tok", target_int2tok)
  end
end #main

function s2s_train(model, source_data, target_data, opts, o)
  loss = 0; sentence_count=0;
  for (source_sentence, target_sentence) in zip(source_data, target_data)
    (source_sentence == nothing) && break;
    loss += s2s(model, source_sentence, target_sentence, o[:atype]; hidden=o[:hidden]);
    sentence_count+=1;
    if o[:gcheck] > 0 && sentence_count == 1 #check gradients only for the first batch
				gradcheck(s2s, model, source_sentence, target_sentence, o[:atype]; gcheck=o[:gcheck], hidden=o[:hidden])
    end
    grads = s2sgrad(model, source_sentence, target_sentence, o[:atype]; hidden=o[:hidden])
    update!(model, grads, opts)
  end
  return loss/sentence_count
end

function s2s_test(model, source_data, target_data, o)
  loss = 0; sentence_count=0;
  for (source_sentence, target_sentence) in zip(source_data, target_data)
    (source_sentence == nothing) && break;
    loss += s2s(model, source_sentence, target_sentence, o[:atype]; hidden=o[:hidden]);
    sentence_count+=1;
  end
  return loss/sentence_count
end

oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module
