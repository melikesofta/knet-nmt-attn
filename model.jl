function initmodel(H, SV, TV, atype)
  init(d...)=atype(xavier(d...))
  model = Dict{Symbol,Any}()
  model[:state1] = init(1,H)
  model[:state2] = init(1,H)
  model[:embed1] = init(SV,H)
  model[:encode1] = [ init(H,H), init(H,H), init(H,H), init(H,H), init(H,H), init(H,H) ]
  model[:encode2] = [ init(H,H), init(H,H), init(H,H), init(H,H), init(H,H), init(H,H) ]
  model[:embed2] = init(TV,H)
  model[:sinit] = init(H,H)
  model[:attn] = [ init(H,1), init(H,H), init(2H,H) ]
  model[:decode] = [ init(H, H), init(H, H), init(2H, H), init(H, H),
  init(H, H), init(2H, H), init(H, H), init(H, H), init(2H, H)]
  model[:output] = [ init(H,TV), init(H,TV), init(2H,TV) ]
  return model
end

function s2s(model, inputs, outputs)
  (final_forw_state, states) = s2s_encode(inputs, model)
  EOS = 1 #ones(Int, 1, length(outputs[1]))
  input = gru_input(model[:embed2], EOS)
  preds = []
  sumlogp = 0.0;

  state = final_forw_state * model[:sinit]
  for output in outputs
    state, context = s2s_decode(model, state, states, input)
    pred = predict(model[:output], state, input, context)
    push!(preds, pred)
    input = gru_input(model[:embed2],output)
  end
  state, context = s2s_decode(model, state, states, input)
  pred = predict(model[:output], state, input, context)
  push!(preds, pred)
  gold = vcat(outputs..., EOS)
  sumlogp = gru_output(gold, preds)
  return -sumlogp
end

s2sgrad = grad(s2s)

function s2s_encode(inputs, model)
  state1 = initstate(inputs[1], model[:state1])
  state2 = initstate(inputs[1], model[:state2])
  states = []
  for (forw_input, back_input) in zip(inputs, reverse(inputs))
    forw_input = gru_input(model[:embed1], forw_input)
    state1 = gru(model[:encode1], state1, forw_input)
    back_input = gru_input(model[:embed1], back_input)
    state2 = gru(model[:encode2], state2, back_input)
    state_cat = hcat(state1, state2)
    push!(states, state_cat)
  end
  return state1, states
end

function s2s_decode(model, state, states, input)
  e = []; alpha = [];
  for j=1:length(states)
    ej = tanh(state * model[:attn][2] + states[j] * model[:attn][3]) * model[:attn][1]
    push!(e, ej)
  end
  sume = 0.0;
  for j=1:length(states)
    sume += sum(exp(e[j]))
  end
  for j=1:length(states)
    push!(alpha, (exp(e[j]) ./ sume))
  end
  c = alpha[1] .* states[1]
  for i=2:length(states)
    c = c .+ (alpha[i] .* states[i])
  end
  state = gru3(model[:decode], state, c, input)
  return state, c
end

function gru(params, h, input)
  z = sigm(input * params[1] + h * params[2])
  r = sigm(input * params[3] + h * params[4])
  h_candidate = tanh(input * params[5] + (r .* h) * params[6])
  h = (1-z) .* h + z .* h_candidate
  return h
end

function gru3(params, h, c, input)
  z = sigm(input * params[1] + h * params[2] + c * params[3])
  r = sigm(input * params[4] + h * params[5] +  c * params[6])
  s_candidate = tanh(input * params[7] + (r .* h) * params[8] + c * params[9])
  h = (1-z) .* h + z .* s_candidate
  return h
end

function gru_output(gold, preds)
  pred = vcat(preds...)
  sumlogp = logprob(gold, pred)
  return sumlogp
end

function predict(param, state, input, context)
  return state * param[1] + input * param[2] + context * param[3]
end

function logprob(output, ypred)
  nrows,ncols = size(ypred)
  index = similar(output)
  @inbounds for i=1:length(output)
    index[i] = i + (output[i]-1)*nrows
  end
  o1 = logp(ypred,2)
  o2 = o1[index]
  o3 = sum(o2)
  return o3
end

function gru_input(param, input)
  p = param[input,:]
  return reshape(p, 1, size(p, 1))
end

function gru_input_back(param, input, grads)
  dparam = zeros(param)
  dparam[input,:]=grads
  return dparam
end

@primitive gru_input(param,input),grads gru_input_back(param,input,grads)

function initstate(idx, state0)
  h = state0
  h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), length(h)), 0)
  return h
end
