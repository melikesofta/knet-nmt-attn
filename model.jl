function initmodel(H, BS, SV, TV, atype)
  init(d...)=atype(xavier(d...))
  model = Dict{Symbol,Any}()
  model[:state1] = init(BS,H)
  model[:state2] = init(BS,H)
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
  EOS = ones(Int, length(outputs[1][1]))
  input = gru_input(model[:embed2], EOS)
  preds = []
  sumlogp = 0.0;
  state = final_forw_state * model[:sinit] # batchsizexhidden

  # masks = inputs[2]
  # inputs = inputs[1]
  # for (forw_input, back_input, forw_mask, back_mask) in zip(inputs, reverse(inputs), masks, reverse(masks))
  masks = outputs[2]
  outputs = outputs[1]
  prev_mask=nothing
  for (output, mask) in zip(outputs, masks)
    state, context = s2s_decode(model, state, states, input; mask=prev_mask)
    pred = predict(model[:output], state, input, context)
    push!(preds, pred)
    input = gru_input(model[:embed2],output)
    prev_mask = mask
  end
  state, context = s2s_decode(model, state, states, input; mask=prev_mask)
  pred = predict(model[:output], state, input, context)
  push!(preds, pred)
  gold = vcat(outputs..., EOS)
  sumlogp = gru_output(gold, preds)
  return -sumlogp/length(inputs[1])
end

s2sgrad = grad(s2s)

function s2s_encode(inputs, model)
  state1 = initstate(inputs[1][1], model[:state1])
  state2 = initstate(inputs[1][1], model[:state2])
  states = []
  masks = inputs[2]
  inputs = inputs[1]
  for (forw_input, back_input, forw_mask, back_mask) in zip(inputs, reverse(inputs), masks, reverse(masks))
    forw_input = gru_input(model[:embed1], forw_input)
    state1 = gru(model[:encode1], state1, forw_input; mask=forw_mask)
    back_input = gru_input(model[:embed1], back_input)
    state2 = gru(model[:encode2], state2, back_input; mask=back_mask)
    state_cat = hcat(state1, state2) # batchsizex2hidden
    push!(states, state_cat)
  end
  return state1, states
end

function s2s_decode(model, state, states, input; mask=nothing)
  e = []; alpha = [];
  for j=1:length(states)
    ej = tanh(state * model[:attn][2] + states[j] * model[:attn][3]) * model[:attn][1]
    # ej: batchsizex1
    push!(e, ej) # TODO: What is more efficient? Pushing or concat?
  end
  sume = 0.0;
  for j=1:length(states)
    sume += exp(e[j]) # batchsizex1
  end
  for j=1:length(states)
    push!(alpha, (exp(e[j]) ./ sume)) # batchsizex1
  end
  c = alpha[1] .* states[1] # batchsizex2hidden
  for i=2:length(states)
    c = c .+ (alpha[i] .* states[i])
  end
  state = gru3(model[:decode], state, c, input; mask=mask)
  return state, c # batchsizexhidden; batchsizex2hidden
end

function gru(params, h, input; mask=nothing)
  z = sigm(input * params[1] + h * params[2]) # batchsizexhidden
  r = sigm(input * params[3] + h * params[4]) # batchsizexhidden
  h_candidate = tanh(input * params[5] + (r .* h) * params[6]) # batchsizexhidden
  h = (1-z) .* h + z .* h_candidate
  return (mask == nothing) ? h : (h .* mask) #batchsizexhidden
end

function gru3(params, h, c, input; mask=nothing)
  # h: batchsizexhidden; c: batchsizex2hidden; input: batchsizexhidden
  # model[:decode] = [ init(H, H), init(H, H), init(2H, H), init(H, H),
  #       init(H, H), init(2H, H), init(H, H), init(H, H), init(2H, H)]
  z = sigm(input * params[1] + h * params[2] + c * params[3]) # batchsizexhidden
  r = sigm(input * params[4] + h * params[5] +  c * params[6]) # batchsizexhidden
  s_candidate = tanh(input * params[7] + (r .* h) * params[8] + c * params[9]) # batchsizexhidden
  h = (1-z) .* h + z .* s_candidate # batchsizexhidden
  return (mask == nothing) ? h : (h .* mask)
end

function gru_output(gold, preds)
  pred = vcat(preds...)
  sumlogp = logprob(gold, pred)
  return sumlogp
end

function predict(param, state, input, context)
  # model[:output] = [ init(H,TV), init(H,TV), init(2H,TV) ]
  return state * param[1] + input * param[2] + context * param[3] # batchsizexTV
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

function gru_input(param, inputs)
  # concatenate corresponding embedding vectors for each
  # word in the batch vertically
  # resulting matrix is of size batchsizexhidden and holds
  # the embedding of one word at each row
  p = reshape(param[inputs[1], :], 1, size(param, 2))
  for i=2:length(inputs)
      row = reshape(param[inputs[i], :], 1, size(param, 2))
      p = vcat(p, row)
  end
  return p
end

function gru_input_back(param, inputs, grads)
  # carry corresponding gradients for each word
  # to rows in dparam array of size SVxH
  dparam = zeros(param)
  for i=1:length(inputs)
    dparam[inputs[i], :] = grads[i, :]
  end
  return dparam
end

@primitive gru_input(param,input),grads gru_input_back(param,input,grads)

function initstate(idx, state0)
  h = state0
  h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), size(h,2)), 0)
  return h
end
