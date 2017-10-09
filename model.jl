function initmodel(H, BS, E, SV, TV, atype)
  init(d...)=atype(xavier(d...))
  bias(d...)=atype(zeros(d...))
  model = Dict{Symbol,Any}()
  model[:forw_state] = init(BS,H)
  model[:back_state] = init(BS,H)

  model[:enc_embed] = init(SV,E)

  model[:forw_encode] = [ init(E,H), init(H,H), init(E,H), init(H,H), init(E,H), init(H,H) ]
  model[:forw_encode_bias] = [ bias(1,H), bias(1,H), bias(1,H) ]

  model[:back_encode] = [ init(E,H), init(H,H), init(E,H), init(H,H), init(E,H), init(H,H) ]
  model[:back_encode_bias] = [ bias(1,H), bias(1,H), bias(1,H) ]

  model[:dec_embed] = init(TV,E)

  model[:sinit] = init(H,H)
  model[:sinit_bias] = bias(1,H)

  model[:attn] = [ init(H,1), init(H,H), init(2H,H) ]
  model[:attn_bias] = [ bias(1,1), bias(1,H) ]

  model[:decode] = [ init(E, H), init(H, H), init(2H, H), init(E, H),
  init(H, H), init(2H, H), init(E, H), init(H, H), init(2H, H)]
  model[:decode_bias] = [ bias(1, H), bias(1, H), bias(1, H) ]

  model[:output] = [ init(H,TV), init(E,TV), init(2H,TV) ]
  model[:output_bias] = [ bias(1,TV) ]
  return model
end

function s2s(model, inputs, outputs, atype)
  batchsize = size(inputs[1][1], 1)
  (final_back_state, states) = s2s_encode(model, inputs, atype)

  EOS = ones(Int, batchsize)
  input = embed(model[:dec_embed], EOS)

  state = tanh(final_back_state * model[:sinit] .+ model[:sinit_bias]) # batchsizexhidden
  prev_mask = nothing

  enc_effect = map(hj -> hj * model[:attn][3], states)
  preds=[];
  (outputs, masks) = outputs
  for (output, mask) in zip(outputs, masks)
      prev_mask = prev_mask == nothing ? prev_mask : atype(prev_mask)
      state, context = s2s_decode(model, state, states, input, enc_effect; mask=prev_mask)
      pred = predict(model[:output], model[:output_bias], state, input, context; mask=prev_mask)
      push!(preds, pred)
      input = embed(model[:dec_embed], output)
      prev_mask = mask
  end
  prev_mask = prev_mask == nothing ? prev_mask : atype(prev_mask)
  state, context = s2s_decode(model, state, states, input, enc_effect; mask=prev_mask)
  pred = predict(model[:output], model[:output_bias], state, input, context; mask=prev_mask)
  push!(preds, pred)
  gold = vcat(outputs..., EOS)
  sumlogp = gru_output(gold, preds)
  return -sumlogp/batchsize
end

s2sgrad = grad(s2s)

function embed(param, inputs)
  emb = reshape(param[inputs[1], :], 1, size(param, 2))
  for i=2:length(inputs)
      embi = reshape(param[inputs[i], :], 1, size(param, 2))
      emb = vcat(emb, embi)
  end
  return emb
end

function gru(weights, bias, h, input; mask=nothing)
  zi = sigm(input * weights[1] + h * weights[2] .+ bias[1])
  r = sigm(input * weights[3] + h * weights[4] .+ bias[2])
  h_candidate = tanh(input * weights[5] + (r .* h) * weights[6] .+ bias[3])
  h = (1 - zi) .* h + zi .* h_candidate
  return (mask == nothing) ? h : (h .* mask) # batchsizexhidden
end

function gru3(weights, bias, h, c, input; mask=nothing)
  zi = sigm(input * weights[1] + h * weights[2] + c * weights[3] .+ bias[1])
  r = sigm(input * weights[4] + h * weights[5] + c * weights[6] .+ bias[2])
  s_candidate = tanh(input * weights[7] + (r .* h) * weights[8] + c * weights[9] .+ bias[3])
  h = (1-zi) .* h + zi .* s_candidate #batchsizexhidden
  return (mask==nothing) ? h : (h .* mask)
end

function s2s_encode(model, inputs, atype)
  forw_state = initstate(inputs[1][1], model[:forw_state])
  back_state = initstate(inputs[1][1], model[:back_state])
  states = []
  (sentence, mask) = inputs
  for (forw_word, back_word, forw_mask, back_mask) in zip(sentence, reverse(sentence), mask, reverse(mask))
      forw_word = embed(model[:enc_embed], forw_word)
      back_word = embed(model[:enc_embed], back_word)
      forw_state = gru(model[:forw_encode], model[:forw_encode_bias], forw_state, forw_word; mask=atype(forw_mask))
      back_state = gru(model[:back_encode], model[:back_encode_bias], back_state, back_word; mask=atype(back_mask))
      push!(states, hcat(forw_state, back_state))
  end
  return back_state, states
end

function s2s_decode(model, state, states, input, enc_effect; mask=nothing)
  e = map(enc_e -> tanh(state * model[:attn][2] .+ model[:attn_bias][2] + enc_e) * model[:attn][1] .+ model[:attn_bias][1], enc_effect)
  expe = map(ej -> exp(ej), e)
  sume = reduce(+, 0, expe)
  alpha = map(expej -> expej ./ sume, expe)
  alpst = map((a, s) -> a .* s, alpha, states)
  c = reduce(+, 0, alpst)
  state = gru3(model[:decode], model[:decode_bias], state, c, input; mask=mask)
  return state, c # batchsizexhidden; batchsizex2hidden
end

function predict(weights, bias, state, input, context; mask=nothing)
  pred = state * weights[1] + input * weights[2] + context * weights[3] .+ bias[1]
  if mask != nothing
      masked_pred = reshape(pred[1, :] .* mask[1], 1, size(pred[1,:], 1))
      for i=2:length(mask)
          new_pred = reshape(pred[i, :] .* mask[i], 1, size(pred[i,:], 1))
          masked_pred = vcat(masked_pred, new_pred)
      end
  end
  return mask == nothing ? pred : masked_pred
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

function gru_output(gold, preds)
  pred = vcat(preds...)
  sumlogp = logprob(gold, pred)
  return sumlogp
end

function initstate(idx, state0)
  h = state0
end