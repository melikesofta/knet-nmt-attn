function initmodel(H, SV, TV, atype)
    init(d...)=atype(xavier(d...))
    model = Dict{Symbol,Any}()
    model[:state1] = [ init(1,H), init(1,H) ]
    model[:state2] = [ init(1,H), init(1,H) ]
    model[:embed1] = init(SV,H)
    model[:merge] = [ init(H,H), init(H,H), init(1,H), init(1,H) ]
    model[:encode1] = [ init(2H,4H), init(1,4H) ]
    model[:encode2] = [ init(2H,4H), init(1,4H) ]
    model[:embed2] = init(TV,H)
    model[:decode] = [ init(2H,4H), init(1,4H) ]
    model[:output] = [ init(H,TV), init(1,TV) ]
    return model
end

function s2s(model, inputs, outputs)
    state = s2s_encode(inputs, model)
    EOS = ones(Int, length(outputs[1]))
    input = lstm_input(model[:embed2], EOS)
    preds = []
    sumlogp = 0

    for output in outputs
        state = lstm(model[:decode], state, input)
        push!(preds, state[1])
        input = lstm_input(model[:embed2],output)
        input = reshape(input, 1, size(input, 1))
    end
    state = lstm(model[:decode], state, input)
    push!(preds, state[1])
    gold = vcat(outputs..., EOS)
    sumlogp = lstm_output(model[:output], preds, gold)
    return -sumlogp
end

s2sgrad = grad(s2s)

function s2s_encode(inputs, model)
  state1 = initstate(inputs[1], model[:state1])
  state2 = initstate(inputs[1], model[:state2])
  for (forw_input, back_input) in zip(inputs, reverse(inputs))
    forw_input = lstm_input(model[:embed1], forw_input)
    forw_input = reshape(forw_input, 1, size(forw_input, 1))
    state1 = lstm(model[:encode1], state1, forw_input)

    back_input = lstm_input(model[:embed1], back_input)
    back_input = reshape(back_input, 1, size(back_input, 1))
    state2 = lstm(model[:encode2], state2, back_input)
  end

  return [ wbf2(state1[1], state2[1], model[:merge]), wbf2(state1[2], state2[2], model[:merge]) ]
end

function wbf2(x1, x2, params)
  wb1 = x1 * params[1] .+ params[3]
  wb2 = x2 * params[2] .+ params[4]
  return sigm(wb1+wb2)
end

function lstm_output(param, preds, gold)
    pred1 = vcat(preds...)
    pred2 = pred1 * param[1]
    pred3 = pred2 .+ param[2]
    sumlogp = logprob(gold, pred3)
    return sumlogp
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

function lstm_input(param, input)
    p = param[input,:]
    return p
end

function lstm_input_back(param, input, grads)
    dparam = zeros(param)
    dparam[input,:]=grads
    return dparam
end

@primitive lstm_input(param,input),grads lstm_input_back(param,input,grads)

function lstm(param, state, input)
    weight,bias = param
    hidden,cell = state
    h       = size(hidden,2)
    # map(x->println(size(x)), [input,hidden,weight,bias])

    gates   = hcat(input,hidden) * weight .+ bias
    forget  = sigm(gates[:,1:h])
    ingate  = sigm(gates[:,1+h:2h])
    outgate = sigm(gates[:,1+2h:3h])
    change  = tanh(gates[:,1+3h:4h])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function predict(param, input)
    o1 = input * param[1]
    o2 = o1 .+ param[2]
    return o2
end

function initstate(idx, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), length(idx), length(c)), 0)
    return (h,c)
end

function _s2s(model, inputs, outputs)
    state = initstate(inputs[1], model[:state0])
    for input in reverse(inputs)
        input = model[:embed1][input,:]
        state = lstm(model[:encode], state, input)
    end
    EOS = ones(Int, length(outputs[1]))
    input = model[:embed2][EOS,:]
    sumlogp = 0
    for output in outputs
        state = lstm(model[:decode], state, input)
        ypred = predict(model[:output], state[1])
        sumlogp += logprob(output, ypred)
        input = model[:embed2][output,:]
    end
    state = lstm(model[:decode], state, input)
    ypred = predict(model[:output], state[1])
    sumlogp += logprob(EOS, ypred)
    return -sumlogp
end
