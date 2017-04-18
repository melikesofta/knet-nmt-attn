function initmodel(H, SV, TV, atype)
    init(d...)=atype(xavier(d...))
    model = Dict{Symbol,Any}()
    model[:state0] = [ init(1,H), init(1,H) ]
    model[:embed1] = init(SV,H)
    model[:encode] = [ init(2H,4H), init(1,4H) ]
    model[:embed2] = init(TV,H)
    model[:decode] = [ init(2H,4H), init(1,4H) ]
    model[:output] = [ init(H,TV), init(1,TV) ]
    return model
end

function s2s(model, inputs, outputs)             #
    state = initstate(inputs[1], model[:state0]) # 14
    for input in reverse(inputs)
        # input = model[:embed1][input,:]
        input = lstm_input(model[:embed1], input) # 85
        input = reshape(input, 1, size(input, 1))
        state = lstm(model[:encode], state, input) # 723
    end
    EOS = ones(Int, length(outputs[1]))
    # input = model[:embed2][EOS,:]
    input = lstm_input(model[:embed2], EOS) # 3
    preds = []
    sumlogp = 0

    for output in outputs
        state = lstm(model[:decode], state, input) # 702
        push!(preds, state[1])
        # ypred = predict(model[:output], state[1])
        # sumlogp += logprob(output, ypred)
        # input = model[:embed2][output,:]
        input = lstm_input(model[:embed2],output) # 61
        input = reshape(input, 1, size(input, 1))
    end
    state = lstm(model[:decode], state, input) # 30
    push!(preds, state[1])
    # ypred = predict(model[:output], state[1])
    # sumlogp += logprob(EOS, ypred)
    gold = vcat(outputs..., EOS) # 1
    sumlogp = lstm_output(model[:output], preds, gold) # 2441
    return -sumlogp
end

s2sgrad = grad(s2s)

function lstm_output(param, preds, gold)
    #for pred in preds
    #  print(int2tok[indmax(pred)], " ")
    #end
    #println()
    pred1 = vcat(preds...) # 46
    pred2 = pred1 * param[1] # 242
    pred3 = pred2 .+ param[2] # 145
    sumlogp = logprob(gold, pred3) # 2006
    return sumlogp
end


function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)     # 1999
    o2 = o1[index]         # 4
    o3 = sum(o2)           # 2
    return o3
end

function lstm_input(param, input)
    p = param[input,:]     # 118
    return p
end

function lstm_input_back(param, input, grads)
    dparam = zeros(param)  # 157
    dparam[input,:]=grads  # 121
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
