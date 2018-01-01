import numpy as np

# Read text from file
# Split it into char set , unique char set

data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d chars, %d unique' % (data_size, vocab_size)


# Map each char into number and number to char

char_to_ix = { ch:i for i,ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}

# Model params
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# weight INPUT to HIDDEN
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01

# weight LAST HIDDEN STATE to HIDDEN
Whh = np.random.randn(hidden_size, hidden_size) * 0.01

# weight HIDDEN to OUTPT
Why = np.random.randn(vocab_size, hidden_size) * 0.01

# bias of HIDDEN
bh = np.zeros((hidden_size, 1))

# bias of OUTPUT
by = np.zeros((vocab_size, 1))


def lossFun(input, targets, hprev):

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # Forward pass
    for t in xrange(len(inputs)):
        # Encode the input char in to vector
        xs[t] = np.zeros((vocab_size,1))
        xs[t][input[t]] = 1

        # Calculate the hidden layer:
        # res = (W*x) + (hW*last_hidden_state)  + B
        # res = tanh(res)
        hs[t] = np.tanh( np.dot( Wxh, xs[t] ) + np.dot(Whh,hs[t-1]) + bh )

        # Un-normalized
        ys[t] = np.dot(Why,hs[t]) + by

        # Soft-max (cross-entropy loss)
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        loss += -np.log(ps[t][targets[t], 0])

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    # Backward pass
    for t in reversed(xrange(len(inputs))):
        # output probabilities
        dy = np.copy(ps[t])
        # derive our first gradient

        # Formular in notebook
        # OUTPUT -> HIDDEN
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)

        # derivative of output bias
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h

        # derivative of tanh function
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw  # derivative of hidden bias

        # INPUT -> HIDDEN
        dWxh += np.dot(dhraw, xs[t].T)  # derivative of input to hidden layer weight

        # HIDDEN -> HIDDEN
        dWhh += np.dot(dhraw, hs[t - 1].T)  # derivative of hidden layer to hidden layer weight

        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5,out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    n is how many characters to predict
    """

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print '----\n %s \n----' % (txt, )


##
## MAIN LOOP
##

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

while n<=1000*100:

    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # sample from the model now and then
    if n % 1000 == 0:
        print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
        sample(hprev, inputs[0], 200)

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter
