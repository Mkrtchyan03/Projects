import numpy as np
import string
import matplotlib.pyplot as plt

# Function to load the text from a file
def load_doc(filenam):
    file = open(filenam, 'r')
    text = file.read()
    file.close()
    return text

# Path to the text or book file
path = ''  # Add the path to the text or book
text = load_doc(path)

# Function to preprocess the text
def data_prepocessing(text):
  tokens = text.split()
  tokens = [t for t in tokens if t not in string.punctuation]
  tokens = [t for t in tokens if t.isalpha()]
  tokens = [t.lower() for t in tokens]
  tokens = ' '.join(tokens)
  return tokens

raw_text = data_prepocessing(text)

chars = sorted(list(set(raw_text)))
char_to_index = {c:ind for ind,c in enumerate(chars)}
index_to_char = {ind:c for ind,c in enumerate(chars)}

encoded_ind = list()
for word in raw_text:
    encoded_ind.append(char_to_index[word])

vocab_size = len(char_to_index)

# Convert the encoded indices to one-hot encoded vectors
ohe = list()
for i in encoded_ind:
    enc = [0]*vocab_size
    enc[i] = 1
    ohe.append(enc)
ohe = np.array(ohe)

target = "" #target what to predict
target_ohe = [0]*vocab_size
target_ohe[char_to_index[target]] = 1
target_ohe = np.array(target_ohe)

class Rnn:
    def __init__(self, seq_length, hidden_size, vocab_size):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # self.Wx = np.random.uniform(-np.sqrt(1/self.vocab_size), np.sqrt(1/self.vocab_size), (self.hidden_size, self.vocab_size))
        # self.Wy = np.random.uniform(-np.sqrt(1/self.hidden_size), np.sqrt(1/self.hidden_size), (self.vocab_size, self.hidden_size))
        # self.Wh = np.random.uniform(-np.sqrt(1/self.hidden_size), np.sqrt(1/self.hidden_size), (self.hidden_size, self.hidden_size))
        # self.bias_h = np.zeros((self.hidden_size, 1))
        # self.bias_o = np.zeros((self.vocab_size, 1))

        xavier_factor_Wx = np.sqrt(1 / self.vocab_size)
        xavier_factor_Wy = np.sqrt(1 / self.hidden_size)
        xavier_factor_Wh = np.sqrt(1 / (self.hidden_size + self.hidden_size))

        self.Wx = np.random.uniform(-xavier_factor_Wx, xavier_factor_Wx, (self.hidden_size, self.vocab_size))
        self.Wy = np.random.uniform(-xavier_factor_Wy, xavier_factor_Wy, (self.vocab_size, self.hidden_size))
        self.Wh = np.random.uniform(-xavier_factor_Wh, xavier_factor_Wh, (self.hidden_size, self.hidden_size))
        self.bias_h = np.zeros((self.hidden_size, 1))
        self.bias_o = np.zeros((self.vocab_size, 1))


    def block(self, x, h):
        x = np.array(x).reshape(-1,1)
        h_t = np.tanh(np.dot(self.Wh, h) + np.dot(self.Wx, x) + self.bias_h)
        y_t = np.dot(self.Wy, h_t) + self.bias_o
        return np.array(h_t), y_t

    def forward(self, input):

        self.hidden_states =np.zeros((self.seq_length, self.hidden_size))
        self.inputs = np.array(input)
        self.outputs = np.zeros((self.seq_length, self.vocab_size))
        self.probs = np.zeros((self.seq_length, self.vocab_size))

        ht = np.zeros((self.hidden_size, 1))

        for s in range(len(input)):
            ht, yt = self.block(input[s], ht)
            self.outputs[s] += yt.reshape(self.vocab_size)
            self.hidden_states[s] += ht.reshape(self.hidden_size)
            self.probs[s] += self.softmax(yt).reshape(self.vocab_size)
        self.yt = yt
        return self.outputs

    def backward(self, probs, ind, learning_rate, betta1=0.9, betta2=0.999, eps=1e-8, eta=0.001):
        n = len(self.inputs)

        d_Wh = np.zeros(self.Wh.shape)
        d_Wx = np.zeros(self.Wx.shape)
        d_Wy = np.zeros(self.Wy.shape)
        d_bh = np.zeros(self.bias_h.shape)
        d_bo = np.zeros(self.bias_o.shape)
        dh_next = np.zeros_like(self.hidden_states[0])

        for t in reversed(range(n)):
            dy = probs[t]
            dy = dy.reshape(-1, 1)

            d_Wy += np.dot(dy, self.hidden_states[t].reshape(1, -1))
            d_bo += dy
            d_h = np.dot(dy.T, self.Wy) + dh_next

            tan_der = (1 - (self.hidden_states[t])**2)
            d_bh += (tan_der * d_h).T
            d_Wh += np.dot((tan_der*d_h).T, self.hidden_states[t].reshape(1, -1))
            d_Wx += np.dot((tan_der*d_h).T, self.inputs[t].reshape(1, -1))
            dh_next = np.dot((tan_der*d_h), self.Wh)

        for dparam in [d_Wh, d_Wx, d_Wy, d_bh, d_bo]:
            np.clip(dparam, -1, 1, out=dparam)

        self.Wh -= learning_rate * d_Wh
        self.Wx -= learning_rate * d_Wx
        self.Wy -= learning_rate * d_Wy
        self.bias_h -= learning_rate * d_bh
        self.bias_o -= learning_rate * d_bo


        # WITH ADAM OPTIMIZER #
        #
        # m_Wh, m_Wx, m_Wy = 0, 0, 0
        # m_bh, m_bo = 0, 0
        #
        # v_Wh, v_Wx, v_Wy = 0, 0, 0
        # v_bh, v_bo = 0, 0
        #
        # m_Wh = betta1*m_Wh + (1-betta1) * d_Wh
        # m_Wx = betta1*m_Wx + (1-betta1) * d_Wx
        # m_Wy = betta1*m_Wy + (1-betta1) * d_Wy
        # m_bh = betta1*m_bh + (1-betta1) * d_bh
        # m_bo = betta1*m_bo + (1-betta1) * d_bo
        #
        # v_Wh = betta2*v_Wh + (1-betta2) * (d_Wh**2)
        # v_Wx = betta2*v_Wx + (1-betta2) * (d_Wx**2)
        # v_Wy = betta2*v_Wy + (1-betta2) * (d_Wy**2)
        # v_bh = betta2*v_bh + (1-betta2) * (d_bh**2)
        # v_bo = betta2*v_bo + (1-betta2) * (v_bo**2)
        #
        # new_m_Wh = m_Wh / (1-betta1**ind)
        # new_m_Wx = m_Wx / (1-betta1**ind)
        # new_m_Wy = m_Wy / (1-betta1**ind)
        # new_m_bh = m_bh / (1-betta1**ind)
        # new_m_bo = m_bo / (1-betta1**ind)
        #
        # new_v_Wh = v_Wh / (1-betta2**ind)
        # new_v_Wx = v_Wx / (1-betta2**ind)
        # new_v_Wy = v_Wy / (1-betta2**ind)
        # new_v_bh = v_bh / (1-betta2**ind)
        # new_v_bo = v_bo / (1-betta2**ind)
        #
        # self.Wh -= eta * new_m_Wh / (np.sqrt(new_v_Wh)+eps)
        # self.Wx -= eta * new_m_Wx / (np.sqrt(new_v_Wx)+eps)
        # self.Wy -= eta * new_m_Wy / (np.sqrt(new_v_Wy)+eps)
        # self.bias_h -= eta * new_m_bh / (np.sqrt(new_v_bh)+eps)
        # self.bias_o -= eta * new_m_bo / (np.sqrt(new_v_bo)+eps)

    def softmax(self, data):
        exps_d = np.exp(data)
        return exps_d / np.sum(exps_d)

    def loss(self, target, probs):
        return sum(-np.log(probs[i][target[i]]) for i in range(len(target)))

    def call(self, input, target, epochs, lr):
        history = []
        for i in range(1, epochs+1):
            err = 0
            out = self.forward(input)
            err += self.loss(target, self.probs)
            prb = self.probs - target
            self.backward(prb, learning_rate=lr, ind=i)
            history.append(err)
            print(f"epoch {i}, Loss {err / len(target)}")

        return history

    def predict(self, input):
        ht = np.zeros((self.hidden_size, 1))
        for s in range(len(input)):
            ht, _ = self.block(input[s], ht)

        _, yt = self.block(input[-1], ht)
        next_char_index = np.argmax(self.softmax(yt))
        return next_char_index