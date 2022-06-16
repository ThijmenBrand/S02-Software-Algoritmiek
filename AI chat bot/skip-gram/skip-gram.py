import numpy as np
import string
from nltk.corpus import stopwords
import torch
import numpy as np
import torch.functional as F
import torch.nn.functional as F

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class word2vec(object):
    def __init__(self):
        #N: het aantal neuronen in de hidden layer
        self.N = 20
        #X: onehot encoding van het input woord
        self.X_train = []
        #Y: softmax output laag met mogelijkheden van ieder woord in het ingegeven vocabulair
        self.y_train = []
        #window_size: het aantal woorden het algoritme neemt voor de training sample
        self.window_size = 2
        #alpha: de stap waarmee het non-convex wordt opgelost om de minimale loss te vinden
        self.alpha = 0.001
        #words: alle woorden in het vocabulair
        self.words = []
        #word_inex: de indexen van de woorden
        self.word_index = {}

    def initialize(self,V,data):
        #V: het aantal unieke woorden in de vocabulair
        self.V = V
        #W: weights tussen input layer en hidden layer
        #Deze functie kent de weights toe op basis van een uniforme verdeling over een interval half open interval met [aantalUniekeWoorden, AantalNeuronen)
        self.W = np.random.randn(self.V, self.N)
        #W1: weight tussen hidden layer en output layer
        self.W1 = np.random.randn(self.N, self.V)
        
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    #X is onehot encoding matrix van de input woorden [[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0]...]
    def feed_forward(self,X):
        #h is de eerste hidden layer in het netwerk en wanneer we de input layer X vermenigvuldigen met de eerste weights W, dan krijgen we een matrix van dimensie N x 1
        self.h = np.dot(self.W.T,X).reshape(self.N,1)
        #na de hidden layer vermenigvuldigen we de nodes van de hidden layer met de tweede weights W1 naar de output layer.
        self.u = np.dot(self.W1.T,self.h)
        #Voor de we bij de output layer zijn moeten we nog een activatie functie toepassen om de waarde van de output layer te berekenen. dit is de softmax functie.
        self.y = softmax(self.u)
        return self.y
        
    def backpropagate(self,x,t):
        #e is de error van de output layer
        e = self.y - np.asarray(t).reshape(self.V,1)
        #dLdW1 is de derivative van de loss met de weights van de hidden layer naar de output layer
        dLdW1 = np.dot(self.h,e.T)
        #X veranderen van een miltidimensionale array naar een dimensiele array om deze te kunnen vermenigvuldigen met de weights van de hidden layer
        X = np.array(x).reshape(self.V,1)
        #e vermenigvuldigen met de weights van de output layer naar de hidden layer en vervolgens vermenigvuldigen met de weights van de input layer
        dLdW = np.dot(X, np.dot(self.W1,e).T)
        #W en W1 aanpassen volgens de gradiënten van de loss
        self.W1 = self.W1 - self.alpha*dLdW1
        self.W = self.W - self.alpha*dLdW
        
    def train(self,epochs):
        #Loop over iedere epoch heen. dit geeft aan hoevaak we de training data moeten doorlopen.
        for x in range(1,epochs):	
            self.loss = 0
            #Voor ieder center woord gaan we de context woorden van de training data doorlopen.
            for j in range(len(self.X_train)):
                #Het center woord in feed forward geeft ons een context
                self.feed_forward(self.X_train[j])
                #Via back prop geven we het center woord met de gewilde context om te trainen
                self.backpropagate(self.X_train[j],self.y_train[j])
                C = 0
                #Voor ieder uniek woord in de hele training zin
                for m in range(self.V):
                    #voor ieder uniek center woord in de zin, kijk je naar de context woorden van de training data.
                    if(self.y_train[j][m]):
                        #als de context woorden van de training data het woord met index m zijn, dan is het woord correct. dus de loss van deze zin is 0.
                        self.loss += -1*self.u[m][0]
                        C += 1
                #Bereken de totale loss van deze epoch om vervolgens in de backprop toegepast te kunnen worden.
                self.loss += C*np.log(np.sum(np.exp(self.u)))
            #Verklijnt de loss om overschieten te voorkomen
            self.alpha *= 1/( (1+self.alpha*x) )
            
    def predict(self,word,number_of_predictions):
        #Wanneer het woord in de oorspronkelijke zin bestaat, kunnen de context woorden worden voorspeld. Anders niet.
        if word in self.words:
            #Verander het woord naar een one-hot encoding om het door het netwerk te kunnen sturen.
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            #Feed forward de input woorden
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
            
            top_context_words = []
            #Door iedere index van de omgekeerde output
            for k in sorted(output,reverse=True):
                #Het oorspronkelijke woord dat in de top context zit toevoegen aan de return array
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
    
            return top_context_words
        else:
            print("Word not found in dictionary")

def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))
    training_data = []
    #Haal alle zinnen uit de corpus woorden
    sentences = corpus.split(".")
    #Voor elke zin in de corpus woorden
    for i in range(len(sentences)):
        #Haal alle whitespaces uit de zin
        sentences[i] = sentences[i].strip()
        #Array met alle woorden in de zin
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence
                                     if word not in stop_words]
        x = [word.lower() for word in x]
        training_data.append(x)

        print(training_data)

    return training_data

def prepare_data_for_training(sentences,w2v):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    
    #for i in range(len(words)):
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
            
            for j in range(i-w2v.window_size,i+w2v.window_size):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)
    w2v.initialize(V,data)

    return w2v.X_train,w2v.y_train

corpus = ""
corpus += """Shall I compare thee to a summer’s day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer’s lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm’d;
And every fair from fair sometime declines,
By chance or nature’s changing course untrimm’d;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wander’st in his shade,
When in eternal lines to time thou growest:
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee."""
epochs = 10000

training_data = preprocessing(corpus)
print(training_data)
w2v = word2vec()

prepare_data_for_training(training_data,w2v)
w2v.train(epochs)
centerword = corpus.split( )
outcome = w2v.predict('thee',3)
print(outcome)
