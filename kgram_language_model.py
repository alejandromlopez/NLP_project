import nltk
import random

class LanguageModel:
    
    def __init__(self, text, k, n, seed=0):
        self.modelSeed = seed
        self.kgramLen = k
        self.ngramLen = n
        cfd = nltk.ConditionalFreqDist()
        i = 0
        for i in range(len(text)-n):
            
            condition = tuple(text[i:i+k])
            cond = tuple(text[i+k:i+k+n])
            cfd[condition][cond] += 1
        
        self.condfd = cfd

    def generate(self, starting_tokens, length, sep=' '):
        l = list()
        i = len(starting_tokens)
        j = 0
        s = ""
        before = self.kgramLen
        for j in range(len(starting_tokens)):
            l.append(starting_tokens[j])
        tok = tuple(starting_tokens[-before:])
    
        for i in range(length):
            x = list(self.sample(tok))
            k = 0
            for k in range(len(x)):
                if (len(l) == length):
                    s = sep.join(l)
                    return s
                else:
                    l.append(x[k])
            tok = tuple(l[-before:])
        s = sep.join(l)
        return s
        
    
    def sample(self, condition):
        if (self.condfd[condition]):
            l = list()
            for w in self.condfd[condition]:
                l.append(self.condfd[condition].freq(w))
            random.seed(self.modelSeed)
            response = random.choices(list(self.condfd[condition]), weights=l, k=1)
            return tuple(response[0])
        else:
            raise KeyError

if __name__ == '__main__':
    # generate list of vocab words from which to randomly
    # select a starting token for generate function

    
    inter_lines = {}

   # with open('transcriptions/vox_extraction.txt', 'r') as interview:
    with open('all_lines.txt', 'r', encoding='utf-8', errors='ignore') as interview:
       
        inter_lines = interview.readlines()

    
    text = [word.lower() for line in inter_lines for word in nltk.word_tokenize(line)]
   
    #print("text")
    #print(text)
    model = LanguageModel(text, 3, 2)
    
    #print(list(model.condfd))
    
    st = random.choice(list(model.condfd))

    #print(tuple(st))

    genResult = model.generate(st, 20, ' ')
    print(genResult)
    