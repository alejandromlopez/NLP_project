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

        # tag corpus with POS and, when generating tweet (by calling sample function), take into account 
        # most likely POS to follow last word of previous k-gram
        # check to see if any of most probably n-grams begins with this POS, and weight selection of next n-gram accordingly
        tagged_text = nltk.pos_tag(text)
        pos = [word[1] for word in tagged_text]
        pos_bigrams = list(nltk.bigrams(text))
        #pos_cfd = nltk.ConditionalFreqDist(pos_bigrams)
        pos_cfd = nltk.ConditionalFreqDist()
        for partos in pos_bigrams:
            condition = tuple(list(partos[0])) #to keep in same format as other cfd, standardize generate/sample function
            cond = tuple(list(partos[1]))
            pos_cfd[condition][cond] += 1

        self.pos_cfd = pos_cfd

    def generate(self, starting_tokens, length, sep=' '):
        l = list()
        i = len(starting_tokens)
        j = 0
        s = ""
        before = self.kgramLen
        for j in range(len(starting_tokens)):
            l.append(starting_tokens[j])
        tok = tuple(starting_tokens[-before:])
        pos_tok = tuple(starting_tokens[-1])

    
        for i in range(length):
            # pass previous token, both k-gram and pos, to sample function
            x = list(self.sample(tok, pos_tok))
            k = 0
            for k in range(len(x)):
                if (len(l) == length):
                    s = sep.join(l)
                    return s
                else:
                    l.append(x[k])
            tok = tuple(l[-before:])
            pos_tok = tuple(l[-1])
        s = sep.join(l)
        return s
        
    
    def sample(self, condition, pos_condition):

        if (self.condfd[condition] and self.pos_cfd[pos_condition]):
            l = list()
            pos_l = list()
            most_likely_pos = self.pos_cfd[pos_condition].max()
            # alter list of weights so that words with the most likely to folow pos correspond to higher weight
            for w in self.condfd[condition]:
                pos_w = nltk.pos_tag(list(w))
                mult_factor = 1
                if pos_w == most_likely_pos:
                    mult_factor = mult_factor * 2
                l.append(self.condfd[condition].freq(w) * mult_factor)
            random.seed(self.modelSeed)
            response = random.choices(list(self.condfd[condition]), weights=l, k=1)
        
            return tuple(response[0])
        else:
            raise KeyError

if __name__ == '__main__':

    inter_lines = {}

    with open('all_lines.txt', 'r', encoding='utf-8') as interview:
       
        inter_lines = interview.readlines()

    
    text = [word.lower() for line in inter_lines for word in nltk.word_tokenize(line)]
   
    model = LanguageModel(text, 3, 2)
    
    st = random.choice(list(model.condfd))

    genResult = model.generate(st, 20, ' ')
    print(genResult)
    