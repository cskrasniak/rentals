
import numpy as np
import pandas as pd
import pickle
import numbers


def remove_md(string):
    """
    Removes angled brackets (< & >) and the markdown formatting inside them from strings. Should 
    remove anything between two angled brackets, anything before a close bracket if there is only a
    close bracket, and anything after an open bracket if there is only an open bracket
    
    Input : str
    output : str
    """

    # if both an open and close angle brackets are in the string
    if ('<' in string) and ('>' in string):
        open_brac = string.find('<')
        close_brac = string.find('>')
        # if there is something inside the brackets, drop that including the brackets
        if open_brac < close_brac:
            string = string.replace(string[open_brac : close_brac + 1],"")
        # if there isn't anything in between them, drop everything before and after
        else:
            string = string.replace(string[:close_brac+1],"")
            string = string.replace(string[open_brac:],"")
        # use recursion to fix any instances where there are multiple opens and closes
        return remove_md(string)
    elif ('<' in string):
        open_brac = string.find('<')
        string = string.replace(string[open_brac:],"")
        return remove_md(string)
    elif ('>' in string):
        close_brac = string.find('>')
        string = string.replace(string[:close_brac+1],"")
        return remove_md(string)
    else:
        return string


def drop_words(word_list):
    """
    Removes stop-words or words that don't really add any meaning so we want to remove to limit
    the number of features in our final data set.

    Input :
        word_list (list of str)
    Output :
        list of str
    """

    wl = list(word_list)
    stop_words = ['this', 'that','we', 'an','be', 'from', 'you', 'for', 'of', 'with', 'is', 'in',
                  'to', 'a','the', 'can','things','also', 'my','us', 'and', 'are', 'your', 'w/',
                  'has', 'on', 'have', 'or', '&', 'will', 'as', 'at', 'it', '-', 'so', '•', 'i',
                  '+','eg', '|', '/']
    
    for sw in stop_words:
        if sw in wl:
            wl.remove(sw)
    return wl


def ordered_unique(seq):
    """
    Takes the unique values of a list, but unlike np.unique, it returns those unique values in the
    same order in which the first occurence appears so ordered_unique([1,1,2,4,3,4]) = [1,2,4,3]
    
    Input : 
        seq (iterable) 

    Output : 
        list
    """

    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_dictionary(load=True, text=None, n_words=4000, overWrite=False, fileDir=\
        '/Users/ckrasnia/Documents/application_materials/rental_data/dictionary.pkl'):
    """
    Retrieves a dictionary of unnique words, where each word is a key and each key is assigned a unique integer. Default
    is to load a previously saved file, but to generate a new dictionary, set load=False and provide a text
    
    Input :
        load (bool) defaults to True, if the dictionary should be loade from fileDir
        text (list of strings) a list of strings from which to generate the dictionary
        n_words (int) number of words to include in the dictionary
        overWrite (bool) only applies if text is provided, determines if the dictionary should be saved
        fileDir (str) the file that contains the dictionary / where the dictionary will be saved if overWrite=True
        
    Returns : 
        a dictionary where there are n_words keys which are strings and the values are a unique integer assigned
        according to the frequency of the words with 1 = most frequent. 0 is reserved for words not in the dictionary
    """
    
    if text != None:
        print('creating new dictionary...')
        u,cts = np.unique(text, return_counts=True)
        words = list(u[np.argsort(cts)[-n_words:]])
        words.reverse()# reverse so that 1 is the most frequent word
        vals = np.arange(1,n_words+1) # reserve 0 for words that are not in the top 4000
        dictionary = {word:val for word, val in zip(words,vals)}
        
        if overWrite:
            f = open(fileDir,"wb")
            pickle.dump(dictionary,f)
            print('new dictionary saved to {}'.format(fileDir))
            f.close()
    elif load: # load file if thats what we want
        print('loading pregenerated dictionary')
        f = open(fileDir,"rb")
        dictionary = pickle.load(f)
        f.close()

    else:
        raise Exception('dictionary must either be loaded with load=True or created by providing \
            a list of strings to the text argument')
        
    return dictionary


def clean_text(column : pd.Series):
    """
    written to clean up the text data to translate it into numerical values for input to a model.
    steps: 
    1. changes nan values to empty strings
    2. lowercases all text
    3. removes markdown formatting
    4. removes special characters
    5. splits into a list of words
    6. drops a list of stopwords to pare down to a set of only meaningful words
    7. stems words with the porter stemming algorithm to remove endings to reduce the number of
        unique words
    8. only keeps the unique words of the list, while preserving order
    
    Input:
        column (pd.Series) with each row containing a string containing with multiple words
        
    Output:
        (pd.Series) the cleaned column with a list of unique words in each row
    """
    column[column.isna()] = ''
    #look at the original distribution of how many words there are per description
    descript_words = column.str.lower()

    # first start by getting rid of ugly formatting 
    descript_words = descript_words.apply(remove_md)
    #also going to want to remove some punctuation
    descript_words = descript_words.str.replace("[():!.,?]","")
    descript_words = descript_words.str.split()

    # get rid of meaningless stop words
    descript_words = descript_words.apply(drop_words)

    # lematize words so that to reduce unique words 
    # this is pretty time intensive due to the loop over the list within each of the rows
    porter = PorterStemmer()
    descript_words = descript_words.map(lambda x: [porter.stem(y) for y in x])

    # now I want unique words within each description
    descript_words = descript_words.apply(ordered_unique)

    # sometimes stop-words reappear after lematization, so try dropping them again
    descript_words = descript_words.apply(drop_words)
    return descript_words


def translate(word_list, dictionary):
    """
    A lossy conversion of words to numbers or numbers back to words. if word_list contains strings,
    it will translate to using the dictionary, if it is ints, it will translate to strings using
    the reverse translation. For words that are not in the dictionary, it will be mapped to None.
    for the reverse translation, 0 gets mapped to None
    
    Input : 
        word_list (list of int or str) list that you want translated
        dictionary (dict with str as keys mapped to a unique int) used to translate between words
            and ints
        
    Returns :
        a translated version of the list, the same length as the input word_list
    """

    # catch if there is no description
    try:
        if len(word_list) == 0:
            translated = np.array([0])

        
        # translate to integers
        elif type(word_list[0]) == str:
            translated = np.array([*map(dictionary.get, word_list)])
            translated[translated==None] = 0

        # translate to strings
        elif np.any([isinstance(x, numbers.Number) for x in word_list]):
            inverse_dict = {v: k for k, v in dictionary.items()}
            translated = np.array([*map(inverse_dict.get, word_list)])
            
        # if a list of Nones, return self as thats the best translation we can do
        elif (np.all([x == None for x in word_list])):
            translated = np.zeros(len(word_list))
            
    except TypeError: # catches the case that there is only a None
        translated = np.array([0])
        
    return translated

def preprocess_text(column, dictionary=None, return_dict = False):
    """
    Applies both the clean_column function and translate function to yield a numerical series
    
    Input : 
        column (pd.Series) a column with rows containing strings, most should have multiple words
        dictionary (None or dict) if None, a dictionary is created from the words in the column and
            the column is translated with that new dictionary, else a dictionary with words as keys
            and unique integers as values
        return_dict (bool) if true, the dictionary will also be returned
        
    Returns : 
        (pd.Series) the same length as the column input with a 1d array of integers in each row who's values are mapped
            to words using the dictionary either provided or generated here. 
        [optional] 
        (dict) a dictionary assigning words to unique integers based on their frequency, see get_dictionary for details
    """
    
    print('cleaning the text...')
    
    cleaned_column = clean_text(column)
    if dictionary == None:
        n = 50000
        idx = np.random.choice(np.arange(len(cleaned_column)),n)
        word_list = [w for j in cleaned_column.iloc[idx] for w in j]
        dictionary = get_dictionary(text=word_list)
        
    print('applying the dictionary')
    
    if return_dict:
        return cleaned_column.apply(translate,args=(dictionary,)), dictionary
    
    return cleaned_column.apply(translate,args=(dictionary,))

##################################################################################################

"""Porter Stemming Algorithm
This is the Porter stemming algorithm, ported to Python from the
version coded up in ANSI C by the author. It may be be regarded
as canonical, in that it follows the algorithm presented in

Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14,
no. 3, pp 130-137,

only differing from it at the points maked --DEPARTURE-- below.

See also http://www.tartarus.org/~martin/PorterStemmer

The algorithm as described in the paper could be exactly replicated
by adjusting the points of DEPARTURE, but this is barely necessary,
because (a) the points of DEPARTURE are definitely improvements, and
(b) no encoding of the Porter stemmer I have seen is anything like
as exact as this version, even with the points of DEPARTURE!

Vivake Gupta (v@nano.com)

Release 1: January 2001

Further adjustments by Santiago Bruno (bananabruno@gmail.com)
to allow word input not restricted to one word per line, leading
to:

release 2: July 2008

Modified by CSK to make it more user friendly for my purpose, changed the stem method
"""

import sys

class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i=0, j=None):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        if j == None:
            j = len(p)-1
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

