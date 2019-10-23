import gensim
from gensim.models import Word2Vec

import nltk
import pandas as pd


import collections
from collections import Counter
#nltk mutual informativity

#note from fritz with // around my edit
#PMI is defined as //pmi(r,c)=log(P(r,c)/(P(r)*P(c)))//, with P(r,c) being the
#probability of co-occurrence and P(r) and P(c) the probability of
#occurrence of two words (estimated via frequency)

#- I considered words as co-occurring if they occurred within a window
#of 5 words:
#no no yes yes yes yes target yes yes yes yes no no


def make_nltktxt(komyagin):
    #komyagin = komyagin.lower()
    #holodkov = komyagin.split()
    return(nltk.Text(komyagin))

def make_ci(nltkText):
    return(ConcordanceIndex(nltkText.tokens))

#for some reason having difficulty subsetting by tokens instead of characters:: instead just use enough 
#characters to later be able to reliably get a 5 word radius

def concordance_fancy(ci, word, width=150, lines=100):
    
    half_width =  (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context
    num = 5
    problem_index = None
    results = []
    offsets = ci.offsets(word)
    #vestigial code from trying to get tokens
   # if len(offsets) > 1:
    #    distances = [j-i for i, j in zip(offsets[:-1], offsets[1:])] 
     #   distances1 = list(enumerate(distances))
      #  for i in distances1:
       #     if i[1] < 10:
        #        num = round((i[1]-1)/2)
         #       problem_index = offsets[0]
          #      print(problem_index)
           #     print(ci._tokens[23])
    if offsets:
        #print([offsets, word])
        for i in offsets:
            query_word = ci._tokens[i]
            #print("q_w" + query_word)
                # Find the context of query word.
            left_context = ci._tokens[max(0, i - context) : i]
            right_context = ci._tokens[i + 1 : i + context]
           # print(right_context)
                # Create the pretty lines with the query_word in the middle.
            left_print = " ".join(left_context)[-half_width:]
            right_print = " ".join(right_context)[:half_width]
                # The WYSIWYG line of the concordance.
            line_print = " ".join([left_print, query_word, right_print])
                # Create the ConcordanceLine
            results.append(line_print)
    return ([num, results, problem_index])

#gathers all contexts of the word in results
def mutual_informativity(ci, target_word, target_word2, total_count):
    results0 = []
 #gets context around a target word
    concordance1 = concordance_fancy(ci, target_word)
    num = 5
    #print(num)
 #takes this context and reshapes it into a -5 to + 5 window
    for i in concordance1[1]:
        #print(i)
        n = i.split()
        for z in n:
            results0.append(z)
    token_fix = list(enumerate(results0))
    results1 = []
#modifications necessary to account for words at the start and end of corpus
    for z in token_fix:
        if z[1] == target_word:
            if z[0] < (num):
                cat = results0[0:(z[0]+num)]
                results1.append(cat)
            elif z[0] > (len(results0) - 1):
                cat = results0[(z[0] -num):(len(results0)-1)]
                results1.append(cat)
            else: 
                cat = results0[(z[0] -num):(z[0] + num)]
                results1.append(cat)
#reshaping list into format that can be more quickly processed    
    results = []
    for i in results1:
        for n in i:
            results.append(n)
#this will return the count of target_word2 in the vicinity of target_word1
    prob_numx = Counter(results)
    prob_num = prob_numx[target_word2]
    P_rc = prob_num/total_count
    P_r = (count(ci, target_word))/total_count
    P_c = (count(ci, target_word2))/total_count
 #checking for indexing errors and overlap errors.
#fix to overlap is a bit of a hack fix, most notable problems are double counting 
#so if number of cooccurances is greater than number of times a word appears in 
#a document, then it replaces cooccurance with the total number of times
#except in the case that this is an odd number (since it's not being double-counted then,
#it's being double counted once and has another appearance)
#odd numbers, is P_c - 1/total. 
#this still may cause some errors
    if P_rc == 0:
        print([num, concordance1, "ERROR"])
        return ([target_word, target_word2, "ERROR"])
    elif P_rc > P_c:
        rounded = int(P_rc/P_c)
        nr = P_rc/P_c
        if (rounded > nr):
            P_n = P_c - 1/total_count
        else:
            P_n = P_c
        mutinf = math.log10(P_n/(P_r*P_c))
        #print([P_rc, P_c, P_r, P_n])
    elif P_rc > P_r:
        rounded = int(P_rc/P_r)
        nr = P_rc/P_r
        if (rounded > nr):
            P_n = P_c -1/total_count
        else:
            P_n = P_c
        mutinf = math.log10(P_n/(P_r*P_c))
        #print([P_rc, P_c, P_r, P_n])
    else:
        mutinf = math.log10(P_rc/(P_r*P_c))
    return ([target_word, target_word2, mutinf])

def count(ci, word):
   
    offsets = ci.offsets(word)

    return len(offsets)

from nltk import ConcordanceIndex
import math
def pmi(text):
    text0 = text.lower()
    text1 = text.split()
    text1 = [i for i in text1 if i not in stop_words]
    total_count = len(text1)
    step_1 = make_nltktxt(text1)
    step_2 = make_ci(step_1)
    pmi_list = []
    text2 = []
    text2 = [i for i in text1 if i not in text2]
    text3 = list(enumerate(text2))
    index = 0
    #realistically only words that at some point occur in a 3 word window are really worth looking at esp with these short texts
    for i in text3:
        if i[0] < (len(text3)-1):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 1][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        if i[0] < (len(text3)-2):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 2][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)        
        if i[0] < (len(text3)-3):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 3][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        else:
            return(pmi_list)
#just to allow for sorting by actual pmi index
def takeSecond(elem):
    #print(elem[2])
    return elem[2]

def pmi_high(pmi_output, n):
    cat = pmi_output
    cat.sort(key = takeSecond, reverse = True)
    return cat[:n]

def pmi_low(pmi_output, n):
    cat = pmi_output
    cat.sort(key = takeSecond, reverse = False)
    return cat[:n]
