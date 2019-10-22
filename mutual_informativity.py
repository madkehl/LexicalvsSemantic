import gensim
from gensim.models import Word2Vec

import nltk
import pandas as pd


import collections
from collections import Counter
#nltk mutual informativity


#PMI is defined as pmi(r,c)=logP(r,c)P(r)P(c), with P(r,c) being the
#probability of co-occurrence and P(r) and P(c) the probability of
#occurrence of two words (estimated via frequency)

#- I considered words as co-occurring if they occurred within a window
#of 5 words:
#no no yes yes yes yes target yes yes yes yes no no

#This is buggy having difficulties with overlap.  Currently implementing the band-aid fix that if it's greater than the number 
#of instances of a word than replacing just with the total bc in all cases I saw it was a multiplicity issue
#e.g. they'd be seen two or four times when they only appear once (never 3)


def make_nltktxt(komyagin):
    #komyagin = komyagin.lower()
    #holodkov = komyagin.split()
    return(nltk.Text(komyagin))

def make_ci(nltkText):
    return(ConcordanceIndex(nltkText.tokens))

#left sides are commented out for gender code
def concordance(ci, word, width=75, lines= 50):
    """
    Rewrite of nltk.text.ConcordanceIndex.print_concordance that returns results
    instead of printing them. 

    See:
    http://www.nltk.org/api/nltk.html#nltk.text.ConcordanceIndex.print_concordance
    """
    half_width = (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context

    results = []
    offsets = ci.offsets(word)
    if offsets:
        lines = len(offsets)
        for i in offsets:
            if lines <= 0:
                break
            #left = (' ' * half_width + ' '.join(ci._tokens[i-context:i]))
            right = ' '.join(ci._tokens[i+1:i+context])
           # left = left[-half_width:]
            right = right[:half_width]
           # results.append('%s %s %s' % (left, ci._tokens[i], right))
            results.append( '%s %s' % (ci._tokens[i], right))
            lines -= 1

    return results



def concordance_fancy(ci, word, width=150, lines=100):
    
    half_width =  (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context
    num = 5
    problem_index = None
    results = []
    offsets = ci.offsets(word)
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
    concordance1 = concordance_fancy(ci, target_word)
    num = 5
    #print(num)
    for i in concordance1[1]:
        #print(i)
        n = i.split()
        for z in n:
            results0.append(z)
    token_fix = list(enumerate(results0))
    results1 = []
    counter = -1
    for z in token_fix:
        counter += 1
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
           
    results = []
    for i in results1:
        for n in i:
            results.append(n)
#this will return the count of target_word2 in the vicinity of target_word1
    prob_numx = Counter(results)
    prob_denom = len(results0)
    prob_num = prob_numx[target_word2]
    P_rc = prob_num/total_count
    P_r = (count(ci, target_word))/total_count
    P_c = (count(ci, target_word2))/total_count
    if P_rc == 0:
        print([num, concordance1, "ERROR"])
        return ([target_word, target_word2, "ERROR"])
    elif P_rc > P_c:
        mutinf = math.log10(P_c*P_r*P_c)
        print([P_rc, P_c, P_r])
    elif P_rc > P_r:
        mutinf = math.log10(P_r*P_r*P_c)
        print([P_rc, P_c, P_r])
    else:
        mutinf = math.log10(P_rc*P_r*P_c)
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
    for i in text3:
        if i[0] < (len(text3)-3):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 3][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        elif i[0] < (len(text3)-2):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 2][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        elif i[0] < (len(text3)-1):
            #print(text2[i[0] + 1])
            item = mutual_informativity(step_2, i[1], text3[i[0] + 1][1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        else:
            return(pmi_list)

def takeSecond(elem):
    #print(elem[2])
    return elem[2]

def pmi_high(pmi_output):
    cat = pmi_output
    cat.sort(key = takeSecond, reverse = True)
    return cat
