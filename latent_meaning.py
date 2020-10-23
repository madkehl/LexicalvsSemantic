import gensim
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

model3 = Word2Vec.load('AAPECS.model')

def latent_meaning(i):
    if(i[0] in model3.wv.vocab) & (i[1] in model3.wv.vocab):
        first_close = list(model3.wv.most_similar(i[0], topn= 5))
        #print([i, first_close])
        second_close = list(model3.wv.most_similar(i[1], topn= 5))
        first_vec = model3.wv.get_vector(i[0])
        second_vec = model3.wv.get_vector(i[1])
        item_dis = dot(first_vec, second_vec)/(norm(first_vec)*norm(second_vec))
        for z in first_close:
            first_vec = first_vec + model3.wv.get_vector(z[0])
        for n in second_close:
            #print(n)
            second_vec = second_vec + model3.wv.get_vector(n[0])
        first_vec = first_vec - model3[i[0]]
        second_vec = second_vec - model3[i[1]]
        cos_sim = dot(first_vec, second_vec)/(norm(first_vec)*norm(second_vec))
        return([i, item_dis, cos_sim])


        

        
