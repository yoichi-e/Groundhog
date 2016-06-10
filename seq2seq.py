
from itertools import izip
from collections import OrderedDict
import timeit
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
numpy.random.seed(1)
#This script is almost the same as http://arxiv.org/abs/1409.0473
#Different point is that this script cant specify batch size.
#batch size will be the number of words of one sentence. 



#Parameters

n=1000#=hidden unit
dh=50#=m
nd=50#n  dash
l=50#=l(emb_size,dh,nd,vocab_size_s,vocab_size_t,l)
lr=0.01#Learning rate
rho=0.95
eps=1e-6
iteration=20






#/Making the dictionary of source sentences and target sentences
#and change the sentence to the series of corresponding numbers
f1=open("/home/yoichi-e/corpus/ali_source.txt","r")
f2=open("/home/yoichi-e/corpus/ali_target.txt","r")


wd_s={}
wd_t={}
source=[]
target=[]


for line in f1:
    print line
    line=line.split(" ")
    source.append(line)
for line in f2:
    print line
    line=line.split(" ")
    target.append(line)


x=2
for i in range(0,len(source)):
    print i
    for j in range(0,len(source[i])):
        wd_s[source[i][j].strip()]=-1
for i in range(0,len(target)):
    print i
    for j in range(0,len(target[i])):
        wd_t[target[i][j].strip()]=-1


source_num=[]
target_num=[]
n2w_s={}
n2w_t={}
for i in range(0,len(source)):
    print i
    sentence=[]
    for j in range(0,len(source[i])):
        if wd_s[source[i][j].strip()]==-1:
            wd_s[source[i][j].strip()]=x
            n2w_s[x]=source[i][j].strip()
            x=x+1
        
        sentence.append(wd_s[source[i][j].strip()])
    source_num.append(sentence)
    print sentence
        
y=2
for i in range(0,len(target)):
    print i
    sentence=[]
    for j in range(0,len(target[i])):
        if wd_t[target[i][j].strip()]==-1:
            wd_t[target[i][j].strip()]=y
            n2w_t[y]=target[i][j].strip()
            y=y+1
        sentence.append(wd_t[target[i][j].strip()])
    target_num.append(sentence)
    print sentence

n2w_s[0]="start"
n2w_t[0]="start"
n2w_s[1]="end"
n2w_t[1]="emd"


print "vocab_source=",x
print "target_num=",y
print "sentences_size",len(source),len(target)
Kx=x#Kx
Ky=y#Ky
#/Making the dictionary of source sentences and target sentences
#and change the sentence to the series of corresponding numbers







def orth_mat(x,y):
 z=numpy.random.randint(0,2,(x,y)).astype(theano.config.floatX)
 q, r = numpy.linalg.qr(z)
 return 0.1*q


def build_shared_zeros(shape, name):
    return theano.shared(
    	value=numpy.zeros(shape, dtype=theano.config.floatX), 
    	name=name, 
    	borrow=True
    )






def updates(parameters,gradients):
        #Optimization by Adadelta
        gradients_sq = [ build_shared_zeros(p.shape.eval(),'grad_sq') for p in parameters ]
        deltas_sq = [ build_shared_zeros(p.shape.eval(),'delta_sq') for p in parameters ]

        gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

        deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

        deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

        gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
        deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
        parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
        return gradient_sq_updates + deltas_sq_updates + parameters_updates




class Param(object):
    #parameters
    def __init__(self,n,m,nd,Kx,Ky,l):
       self.n=n
       self.E = theano.shared(numpy.random.normal(0, 0.1, (Kx, m)).astype(theano.config.floatX),borrow=True)
       self.W_f = theano.shared(numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wz_f = theano.shared(numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wr_f = theano.shared(numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.U_f = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Uz_f = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Ur_f = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.W_b = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wz_b = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wr_b = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.U_b = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Uz_b = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Ur_b = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Ws = theano.shared(   numpy.random.normal(0, 0.1, \
                (n, n)).astype(theano.config.floatX))
       
       self.E_d = theano.shared(   numpy.random.normal(0, 0.1, (Ky, m)).astype(theano.config.floatX),borrow=True)
       self.W = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wz = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.Wr = theano.shared(   numpy.random.normal(0, 0.1, \
                (m, n)).astype(theano.config.floatX))
       self.U = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Uz = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.Ur = theano.shared(orth_mat(n,n).astype(theano.config.floatX))
       self.C = theano.shared(   numpy.random.normal(0, 0.1, \
                (2* n, n)).astype(theano.config.floatX))
       self.Cz = theano.shared(   numpy.random.normal(0, 0.1, \
                (2* n, n)).astype(theano.config.floatX))
       self.Cr = theano.shared(   numpy.random.normal(0, 0.1, \
                (2*n, n)).astype(theano.config.floatX))
       self.va=theano.shared(numpy.zeros(nd).astype(theano.config.floatX))
       self.Wa = theano.shared(   numpy.random.normal(0, 0.01, \
                (n, nd)).astype(theano.config.floatX))
       self.Ua = theano.shared(   numpy.random.normal(0, 0.01, \
                (2*n, nd)).astype(theano.config.floatX))
       self.Wo=theano.shared(   numpy.random.normal(0, 0.1, \
                (Ky,l)).astype(theano.config.floatX))
       self.Uo=theano.shared(   numpy.random.normal(0, 0.1, \
                (n, 2*l)).astype(theano.config.floatX))
       self.Vo=theano.shared(   numpy.random.normal(0, 0.1, \
                (m, 2*l)).astype(theano.config.floatX))
       self.Co=theano.shared(   numpy.random.normal(0, 0.1, \
                (2*n, 2*l)).astype(theano.config.floatX))
       self.bfz=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bfr=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bf=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bbz=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bbr=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bb=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.ba=theano.shared(numpy.zeros(nd).astype(theano.config.floatX))
       self.bdz=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bdr=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bd=theano.shared(numpy.zeros(n).astype(theano.config.floatX))
       self.bo=theano.shared(numpy.zeros(2*l).astype(theano.config.floatX))
       self.h0 = theano.shared(   numpy.zeros(n).astype(theano.config.floatX))
       self.parameters=[self.E,self.E_d,self.W_f,self.Wz_f,self.Wr_f,self.U_f,self.Uz_f,self.Ur_f,self.W_b,self.Wz_b,self.Wr_b,self.U_b,self.Uz_b,self.Ur_b,self.W,self.Wz,self.Wr,self.U,self.Uz,self.Ur,self.C,self.Cz,self.Cr,self.va,self.Wa,self.Ua,self.Wo,self.Uo,self.Vo,self.Co,self.Ws,self.bfz,self.bfr,self.bf,self.bbz,self.bbr,self.bb,self.ba,self.bdz,self.bdr,self.bd,self.bo]



class encdec(object):
    def __init__(self,params):
        def birnn_forward(xe,hb):
               #forward
               zi_f=T.nnet.sigmoid(T.dot(xe, params.Wz_f)+T.dot(hb, params.Uz_f)+params.bfz)
               ri_f=T.nnet.sigmoid(T.dot(xe, params.Wr_f)+T.dot(hb, params.Ur_f)+params.bfr)
               h=(1-zi_f)*hb+zi_f*T.tanh(T.dot(xe, params.W_f)+T.dot(ri_f*hb, params.U_f)+params.bf)
               return h
        
        def birnn_backward(xe,hb):
               #backward
               zi_b=T.nnet.sigmoid(T.dot(xe, params.Wz_b)+T.dot(hb, params.Uz_b)+params.bbz)
               ri_b=T.nnet.sigmoid(T.dot(xe, params.Wr_b)+T.dot(hb, params.Ur_b)+params.bbr)
               h=(1-zi_b)*hb+zi_b*T.tanh(T.dot(xe, params.W_b)+T.dot(ri_b*hb, params.U_b)+params.bb)
               return h
        
        def cont(hi,sb):
           context=T.dot(T.nnet.softmax(T.dot(T.tanh(T.dot(hi,params.Ua)+T.dot(sb,params.Wa)+params.ba),params.va.T))[0],hi)
           return context
        
        def decode(ye,sb,h):
           context=cont(h,sb)
           zi=T.nnet.sigmoid(T.dot(ye, params.Wz)+T.dot(sb, params.Uz)+T.dot(context, params.Cz)+params.bdz)
           ri=T.nnet.sigmoid(T.dot(ye, params.Wr)+T.dot(sb, params.Ur)+T.dot(context, params.Cr)+params.bdr)
           s=(1-zi)*sb+zi*T.tanh(T.dot(ye, params.W)+T.dot(ri*sb, params.U)+T.dot(context, params.C)+params.bd)
           return s,context
        
        def y_train(yib,yi,sb,c,h):
            yib_emb=params.E_d[yib]
            ti=T.max((T.dot(sb,params.Uo)+T.dot(yib_emb,params.Vo)+T.dot(c,params.Co)+params.bo).reshape((l,2)),axis=1)
            p=-T.log(T.nnet.softmax(T.dot(params.Wo,ti.T))[0][yi])
            yi_emb=params.E_d[yi]
            s,c=decode(yi_emb,sb,h)
            return s,c,p
        
        def y_pred(sb,yib,c,h,end):
            yib_emb=params.E_d[yib]
            ti=T.max((T.dot(sb,params.Uo)+T.dot(yib_emb,params.Vo)+T.dot(c,params.Co)+params.bo).reshape((l,2)),axis=1)
            result_o=T.dot(params.Wo,ti.T)
            yi=T.argmax(result_o,axis=0)
            yi=yi.reshape((1,))
            si,ci=decode(params.E_d[yi],sb,h)
            return (si,yi,ci),theano.scan_module.until(T.eq(end,yi.dimshuffle('x')[0]))
        
        #encoder_decoder
        so=T.ivector()
        ta=T.ivector()
        s_e=params.E[so]
        t_e=params.E_d[ta]
        so_rev=so[::-1]
        s_e_rev=params.E[so_rev]
        ta1=T.concatenate([[0],ta],axis=0)
        ta2=T.concatenate([ta,[1]],axis=0)


        #encoder
        result_f, _ = theano.scan(fn=birnn_forward,sequences=s_e, outputs_info=params.h0)
        result_b, _ = theano.scan(fn=birnn_backward,sequences=s_e_rev,outputs_info=params.h0)
        h=T.concatenate([result_f,result_b[::-1]],axis=1)
        
        
        #decoder
        s0=T.dot(result_b[-1],params.Ws)
        c1=cont(h,s0)
        SCP, _ = theano.scan(fn=y_train,sequences=[ta1,ta2],outputs_info=[s0,c1,None],non_sequences=[h])
        p=SCP[2]
        

        #training
        sp=T.mean(p)
        grad=T.grad(sp, params.parameters)
        update=updates(params.parameters,grad)
        self.train=theano.function([so,ta],[p],updates=update)
        
        
        #prediction
        end=T.lscalar()
        y_ini=T.lscalar()
        y0=y_ini.dimshuffle('x')
        [s,yi,ci],_=theano.scan(fn=y_pred,outputs_info=[s0.reshape((1,n)),y0,c1],non_sequences=[h,end],n_steps=100)
        self.pred=theano.function([so,end,y_ini],[yi])



p=Param(n,dh,nd,Kx,Ky,l)
en=encdec(p)
f3=open("result.txt","w")



for itr in range(0,iteration):
    for i in range(0,len(source)):
        print itr,i
        print en.train(source_num[i],target_num[i])
        if i%1000==0:
            print target_num[i]
            print en.pred(source_num[i],1,0)



for i in range(0,len(source)):
    print i
    tar=en.pred(source_num[i],1,0)
    print tar
    for j in range(0,len(tar[0])):
        print n2w_t[tar[0][j][0]]
        f3.write(n2w_t[tar[0][j][0]])
        f3.write(" ")
    f3.write("\n")    
f1.close()
f2.close()
f3.close()   
