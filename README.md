Name
====

Overview

## How to use
 Change them  to your own path.   
f1=open("/home/yoichi-e/Deepl/data/sample_source.txt","r")

f2=open("/home/yoichi-e/Deepl/data/sample_target.txt","r")

f3=open("result.txt","w")

f4=open("/home/yoichi-e/Deepl/data/sample_test.txt","r")

## Command
 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python seq2seq.py
 
## Environment
 THEANO 0.9.0

