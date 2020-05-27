import tensorflow as tf
import numpy as np
from random import shuffle
from time import time
from math import exp

def padlist(l,max_seq):
    return l+[0]*(max_seq-len(l))

def batcher(data,epochs,batchsize):
    pointers=dict()
    batches=list()
    for l in data:
        n=len(data[l]["seq"])
        pointers[l]=list(range(n))
        batches=batches+[(l,batchsize)]*(n//batchsize)
        if n%batchsize > 0:
            batches=batches+[(l,n%batchsize)]
    for e in range(epochs):
        shuffle(batches)
        pointer=dict()
        for l in data:
            shuffle(pointers[l])
            pointer[l]=0
        for l,bs in batches:
            batch={"seq":list(), "next":list(), "slen":list()}
            for p in pointers[l][pointer[l]:pointer[l]+bs]:
                batch["seq"].append(padlist(data[l]["seq"][p],l))
                batch["next"].append(padlist(data[l]["next"][p],l))
                batch["slen"].append(data[l]["slen"][p])
            pointer[l]=pointer[l]+bs
            yield e,batch,l,bs


if __name__ == '__main__':
    
    epochs    =40
    batch_size=64
    lstm_size =650
    emb_size  =650
    rate      = 10
    lens      = [100]
    dropout   = 0.2
    keep_prob = 1-dropout
    clip      = 0.25

    max_seq   =25

    vocab=dict()
    vocab["<PAD>"]=0
    vocab["<END>"]=1
    i=2
    with open("vocab.txt") as f:
        for line in f:
            token=line.rstrip("\n")
            vocab[token]=i
            i=i+1
    v_size=len(vocab)

    eval=dict()
    for l in lens:
        for split in [eval]:
            split[l]={"seq":list(), "next":list(), "slen":list()}
    for split,fname,tname in [(eval,"generated.text","generated.tab")]:
        cdict=dict()
        with open(tname) as t:
            flag=0
            for i,line in enumerate(t):
                if flag==0:
                    flag=1
                else:
                    l=(i-1)//2
                    c=line.split("\t")[6]
                    if l in cdict:
                        assert cdict[l]==c, str(l)+" "+c+" "+cdict[l]
                    else:
                        cdict[l]=c
        with open(fname) as f:
            for j,line in enumerate(f):
                seq=list()
                next=list()
                tokens=line.split()
                for i,token in enumerate(tokens):
                    if token in vocab:
                        t=vocab[token]
                    else:
                        t=vocab["<unk>"]
                    seq.append(t)
                    if i>0:
                        next.append(t)
                next.append(vocab["<END>"])
                slen=len(tokens)
                if cdict[j]=="original":
                    for l in lens:
                        if slen<=l:
                            split[l]["seq"].append(seq)
                            split[l]["next"].append(next)
                            split[l]["slen"].append(slen)
                            break
    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,None])
        tfnext =tf.placeholder(tf.int32,shape=[None,None])
        tfslen =tf.placeholder(tf.int32,shape=[None])
        tfl    =tf.placeholder(tf.int32,shape=[])
        tfbs   =tf.placeholder(tf.int32,shape=[])
        
        tfmask = tf.sequence_mask(tfslen,maxlen=tfl,dtype=tf.float32)

        e=tf.get_variable("e", shape=[v_size, emb_size], initializer=tf.contrib.layers.xavier_initializer())
        
        w=tf.get_variable("w", shape=[lstm_size, v_size], initializer=tf.contrib.layers.xavier_initializer())

        embedded=tf.gather(e,tfseq)

        lstm1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_size),input_keep_prob=keep_prob,output_keep_prob=keep_prob)
        state1 = lstm1.zero_state(tfbs,dtype=tf.float32)
        lstm2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_size),output_keep_prob=keep_prob)
        state2 = lstm2.zero_state(tfbs,dtype=tf.float32)
        #inputs=tf.unstack(embedded,axis=1)
        #outputs=[None]*max_seq
        #for i in range(max_seq):
        #    o, state = lstm(inputs[i], state)
        #    outputs[i]=tf.matmul(o,w)
        #logits = tf.stack(outputs,axis=1)
        
        outputs1, state=tf.nn.dynamic_rnn(lstm1, embedded, sequence_length=tfslen, initial_state=state1, dtype=tf.float32, scope="layer1")
        outputs2, state=tf.nn.dynamic_rnn(lstm2, outputs1, sequence_length=tfslen, initial_state=state2, dtype=tf.float32, scope="layer2")
        logits=tf.tensordot(outputs2,w,axes=[[2],[0]])


        loss = tf.contrib.seq2seq.sequence_loss(logits,tfnext,tfmask)
        opt = tf.train.GradientDescentOptimizer(rate)
        gvs  = opt.compute_gradients(loss)
        grads=[grad for grad,var in gvs]
        vs   =[var for grad,var in gvs]
        clipped_gs, norm = tf.clip_by_global_norm(grads, clip)
        clipped_gvs = zip(clipped_gs, vs)
        descend = opt.apply_gradients(clipped_gvs)
        #init = tf.initialize_all_variables()

        saver=tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess,"./model.ckpt")

        i=1
        n=0
        r=1
        dr=1000
        best_dloss=1000000
        loss_sum=0
        loss_val=0
        acc_sum=0
        m_sum=0
        start=time()
        for epoch in range(1):
            for e,batch,l,bs in batcher(eval,1,batch_size):
                feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfslen:batch["slen"], tfl:l, tfbs:bs}
                loss_val, = sess.run([loss], feed_dict=feed_dict)

                loss_sum=loss_sum+loss_val*bs
                n=n+bs
        print("eval:",epoch,n,loss_sum/n,exp(loss_sum/n),(time()-start)/n)





