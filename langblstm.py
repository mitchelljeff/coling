import tensorflow as tf
import numpy as np
from random import shuffle
from time import time

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
        shuffle(pointers)
        shuffle(batches)
        pointer=dict()
        for l in data:
            pointer[l]=0
        for l,bs in batches:
            batch={"seq":list(), "next":list(), "slen":list()}
            for p in pointers[pointer[l]:pointer[l]+bs]:
                batch["seq"].append(padlist(data[l]["seq"][p],l))
                batch["next"].append(padlist(data[l]["next"][p],l))
                batch["slen"].append(data[l]["slen"][p])
            pointer[l]=pointer[l]+bs
            yield e,batch,bs


if __name__ == '__main__':
    
    epochs    =40
    batch_size=64
    lstm_size =650
    emb_size  =650
    rate      = 0.001
    lens      = [25,50,100]

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

    train=dict()
    dev=dict()
    test=dict()
    for l in lens:
        for split in [train,dev,test]:
            split[l]={"seq":list(), "next":list(), "slen":list()}
    for split,fname in [(train,"train.txt"),(dev,"valid.txt"),(test,"test.txt")]:
        with open(fname) as f:
            for line in f:
                seq=list()
                next=list()
                tokens=line.split()
                for i,token in enumerate(tokens):
                    seq.append(vocab[token])
                    if i>0:
                        next.append(vocab[token])
                next.append(vocab["<END>"])
                slen=len(tokens)
                for l in lens:
                    if slen<=l:
                        split[l]["seq"].append(seq)
                        split[l]["next"].append(next)
                        split[l]["slen"].append(slen)
    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfnext =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfslen =tf.placeholder(tf.int32,shape=[None])
        tfbs   =tf.placeholder(tf.int32,shape=[])
        
        tfmask = tf.sequence_mask(tfslen,maxlen=max_seq,dtype=tf.float32)

        e=tf.get_variable("e", shape=[v_size, emb_size], initializer=tf.contrib.layers.xavier_initializer())
        
        w=tf.get_variable("w", shape=[lstm_size, v_size], initializer=tf.contrib.layers.xavier_initializer())

        embedded=tf.gather(e,tfseq)

        lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_size)
        state1 = lstm1.zero_state(tfbs,dtype=tf.float32)
        lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_size)
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
        optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
        init = tf.initialize_all_variables()

        saver=tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        i=1
        n=0
        r=100
        dr=1000
        loss_sum=0
        loss_val=0
        acc_sum=0
        m_sum=0
        start=time()
        for epoch,batch,bs in batcher(train,epochs,batch_size,max_seq):
            feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfslen:batch["slen"], tfbs:bs}
            _, loss_val = sess.run([optimizer,loss], feed_dict=feed_dict)

            loss_sum=loss_sum+loss_val
            n=n+bs
            if i%r == 0:
                print("train:",epoch,n,loss_sum/n,(time()-start)/n)
                loss_sum=0
                n=0
                start=time()
            if (i-r)%dr == 0:
                dloss_sum=0
                dn=0
                #for depoch,dbatch,dbs in batcher(dev,1,batch_size):
                #    feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfslen:dbatch["slen"], tfbs:dbs}
                #    dloss_val, = sess.run([loss], feed_dict=feed_dict)
                #    dloss_sum=dloss_sum+dloss_val
                #    dn=dn+dbs
                #print("***** val:",epoch,dn,dloss_sum/dn, "*****")
                dloss_sum=0
                dn=0
            i=i+1





