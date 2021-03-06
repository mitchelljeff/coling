import tensorflow as tf
import numpy as np
from random import shuffle


def batcher(data,epochs,batchsize):
    pointers=list(range(len(data["seq"])))
    for e in range(epochs):
        shuffle(pointers)
        for pointer in range(0,len(pointers),batchsize):
            batch={"seq":list(), "next":list(), "slen":list()}
            for p in pointers[pointer:pointer+batchsize]:
                batch["seq"].append(data["seq"][p])
                batch["next"].append(data["next"][p])
                batch["slen"].append(data["slen"][p])
            yield e,batch,len(batch["seq"])


if __name__ == '__main__':
    
    epochs    =100
    batch_size=128
    lstm_size =200
    emb_size  =20
    rate      = 0.001

    max_seq   =100

    vocab=dict()
    vocab["<PAD>"]=0
    vocab["<END>"]=1
    i=2
    with open("vocab") as f:
        for line in f:
            token=line.rstrip("\n")
            vocab[token]=i
            i=i+1
    v_size=len(vocab)


    train={"seq":list(), "next":list(), "slen":list()}
    dev={"seq":list(), "next":list(), "slen":list()}
    test={"seq":list(), "next":list(), "slen":list()}
    longdev={"seq":list(), "next":list(), "slen":list()}
    for split,fname in [(train,"abcdtrain"),(dev,"abcddev"),(test,"abcdtest")]:
        with open(fname) as f:
            for line in f:
                seq=np.zeros((max_seq,), dtype=int)
                next=np.zeros((max_seq,), dtype=int)
                tokens=line.split()
                assert len(tokens)<max_seq
                for i,token in enumerate(tokens):
                    seq[i]=vocab[token]
                    if i>0:
                        next[i-1]=vocab[token]
                next[i]=vocab["<END>"]
                slen=len(tokens)
                split["seq"].append(seq)
                split["next"].append(next)
                split["slen"].append(slen)
    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfnext =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfslen =tf.placeholder(tf.int32,shape=[None])
        tfbs   =tf.placeholder(tf.int32,shape=[])
        
        tfmask = tf.sequence_mask(tfslen,maxlen=max_seq,dtype=tf.float32)

        e=tf.get_variable("e", shape=[v_size, emb_size], initializer=tf.contrib.layers.xavier_initializer())
        
        w=tf.get_variable("w", shape=[1, 1, lstm_size, v_size], initializer=tf.contrib.layers.xavier_initializer())

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
        logits=tf.reduce_sum(tf.multiply(tf.expand_dims(outputs2,axis=-1),w),axis=2)


        loss = tf.contrib.seq2seq.sequence_loss(outputs2,tfnext,tfmask)
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
        acc_sum=0
        m_sum=0
        for epoch,batch,bs in batcher(train,epochs,batch_size):
            feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfslen:batch["slen"], tfbs:bs}
            _, loss_val = sess.run([optimizer,loss], feed_dict=feed_dict)

            loss_sum=loss_sum+loss_val
            n=n+bs
            if i%r == 0:
                print("train:",epoch,n,loss_sum/n)
                loss_sum=0
                n=0
            if (i-r)%dr == 0:
                dloss_sum=0
                dn=0
                for depoch,dbatch,dbs in batcher(dev,1,batch_size):
                    feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfslen:dbatch["slen"], tfbs:dbs}
                    dloss_val, = sess.run([loss], feed_dict=feed_dict)
                    dloss_sum=dloss_sum+dloss_val
                    dn=dn+dbs
                print("***** val:",epoch,dn,dloss_sum/dn, "*****")
                dloss_sum=0
                dn=0
            i=i+1





