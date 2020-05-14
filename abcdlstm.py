import tensorflow as tf
import numpy as np
from random import shuffle


longlens=[14,16,18]


def batcher(data,epochs,batchsize):
    pointers=list(range(len(data["seq"])))
    for e in range(epochs):
        shuffle(pointers)
        for pointer in range(0,len(pointers),batchsize):
            batch={"seq":list(), "next":list(), "mask":list(), "bdmask":list(), "slen":list()}
            for p in pointers[pointer:pointer+batchsize]:
                batch["seq"].append(data["seq"][p])
                batch["next"].append(data["next"][p])
                batch["mask"].append(data["mask"][p])
                batch["bdmask"].append(data["bdmask"][p])
            yield e,batch,len(batch["seq"])


if __name__ == '__main__':
    
    epochs    =100
    batch_size=128
    lstm_size =200
    emb_size  =20
    rate      = 0.001

    max_seq   =100


    vocab={"pad":0, "a1":1, "b1":2, "a2":3, "b2":4, "c1":5, "d1":6, "c2":7, "d2":8, "c3":9, "d3":10, "e":11, "oo":12}
    v_size=len(vocab)


    train={"seq":list(), "next":list(), "mask":list(), "bdmask":list()}
    dev={"seq":list(), "next":list(), "mask":list(), "bdmask":list()}
    test={"seq":list(), "next":list(), "mask":list(), "bdmask":list()}
    longdev={"seq":list(), "next":list(), "mask":list(), "bdmask":list()}
    for split,fname in [(train,"abcdtrain"),(dev,"abcddev"),(test,"abcdtest")]:
        with open(fname) as f:
            for line in f:
                seq=np.zeros((max_seq,), dtype=int)
                next=np.zeros((max_seq,), dtype=int)
                mask=np.zeros((max_seq,), dtype=float)
                bdmask=np.zeros((max_seq,), dtype=float)
                slen=np.zeros((1),dtype=int)
                chars=line.split()
                assert len(chars)<max_seq
                for i,char in enumerate(chars):
                    seq[i]=vocab[char]
                    mask[i]=1
                    if char[1]=="2":
                        bdmask[i-1]=1
                    if i>0:
                        next[i-1]=vocab[char]
                next[i]=vocab["e"]
                split["seq"].append(seq)
                split["next"].append(next)
                split["mask"].append(mask)
                split["bdmask"].append(bdmask)

    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfnext =tf.placeholder(tf.int32,shape=[None,max_seq])
        tfmask =tf.placeholder(tf.float32,shape=[None,max_seq])
        tfbdmask =tf.placeholder(tf.float32,shape=[None,max_seq])
        tfbs   =tf.placeholder(tf.int32,shape=[])

        e=tf.get_variable("e", shape=[v_size, emb_size], initializer=tf.contrib.layers.xavier_initializer())
        
        w=tf.get_variable("w", shape=[lstm_size, v_size], initializer=tf.contrib.layers.xavier_initializer())

        embedded=tf.gather(e,tfseq)

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        state = lstm.zero_state(tfbs,dtype=tf.float32)
        inputs=tf.unstack(embedded,axis=1)
        outputs=[None]*max_seq
        for i in range(max_seq):
            o, state = lstm(inputs[i], state)
            outputs[i]=tf.matmul(o,w)
        logits = tf.stack(outputs,axis=1)

        loss = tf.contrib.seq2seq.sequence_loss(logits,tfnext,tfmask)
        pred = tf.cast(tf.argmax(logits,axis=2),tf.int32)
        #acc=tf.constant(0)
        correct=tf.multiply(tf.cast(tf.equal(tfnext,pred),tf.float32),tfbdmask)
        acc=tf.reduce_sum(correct)
        msum=tf.reduce_sum(tfbdmask)
        accbypos=tf.constant(0)
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
            feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfmask:batch["mask"], tfbdmask:batch["bdmask"], tfbs:bs}
            _, loss_val, acc_val, m_val = sess.run([optimizer,loss,acc,msum], feed_dict=feed_dict)

            loss_sum=loss_sum+loss_val
            acc_sum=acc_sum+acc_val
            m_sum=m_sum+m_val
            n=n+bs
            if i%r == 0:
                print("train:",epoch,n,loss_sum/n,acc_sum/m_sum)
                loss_sum=0
                acc_sum=0
                m_sum=0
                n=0
            if (i-r)%dr == 0:
                dloss_sum=0
                dacc_sum=0
                dm_sum=0
                daccbypos_sum=0
                dn=0
                for depoch,dbatch,dbs in batcher(dev,1,batch_size):
                    feed_dict={tfseq:dbatch["seq"], tfnext:dbatch["next"], tfmask:dbatch["mask"], tfbdmask:dbatch["bdmask"], tfbs:dbs}
                    dloss_val, dacc_val, dm_val = sess.run([loss,acc, msum], feed_dict=feed_dict)
                    dloss_sum=dloss_sum+dloss_val
                    dacc_sum=dacc_sum+dacc_val
                    dm_sum=dm_sum+dm_val
                    #daccbypos_sum=daccbypos_sum+daccbypos_val
                    dn=dn+dbs
                print("***** val:",epoch,dn,dloss_sum/dn,dacc_sum/dm_sum,daccbypos_sum/dn, "*****")
                dloss_sum=0
                dacc_sum=0
                dm_sum=0
                daccbypos_sum=0
                acc14_sum=0
                acc16_sum=0
                acc18_sum=0
                sum14_sum=0
                sum16_sum=0
                sum18_sum=0
                dn=0
            i=i+1





