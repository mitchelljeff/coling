import tensorflow as tf
import numpy as np
from random import shuffle
from time import time
from math import exp
import numpy as np
from scipy.spatial.distance import cosine

splookup={"sing":{"correct":"sing", "wrong":"plur"}, "plur":{"correct":"plur", "wrong":"sing"}}

def padlist(l,max_seq):
    return l+[0]*(max_seq-len(l))

def batcher(data,epochs,batchsize):
    batches=list()
    n=len(data["seq"])
    pointers=list(range(n))
    batches=batches+[batchsize]*(n//batchsize)
    if n%batchsize > 0:
        batches=batches+[n%batchsize]
    for e in range(epochs):
        shuffle(batches)
        pointer=0
        for bs in batches:
            batch={"seq":list(), "next":list(), "slen":list(), "correct":list(), "wrong":list(), "form":list(), "from":list(), "s_id":list()}
            for i,p in enumerate(pointers[pointer:pointer+bs]):
                batch["seq"].append(padlist(data["seq"][p],100))
                batch["next"].append(padlist(data["next"][p],100))
                batch["slen"].append(data["slen"][p])
                batch["correct"].append([i,data["pos"][p],data["correct"][p]])
                batch["wrong"].append([i,data["pos"][p],data["wrong"][p]])
                batch["form"].append(data["form"][p])
                batch["from"].append(data["from"][p])
                batch["s_id"].append(data["s_id"][p])
            pointer=pointer+bs
            yield e,batch,l,bs


if __name__ == '__main__':
    
    epochs    =40
    batch_size=64
    lstm_size =650
    emb_size  =650
    rate      = 10
    lens      = [100]
    dropout   = 0.0
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
    for s_id in range(410):
        eval[s_id]=dict()
        eval[s_id]={"seq":list(), "next":list(), "slen":list(), "correct":list(), "wrong":list(), "pos":list(), "form":list(), "from":list(), "s_id":list()}
    fname,tname,ename = ("generated.text","generated.tab","generated.eval")
    tdict=dict()
    with open(ename) as e:
        for i,line in enumerate(e):
            t=int(line)
            tdict[i]=t
    cdict=dict()
    spdict={"correct":dict(), "wrong":dict()}
    fdict ={"correct":dict(), "wrong":dict()}
    splist=list()
    ids   =dict()
    with open(tname) as t:
        flag=0
        for i,line in enumerate(t):
            if flag==0:
                flag=1
            else:
                l=(i-1)//2
                c=line.split("\t")[6]
                sp=line.split("\t")[4]
                cw=line.split("\t")[5]
                form=line.split("\t")[3]
                if l in cdict:
                    assert cdict[l]==c, str(l)+" "+c+" "+cdict[l]
                    assert len(spd)==1, str(spd)
                    spd[splookup[form][cw]]=vocab[sp]
                    if c=="original":
                        splist.append(spd)
                else:
                    cdict[l]=c
                    spd={splookup[form][cw]:vocab[sp]}
                spdict[cw][l]=vocab[sp]
                fdict[cw][l]=form
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
                eval[j]["seq"].append(seq)
                eval[j]["next"].append(next)
                eval[j]["slen"].append(slen)
                eval[j]["correct"].append(spdict["correct"][j])
                eval[j]["wrong"].append(spdict["wrong"][j])
                eval[j]["pos"].append(tdict[j])
                eval[j]["form"].append(fdict["correct"][j])
                eval[j]["from"].append("original")
                eval[j]["s_id"].append(j)
                ids[j]={"original":list(), "generated":list()}
                for spd in splist:
                    c=fdict["correct"][j]
                    w=splookup[fdict["correct"][j]]["wrong"]
                    if spd[c]!=spdict["correct"][j]:
                        eval[j]["seq"].append(seq)
                        eval[j]["next"].append(next)
                        eval[j]["slen"].append(slen)
                        eval[j]["correct"].append(spd[c])
                        eval[j]["wrong"].append(spd[w])
                        eval[j]["pos"].append(tdict[j])
                        eval[j]["form"].append(fdict["correct"][j])
                        eval[j]["from"].append("generated")
                        eval[j]["s_id"].append(j)
    
    for s_id in range(410):
        if len(eval[s_id]["seq"])==0:
            del eval[s_id]
    
    graph=tf.Graph()

    with graph.as_default():

        tfseq  =tf.placeholder(tf.int32,shape=[None,None])
        tfnext =tf.placeholder(tf.int32,shape=[None,None])
        tfslen =tf.placeholder(tf.int32,shape=[None])
        tfl    =tf.placeholder(tf.int32,shape=[])
        tfbs   =tf.placeholder(tf.int32,shape=[])

        tfcorrect  =tf.placeholder(tf.int32,shape=[None,3])
        tfwrong    =tf.placeholder(tf.int32,shape=[None,3])        
        
        
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
        
        clogit=tf.gather_nd(logits,tfcorrect)
        wlogit=tf.gather_nd(logits,tfwrong)
        cwdiff=clogit-wlogit
        
        lstmvars=tf.trainable_variables(scope="layer")
        lstmgrads=tf.gradients(cwdiff,lstmvars)


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
        n_correct=0
        acc_sum=0
        m_sum=0
        start=time()
        for s_id in eval:
            vs=dict()
            fdict=dict()
            f2dict=dict()
            i=0
            for e,batch,l,bs in batcher(eval[s_id],1,1):
                feed_dict={tfseq:batch["seq"], tfnext:batch["next"], tfslen:batch["slen"], tfcorrect:batch["correct"], tfwrong:batch["wrong"], tfl:l, tfbs:bs}
                grad_vals, = sess.run([lstmgrads], feed_dict=feed_dict)
                flat=list()
                for arr in grad_vals:
                    flat.append(arr.flatten())
                v=np.concatenate(flat)
                j=batch["s_id"][0]
                ids[j][batch["from"][0]].append(v)
                fdict[j]=batch["form"][0]
                #print(i)
                i=i+1
                n=n+bs
            for i in [s_id]:
                assert len(ids[i]["original"])==1, str(len(ids[i]["original"]))
                v1=ids[i]["original"][0]
                for v2 in ids[i]["generated"]:
                    c=cosine(v1,v2)
                    print(c,fdict[i],fdict[i],fdict[i]==fdict[i],"same")
                ids[i]["generated"]=list()
        for i in ids:
            if len(ids[i]["original"])==1:
                v1=ids[i]["original"][0]
                for j in ids:
                    if len(ids[j]["original"])==1 and i != j:
                        v2=ids[j]["original"][0]
                        c=cosine(v1,v2)
                        print(c,fdict[i],fdict[j],fdict[i]==fdict[j],"diff")





