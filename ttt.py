#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:39:18 2020

@author: gian rothfuchs
"""

import numpy as np
import random as rnd
import itertools as it
import collections as cl

class tictactoe:
    def __init__(self,bsize,p1,p2,showField,verbose):
        self.verbose = verbose
        self.boardSize = bsize
        self.p1 = p1
        self.p2 = p2
        self.name2num = {self.p1:1,self.p2:-1}
        self.num2name = {1:self.p1,-1:self.p2}
        self.transitionMap = {self.p1:self.p2,self.p2:self.p1}
        self.stateHash2Num, self.stateNum2Str, self.stateSpace = self.getStateSpace()
        self.resetBoard()
        self.showField = showField
        
    
        
    def resetBoard(self):
        self.board = np.zeros((self.boardSize,self.boardSize),dtype=np.int8)
        self.gameOver = False
        self.State = None
        self.rewardVal = 0
        self.emptyFields = self.boardSize**2
        self.actionSpace = self.getEmptyFields()
        self.randomStart()
        
    def randomStart(self):
        playerToken = np.array([1,-1])
        np.random.shuffle(playerToken)
        self.currentPlayer = self.num2name[playerToken[0]]
        if self.verbose: print(self.currentPlayer +" goes first")
        
    def getState(self):
        self.State = tuple(self.board.reshape(self.boardSize**2))
        return self.State 
    
    def getStateSpace(self):
        dictsStateStr2Num = {}
        dictStateNum2Str = {}
        stateList = []
        ctr = 0
        ctrState = 0
        for line in list(it.product(range(-1,2), repeat=self.boardSize**2)):
                
            ctr += 1
            occurrencesDict = cl.Counter(line)
            if occurrencesDict[1] <= int(round((self.boardSize**2+1)/2)) and occurrencesDict[-1] <= int(round((self.boardSize**2+1)/2)):
                dictsStateStr2Num[tuple(line)] = ctrState
                dictStateNum2Str[ctrState] = str(np.array(line))
                stateList.append(ctrState)
                ctrState += 1
        return dictsStateStr2Num, dictStateNum2Str, stateList
        
    def getWinner(self):
        col_p1 = np.any(np.equal(np.sum(self.board,0),np.full(self.boardSize, self.boardSize)))
        row_p1 = np.any(np.equal(np.sum(self.board,1),np.full(self.boardSize, self.boardSize)))
        diag1_p1 = np.equal(np.sum(np.multiply(self.board,np.eye(self.boardSize))),self.boardSize)
        diag2_p1 = np.equal(np.sum(np.multiply(self.board,np.rot90(np.eye(self.boardSize)))),self.boardSize)
        
        col_p2 = np.any(np.equal(np.sum(self.board,0),np.full(self.boardSize, -self.boardSize)))
        row_p2 = np.any(np.equal(np.sum(self.board,1),np.full(self.boardSize, -self.boardSize)))
        diag1_p2 = np.equal(np.sum(np.multiply(self.board,np.eye(self.boardSize))),-self.boardSize)
        diag2_p2 = np.equal(np.sum(np.multiply(self.board,np.rot90(np.eye(self.boardSize)))),-self.boardSize)
        
        #print(np.array([col_p1,row_p1,diag1_p1,diag2_p1]))
        if np.any(np.array([col_p1,row_p1,diag1_p1,diag2_p1])):
            self.gameOver = True
            if self.verbose: print('GAME OVER: ' + self.p1 +' wins')
            self.rewardVal = 1.0
            return self.p1 # p1 wins
        else:
            #print(np.array([col_p2,row_p2,diag1_p2,diag2_p2]))
            if np.any(np.array([col_p2,row_p2,diag1_p2,diag2_p2])):
                self.gameOver = True
                if self.verbose: print('GAME OVER: ' + self.p2 +' wins')
                self.rewardVal = -1.0
                return self.p2 #p2 wins
            else:
                if self.emptyFields == 0:
                    self.gameOver = True
                    if self.verbose: print('GAME OVER: Tie ')
                    self.rewardVal = 0.0
                    return 'tie' # tie
                else:    
                    self.rewardVal = -0.01
                    return 'na' # game in progres
                
    def getEmptyFields(self):
        self.emptyFields = np.sum(self.board == 0)
        if self.emptyFields > 0:
            res = np.where(self.board == 0)
            return list(zip(res[0], res[1]))
        else:
            return list()
    
    
    def doMove(self,field,player):
        otpt = False
        if self.currentPlayer == player:
            if field in self.getEmptyFields():
                if not self.gameOver:
                    self.board[field] = self.name2num[self.currentPlayer]
                    if self.showField:
                        print(self.currentPlayer + " played: {0}".format(field))
                        print(self.board)
                    self.currentPlayer = self.transitionMap[self.currentPlayer]
                    self.getWinner()
                    self.getEmptyFields()
                    otpt = True
                else:
                    if self.verbose: print("doMove: game over")
            else:
                if self.verbose: print("doMove: field is already taken")
        else:
            if self.verbose: print("doMove: Its not player " + player + "'s turn")
        return otpt

    def randomPlayer2(self):
        if self.p2 == self.currentPlayer:
            if not self.gameOver:
                availableFields = self.getEmptyFields()
                randArray = np.random.choice(len(availableFields), 1)
                randAction = availableFields[randArray.item()]
                self.doMove(randAction,self.p2)
            else:
                if self.verbose: print("game is over")
        else:
            if self.verbose: print("Its p1's turn, p2 cannot move now")
    
    def resetSim(self):
        self.resetBoard()
        if self.currentPlayer == self.p2:
            self.randomPlayer2()
        return self.stateHash2Num[self.getState()]
            
            
    def doSimMove(self,field):
        if not self.gameOver:
            if self.p1 == self.currentPlayer:
                ok = self.doMove(field,self.p1)
                self.getWinner()
                if ok:
                    if not self.gameOver:
                        self.randomPlayer2()
                        self.getWinner()
        return self.gameOver,self.stateHash2Num[self.getState()],self.rewardVal 
                



"""        
env = tictactoe(3,'p1','p2',False,False)

s =env.resetSim()

gameOverTrue = env.gameOver
counter = 0

actSpace = env.actionSpace
sizeActionSpace = len(actSpacte)
sizeStateSpace = len(env.stateSpace)


tf.reset_default_graph()

inputs1 = tf.placeholder(shape=[1,sizeStateSpace],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([sizeStateSpace,sizeActionSpace],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,sizeActionSpace],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 500
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    for i in range(num_episodes):
        #Reset environment and get first new observation
        if i % 10 == 0:
            print("This is episode: " + str(i).rjust(5))
        s = env.resetSim()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(sizeStateSpace)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = rnd.choice(range(sizeActionSpace))
            #Get new state and reward from environment
            d,s1,r = env.doSimMove(actSpace[a[0]])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(sizeStateSpace)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,actSpace[a[0]]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(sizeStateSpace)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
    saver.save(sess, "saved_models/testing_si")
    print("Model Saved.")
        
plt.plot(rList)
plt.plot(jList)
"""

