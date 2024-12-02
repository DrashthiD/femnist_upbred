import numpy as np
from split_data import split_data as sd
from model import gradient
import time
from multiprocessing import Pool
import random
import json
class Agent:
    def __init__(self,b,c,d,e,data,target,m, device):
        self.b=b  #beta for accuracy function
        self.c=c  #cost for per data sample
        k = int(d / 10)
        self.d = d - k #max contribution possible
        self.e=e  #current strategy
        self.data=data #entire dataset available to the agent
        self.target=target #targets for the entire dataset
        self.data_split=self.data[0:int(self.e)] #The data that the agent trains on
        self.target_split=self.target[0:int(self.e)] #targets for the training data
        self.data = data[k:]
        self.target = target[k:]
        self.data_test = data[0:k]
        self.target_test = target[0:k]
        self.data_split = self.data[0:int(self.e)]
        self.target_split = self.target[0:int(self.e)]
        self.e_old = 1
        self.grad = np.zeros(m)
        self.loss=0
        self.loss_o = 0
        self.device = device
        self.r_i=1
        self.b_i=1
    #Utility for agent
    def get_utility(self,w,s,n):
        a_i=self.get_accuracy(w,s)
        sminus=(s-self.e)/(n-1)
        p=self.b*(self.e-sminus)
        u_i=a_i-self.c*(self.e)+p
        return u_i
    def get_utility_np(self,w,s,n):
        a_i=self.get_accuracy_np(w,s)
        sminus=(s-self.enp)/(n-1)
        p=0
        u_i=a_i-self.c*(self.enp)+p
        return u_i
    #payment for the agent
    def get_payment(self,w,s,n):
        sminus=(s-self.e)/(n-1)
        p=self.b*(self.e-sminus)
        return p
    #accuracy for the agent
    def get_accuracy(self,w,s):
        self.loss=-gradient(self.data_test, self.target_test, w)[1]
        a=self.r_i+self.b_i*self.loss
        print(a,self.loss)
        return a
    def get_accuracy_np(self,w,s):
        loss=gradient(self.data_test,self.target_test,w)[1]
        a=self.a_i-self.b_i*loss/(s)
        return a
    #gradient of the agents utility wrt strategies
    def gradient_u_s(self, s, w_o, w):
        self.get_accuracy(w,s)
        w = self.loss - self.loss_o
        if self.e - self.e_old == 0:
            w = 0
        else:
            w = w / (self.e - self.e_old)
        grad_a = -self.b_i*w - self.c + self.b
        self.loss_o=self.loss
        return grad_a
    def gradient_u_s_np(self,s,w):
        loss=gradient(self.data_split,self.target_split,w)[1]
        grad_a=self.b_i*(loss)/(s**2)-self.c
        #grad_a=(loss)/(s**2)
        return grad_a
    #cost incurred by the agent
    def get_cost(self):
        return self.e*self.c
    #gradient of the accuracy function for the agent
    def gradient_acc_w(self,s,w):
        grad=-self.b_i*gradient(self.data_split,self.target_split,w)[0]
        return grad
    def gradient_acc_w_FedAvg(self,s,w):
        grad=-self.b_i*gradient(self.data,self.target,w)[0]
        return grad
    def gradient_acc_w_np(self,s,w):
        grad=-self.b_i*gradient(self.data_split,self.target_split,w)[0]/s
        return grad
    #Updating agents strategy
    def update(self,delta,s,w_o,w):
        k = self.gradient_u_s(s, w_o, w)
        self.e_old = self.e
        self.e = self.e + delta * k
        if self.e<=1 and k<0:
            self.e=1
            k=0
        elif self.e>=self.d and k>0:
            self.e=self.d
            f=0
        self.data_split=self.data[0:int(self.e)]
        self.target_split=self.target[0:int(self.e)]
        return self.e,k
    def update_np(self,delta,s,w):
        k=self.gradient_u_s_np(s,w)
        self.e=self.e+delta*k
        if self.e<=1 and k<0:
            self.e=1
            k=0
        elif self.e>=self.d and k>0:
            self.e=self.d
            k=0
        #Updating the resulting dataset
        # self.data_split_np=self.data[0:int(self.enp)]
        # self.target_split_np=self.target[0:int(self.enp)]
        return self.e,k   
def worker(x) :
    device="cuda"
    n=60
    data={"Accuracy_alldata":np.zeros(n),"iterations":0,"costs":np.full(n,0.005),"beta":np.full(n,1),"s_max":np.full(n,1000),"Initial":np.full(n,x),"utility":np.zeros(n),"payment":np.zeros(n),"accuracy":np.zeros(n),"time":0,"w":0,"s":0}
     #Number of agents   #current strategy
    e=data["Initial"] 
    d=data["s_max"]
    m=50890 
    split_data,split_target=sd(n,data["s_max"][0:n-1])
    person=[]
    #Initializing values for each agent in the system
    
    #without payments upbred
    for i in range(n):
        person.append(Agent(data["beta"][i],data["costs"][i],data["s_max"][i],data["Initial"][i],split_data[i],split_target[i],m,device))
    print("this is good",len(person))
    
    k=np.zeros(n)
    w=np.zeros(m)  #Initial model parameter values
    ds=np.ones(n)  
    dw=np.random.rand(n,m)
    u=np.zeros(n)
    e=data["Initial"]
    delta=0.5  #stepsize for strategy updates
    eta=0.5 
    for i in range(n):
        print(person[i].update_np(delta,np.sum(e),w))
       #stepsize for parameter updates
    #split_data,split_target=split_data(n,d[0:n-1])
    w_o=np.ones(m)
    epsilon=0.01 #error margin
    #Splitting data among agents at random
    
    k=0
    start_time=time.time()
    while np.linalg.norm(ds)>0.01 or np.linalg.norm(np.sum(dw, axis=0))>=0.01:
        # Updating agents strategies
        e, ds = zip(*[person[i].update_np(delta, np.sum(e), w) for i in range(n)])
        e = np.array(e)
        en=e/(np.sum(e))
        ds = np.array(ds)
        dw = np.array([person[i].gradient_acc_w_np(np.sum(e), w) for i in range(n)])
        w_o=w
        w += eta * np.sum(dw * en[:, np.newaxis], axis=0)
        if k % 500 == 0:
            print("Iter=", k)
            print(np.linalg.norm(np.sum(dw, axis=0)))
            print(np.linalg.norm(ds))
            print(e)
        if k>=5000:
            break
        k=k+1
    #Strategy updates are complete
    print("upbred without payments compelete",k)  
    end=time.time()-start_time
    #parameter updates are complete
    print("Training complete,number of iterations=",k)
    print("Time taken",end)
    a=np.zeros(n)
    u=np.zeros(n)
    sum=np.sum(e)
    a = np.array([p.get_accuracy(w, sum) for p in person])
    # data["utility_without payments"]=u
    # data["payment"]=p
    data["accuracy_np"]=a
    data["time_np"]=end
    data["iterations_np"]=k
    data["s_np"]=e

#####2phase#########
    person=[]
    for i in range(n):
        person.append(Agent(data["beta"][i],data["costs"][i],data["s_max"][i],data["Initial"][i],split_data[i],split_target[i],m,device))
    k=np.zeros(n)
    e=data["Initial"] 
    w=np.zeros(m)  #Initial model parameter values
    ds=np.ones(n)  
    dw=np.random.rand(n,m)
    u=np.zeros(n)
    delta=10  #stepsize for strategy updates
    eta=0.5   #stepsize for parameter updates
    #split_data,split_target=split_data(n,d[0:n-1])
    w_o=np.ones(m)
    epsilon=0.01 #error margin
    ###Updating agent strategies
    k=0
    start_time=time.time()
    while np.linalg.norm(np.array(e)-np.array(d-d/10))!=0:
        # Updating agents strategies
        e, ds = zip(*[person[i].update(delta, np.sum(e),w_o, w) for i in range(n)])
        e = np.array(e)
        en=e/(np.sum(e))
        ds = np.array(ds)
        if k % 500 == 0:
            print("Iter=", k)
            print(np.linalg.norm(np.sum(dw, axis=0)))
            print(np.linalg.norm(ds))
            print(e)
        k=k+1
    #Strategy updates are complete
    print("Strategy at optimal,number of iterations=",k)  
    #Updating parameters  
    while np.linalg.norm(np.sum(dw,axis=0))>0.01:
        dw = np.array([person[i].gradient_acc_w(np.sum(e), w) for i in range(n)])
        w_o=w
        w += eta * np.sum(dw * en[:, np.newaxis], axis=0)
        if k % 500 == 0:
            print("Iter=", k)
            print(np.linalg.norm(np.sum(dw, axis=0)))
        k=k+1
    end=time.time()-start_time
    #parameter updates are complete
    print("Training complete,number of iterations=",k)
    print("Time taken",end)
    a=np.zeros(n)
    u=np.zeros(n)
    sum=np.sum(e)
    a = np.array([p.get_accuracy(w, sum) for p in person])
    u = np.array([p.get_utility(w, sum, n) for p in person])
    p=np.zeros(n)
    p = np.array([p.get_payment(w,sum,n) for p in person])
    data["utility"]=u
    data["payment"]=p
    data["accuracy"]=a
    data["time"]=end
    data["iterations"]=k
    data["s"]=e
    ##Vanilla FedAvg###
    delta=0.5  #stepsize for strategy updates
    eta=0.5
    person=[]
    w=np.zeros(m)
    dw=np.random.rand(n,m)
    for i in range(n):
        person.append(Agent(data["beta"][i],data["costs"][i],data["s_max"][i],data["Initial"][i],split_data[i],split_target[i],m,device))
    k=0
    e=data["Initial"] 
    start=time.time()
    while np.linalg.norm(np.sum(dw,axis=0))>0.01:
        en=e/(np.sum(e))
        dw = np.array([person[i].gradient_acc_w_FedAvg(np.sum(e), w) for i in range(n)])
        w += eta * np.sum(dw * en[:, np.newaxis], axis=0)
        if k % 500 == 0:
            print("Iter=", k)
            print(np.linalg.norm(np.sum(dw, axis=0)))
        k=k+1
    end=time.time()-start
    print("FedAvg complete")
    print("Time taken",end,k)
    data["FedAvg_time"]=end
    a=np.zeros(n)
    u=np.zeros(n)
    sum=np.sum(e)
    a = np.array([p.get_accuracy(w, sum) for p in person])
    u = np.array([p.get_utility(w, sum, n) for p in person])
    data["utility_FedAvg"]=u
    data["accuracy_FedAvg"]=a


    ##FedAvg fixed dataset at equilibrium
    delta=0.5  #stepsize for strategy updates
    eta=0.5
    person=[]
    w=np.zeros(m)
    dw=np.random.rand(n,m)
    data["s_np"]=[10]*n
    for i in range(n):
        person.append(Agent(data["beta"][i],data["costs"][i],data["s_np"][i],data["s_np"][i],split_data[i],split_target[i],m,device))
    k=0
    e=data["s_np"]
    print(e) 
    start=time.time()
    while np.linalg.norm(np.sum(dw,axis=0))>0.01:
        en=e/(np.sum(e))
        dw = np.array([person[i].gradient_acc_w_FedAvg(np.sum(e), w) for i in range(n)])
        w += eta * np.sum(dw * en[:, np.newaxis], axis=0)
        if k % 500 == 0:
            print("Iter=", k)
            print(np.linalg.norm(np.sum(dw, axis=0)))
        k=k+1
    end=time.time()-start
    print("FedAvg complete_equilibrium")
    print("Time taken",end,k)
    data["FedAvg_time_eq"]=end
    a=np.zeros(n)
    u=np.zeros(n)
    sum=np.sum(e)
    a = np.array([p.get_accuracy(w, sum) for p in person])
    u = np.array([p.get_utility(w, sum, n) for p in person])
    data["utility_FedAvg_eq"]=u
    data["accuracy_FedAvg_eq"]=a
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    return data


if __name__ == '__main__':
    x=random.choices(range(1,10),k=10)
    with Pool() as p:
        output = p.map(worker, x)
    with open('Agent_60_2phase_cuda_parallel_200.json', 'w') as json_file:
        json.dump(output, json_file)
        print("done")