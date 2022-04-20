import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs

    

def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput):
    
    
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float("inf")
    
   
    Positions=numpy.random.randint(2, size=(SearchAgents_no,dim)) 
    
    Convergence_curve1=numpy.zeros(Max_iter)
    Convergence_curve2=numpy.zeros(Max_iter)

    s=solution()

     # Loop counter
    print("\nGWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)
                  
            while numpy.sum(Positions[i,:])==0:   
                 Positions[i,:]=numpy.random.randint(2, size=(1,dim))

            fitness=objf(Positions[i,:],trainInput,trainOutput,dim)
            
            if fitness<Alpha_score :
                Alpha_score=fitness;
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter);
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() 
                r2=random.random() 
                
                A1=2*a*r1-a; 
                C1=2*r2; 
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); 
                temp=transfer_functions_benchmark.s1(A1*D_alpha)
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                if (Alpha_pos[j]+temp)>=1:
                    X1=Alpha_pos[j]+temp
                
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a;
                C2=2*r2;
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); 
                temp=transfer_functions_benchmark.s1(A2*D_beta)
                
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                    
                if (Beta_pos[j]+temp)>=1:
                    X2=Beta_pos[j]+temp
                
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a;
                C3=2*r2;
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]);  
                
                temp=transfer_functions_benchmark.s1(A3*D_delta)
                if temp<numpy.random.uniform(0,1):
                    temp=0
                else:
                    temp=1
                    
                if (Delta_pos[j]+temp)>=1:
                    X3=Delta_pos[j]+temp
                
            Positions[i,j]=(X1+X2+X3)/3
            
        featurecount=0
        for f in range(0,dim):
            if Alpha_pos[f]==1:
                featurecount=featurecount+1
                
        Convergence_curve1[l]=Alpha_score;
        Convergence_curve2[l]=featurecount;
        if (l%1==0):
                print(['At iteration'+ str(l+1)+' the best fitness on training is:'+ str(Alpha_score)+', the best number of features: '+str(featurecount)]);
    
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Alpha_pos
    s.convergence1=Convergence_curve1
    s.convergence2=Convergence_curve2

    s.optimizer="GWO"
    s.objfname=objf.__name__
    return s
    

