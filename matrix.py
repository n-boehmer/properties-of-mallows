import numpy as np


def add_pows(phi,n):
  s=0
  phi_pow=1
  for i in range(n):
    s+=phi_pow
    phi_pow*=phi
  return s

# probability that candidate m is ranked at most i
def pos_at_most(m,i,phi):
    S=0
    for j in range(i):
      S+=phi**(m-1-j)   
    return S*1/add_pows(phi,m)#(1-phi)/(1-phi**(m))

def Z(phi,m):
  prod=1
  fac=0
  for i in range(m):
    fac+=phi**i
    prod*=fac
  return prod
  
  

def mallows_matrix(m,phi):
  P=np.ones((m,m))
  for n in range(2,m+1):
    A=np.ones((m,m))
    for i in range(n):
      x=add_pows(phi,n)
      A[0,i]=phi**(i)*1/x#(1-phi)/(1-phi**n)
      A[i,0]=A[0,i]
      A[n-i-1,n-1]=A[0,i]
      A[n-1,n-i-1]=A[0,i]
    for i in range(1,n-1):
      for j in range(1,n-1):
        A[i,j]=P[i,j]+P[i,j-1]*pos_at_most(n,j,phi)-P[i,j]*pos_at_most(n,j+1,phi)

    P=A
  return P

#print(mallows_matrix(260,0.5))
