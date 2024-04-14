import numpy as np
from scipy import stats

######################################################

def WeibullMinFunction(x):
    s1=0;s2=0;b=x[0];
    c=(sum(y**b))/m
    for i in range(n):
          z=y[i]**b
          s2+=z*np.log(z)
          s1+=(1-r[i])*np.log(z)
    c1=m+s1-s2/c
    return(c1**2)

############################################################

def NormalMinFunction(x):
    z =(xx-x[0])/x[1]
    p = stats.norm.cdf(z);d = stats.norm.pdf(z)
    psi=d/(1-p);s1 =sum(psi);s2 =sum(psi*z)
    c1=cp-x[0]+x[1]*s1/m
    c2=cko**2+(cp-x[0])**2+x[1]**2*(s2/m-1)
    return(c1**2+c2**2)

##############################################################

def UpDownMinFunction(x):

    if  x[1]<=0 or x[1]>1.5*s: return(10000000)
    s1=0;s2=0;
    for i in range(klevel):
        z=(y[i]-x[0])/x[1]
        if ts=="Normal":
              p=stats.norm.cdf(z)
              d=stats.norm.pdf(z)
        if ts=="Weibull":
              p=1.-np.exp(-np.exp(z))
              d=np.exp(z-(np.exp(z)))
        if p<=0 or p>=1: return(10000000)
        fiz=(pp[i]-p)*nsample[i]*d/(p*(1.-p))
        s1=s1+fiz
        s2=s2+fiz*z
    return(s1*s1+s2*s2)

################"Инициализация симплекса########################

def init_simplex(x0):
    step=0.1
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0
    for i in range(n):
        direction = np.zeros(n)
        direction[i] = step
        simplex[i+1] = x0 + direction
    return simplex

########Оценка симплекса#######################################

def evaluate_simplex(simplex, func):
    return np.array([func(x) for x in simplex])

##########Отражение########################################

def reflect(xh, centroid):
    alpha=1.0
    return centroid + alpha * (centroid - xh)

##########"Растяжение#####################################

def expand(xr, centroid):
    gamma=2.0
    return centroid + gamma * (xr - centroid)

###########Сжатие#######################################

def contract(xh, centroid):
    rho=0.5
    return centroid + rho * (xh - centroid)

##########Уменьшение размера#############################

def shrink(simplex, xl):
    sigma=0.5
    n = len(simplex)
    new_simplex = np.zeros_like(simplex)
    new_simplex[0] = xl
    for i in range(1, n):
        new_simplex[i] = xl + sigma * (simplex[i] - xl)
    return new_simplex

##########Метод Нелдера-Мида для минимизации функции#############

def nelder_mead(func, x0, maxiter,tol):
  
    simplex = init_simplex(x0)
    counter = 0
    for _ in range(maxiter):
        counter += 1
        f_values = evaluate_simplex(simplex, func)
        order = f_values.argsort()
        simplex = simplex[order]
        f_values = f_values[order]

        if np.max(np.abs(f_values - f_values[0])) <= tol:break  # Условие остановки

        centroid = np.mean(simplex[:-1], axis=0)  # Центроид всех точек, кроме худшей
        xr = reflect(simplex[-1], centroid)
        fr = func(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
        elif fr < f_values[0]:
            xe = expand(xr, centroid)
            fe = func(xe)
            simplex[-1] = xe if fe < fr else xr
        else:
            xc = contract(simplex[-1], centroid)
            fc = func(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
            else:
                simplex = shrink(simplex, simplex[0])

    return simplex[0], func(simplex[0]), counter


##############MLE_Normal#############################################

global xx,y,m,cp,cko,n,r,pp,nsample,ts,klevel

finp=open("Inp/MLE_Normal.inp")
finp.readline()
n=int(finp.readline())
ss=finp.readline()
y=tuple(map(float,finp.readline().split(" ")))
ss=finp.readline()
r=tuple(map(int,finp.readline().split(" ")))
finp.close()
fout=open('Out/MLE_Normal.out','w')
n=len(y)
k=sum(r)
m=n-k

yy=tuple(map(float,(y[i] for i in range(n)  if(r[i]==0))))
xx=tuple(map(float,(y[i] for i in range(n)  if(r[i]==1))))

print("Sample size n=",n,file=fout)
print("Sample:",y,file=fout)

cp=np.average(yy)
cko=np.std(yy)
print("MO and Std by observed values",file=fout)
print("a0=",cp,file=fout)
print("s0=",cko,file=fout)

lim=1000
eps=1e-15

res,q,num_iters = nelder_mead(NormalMinFunction,[cp,cko],lim,eps)

print("MO and Std by MLE",file=fout)
print("a=",res[0],file=fout)
print("s=",res[1],file=fout)

print("q=",q,file=fout)
print("iter=",num_iters,file=fout)

fout.close()

###########MLE_Weibull#####################################


finp=open("Inp/MLE_Weibull.inp")
finp.readline()
n=int(finp.readline())
ss=finp.readline()
y=tuple(map(float,finp.readline().split(" ")))
ss=finp.readline()
r=tuple(map(int,finp.readline().split(" ")))
finp.close()

fout=open('Out/MLE_Weibull.out','w')
n=len(y)
k=sum(r)
m=n-k


yy=tuple(map(float,(y[i] for i in range(n)  if(r[i]==0))))
xx=tuple(map(float,(y[i] for i in range(n)  if(r[i]==1))))

print("Sample size n=",n,file=fout)
print("Sample:",y,file=fout)
print("Censorized sample size k=",k,file=fout)
print("Censorized values:",xx,file=fout)

b=0.5

lim=1000
eps=1e-15

res,q,num_iters = nelder_mead(WeibullMinFunction,[b],lim,eps)

b=res[0]
c=(sum(y**b))/m
a=(np.log(c))/b
s=1/b 
print("b and c by MLE",file=fout)
print("b=",b,file=fout)
print("c=",c,file=fout)
print("sw=1/b=",s,file=fout)
print("aw=lnc/b=",a,file=fout)
print("q=",q,file=fout)
print("iter=",num_iters,file=fout)

fout.close()

##################UpDown###############################

#ts="Normal"
ts="Weibull"

finp=open("Inp/UpDown_"+ts+".inp")
finp.readline()
klevel=int(finp.readline())
ss=finp.readline()
y=tuple(map(float,finp.readline().split(" ")))
ss=finp.readline()
nsample=tuple(map(int,finp.readline().split(" ")))
ss=finp.readline()
nfailure=tuple(map(int,finp.readline().split(" ")))
finp.close()

if ts == "Weibull": y=np.log(y)

#/Приближенный расчет оценок среднего и среднего квадратичного отклонения (cp,s)
#Dixon W. Т., Mood A. М. J. Amer. Statist. Ass., v. 43, 1948, p. 109.

nnon=list(0 for i in range(klevel))
pp=list(0. for i in range(klevel))

kfail = 0; knon = 0; ksigne = 1;
for  i in range(klevel):
    kfail=kfail+nfailure[i]
    nnon[i]=nsample[i]-nfailure[i]
    knon += nnon[i]

if  kfail < knon: ksigne = -1 #Расчет ведут по разрушенным образцам
s1 = 0; s2 = 0; s3 = 0;
for  i in range(klevel):
    if  ksigne== -1:
        s1+=i*nfailure[i]
        s2+=i*i*nfailure[i]
        s3+=nfailure[i]
    if  ksigne==1:  #Расчет ведут по не разрушенным образцам
        s1+=i*nnon[i]
        s2+=i*i*nnon[i]
        s3+=nnon[i]

d=y[1]-y[0]
a=y[0]+d*(s1/s3+ksigne*0.5) #Оценка среднего
s4=(s3*s2-s1*s1)/(s3*s3)
s=1.62*d*(s4+0.029) #Оценка ско

fout=open("Out/UpDown_"+ts+".out","w")

print("Sample:",y,file=fout)
print("n sample:",nsample,file=fout)
print("n failure:",nfailure,file=fout)
print("MO and Std by Approximation",file=fout)
print("a0=",a,file=fout)
print("s0=",s,file=fout)

nn = 0
for i in range(klevel):
    pp[i]=nfailure[i]/nsample[i]
    nn+=nsample[i]

print("Prob:",pp,file=fout)

lim=1000
eps=1e-15
cko=s
cp=a
xz=[a,s]
res,q,num_iters = nelder_mead(UpDownMinFunction,xz,lim,eps)

print("MO and Std by MLE",file=fout)
print("a=",res[0],file=fout)
print("s=",res[1],file=fout)

print("q=",q,file=fout)
print("iter=",num_iters,file=fout)

fout.close()
