import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import math
import pandas as pd
from scipy.stats import norm,qmc
seed = 1000
def EuropeanOption(S,σ,r,q,T,K,ot):
    d1=(np.log(S/K)+(r-q)*T)/(σ*(np.sqrt(T)))+1/2*σ*np.sqrt(T)
    d2=(np.log(S/K)+(r-q)*T)/(σ*(np.sqrt(T)))-1/2*σ*np.sqrt(T)
    if ot=='call':
        C= S*norm.cdf(d1,loc=0,scale=1)*np.exp(-q*T)-K*np.exp(-r*T)*norm.cdf(d2,loc=0,scale=1)
        return C
    if ot=='put':
        P= K*np.exp(-r*T)*norm.cdf(-d2,loc=0,scale=1)-S*norm.cdf(-d1,loc=0,scale=1)*np.exp(-q*T)
        return P
def deviCP(S,K,T,σ,r,q):
    d1=(np.log(S/K)+(r-q)*T)/(σ*(np.sqrt(T)))+1/2*σ*np.sqrt(T)
    devi=S*np.exp(-q*T)*np.sqrt(T)*norm.pdf(d1)
    return devi
def find_vol_newton(S,r,q,T,K,optionValue,call_put):
    price=0
    if call_put=='call':
        price = EuropeanOption(S,0.0001,r,q,T,K,'call')
    else:
        price = EuropeanOption(S,0.0001,r,q,T,K,'put')
    if price-optionValue>1.0e-5:
        return float('nan')
    σ0=np.sqrt(2*np.abs(np.log(S/K)+(r-q)*T)/T)
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    σ=σ0
    for i in range(MAX_ITERATIONS):
        if call_put=='call':
            price = EuropeanOption(S,σ,r,q,T,K,'call')
        else:
            price = EuropeanOption(S,σ,r,q,T,K,'put')
        vega = deviCP(S,K,T,σ,r,q)
        diff = optionValue - price  # f(x)
        #print(i, σ, diff)
        if (abs(diff) < PRECISION):
            return σ
        σ = σ + diff/vega # f(x) / f'(x)
    return σ
def find_vol_bisection(S,r,q,T,K,optionValue,call_put):
    σMax=10
    σMin=0.0001
    σMid=(σMax+σMin)/2
    if call_put=='call':
        while abs(EuropeanOption(S,σMid,r,q,T,K,'call')-optionValue)>1.0e-5:
            if EuropeanOption(S,σMin,r,q,T,K,'call')<optionValue<EuropeanOption(S,σMid,r,q,T,K,'call'):
                σMax=σMid
                σMid=(σMax+σMin)/2
            elif EuropeanOption(S,σMax,r,q,T,K,'call')>optionValue>EuropeanOption(S,σMid,r,q,T,K,'call'):
                σMin=σMid
                σMid=(σMax+σMin)/2
            else:
                return float('nan')
    else:
        while abs(EuropeanOption(S,σMid,r,q,T,K,'put')-optionValue)>1.0e-5:
            if EuropeanOption(S,σMin,r,q,T,K,'put')<optionValue<EuropeanOption(S,σMid,r,q,T,K,'put'):
                σMax=σMid
                σMid=(σMax+σMin)/2
            elif EuropeanOption(S,σMax,r,q,T,K,'put')>optionValue>EuropeanOption(S,σMid,r,q,T,K,'put'):
                σMin=σMid
                σMid=(σMax+σMin)/2
            else:
                return float('nan')
    return σMid
def GeometricAsianOption(S,σ,r,T,K,n,ot):
    σNew=σ*np.sqrt((n+1)*(2*n+1)/(6*n*n))
    u=(r-1/2*σ*σ)*(n+1)/(2*n)+1/2*σNew*σNew
    d1new=(np.log(S/K)+(u+1/2*σNew*σNew)*T)/(σNew*(np.sqrt(T)))
    d2new=d1new-σNew*(np.sqrt(T))
    if ot=='call':
        C=np.exp(-r*T)*(norm.cdf(d1new,loc=0,scale=1)*S*np.exp(u*T)-K*norm.cdf(d2new,loc=0,scale=1))
        return C
    if ot=='put':
        P=np.exp(-r*T)*(K*norm.cdf(-d2new,loc=0,scale=1)-norm.cdf(-d1new,loc=0,scale=1)*S*np.exp(u*T))
        return P
def GeometricBasketOption(S1,S2,σ1,σ2,r,T,K,p,ot):
    σBg=np.sqrt(σ1*σ1+σ2*σ2+σ1*σ2*p+σ2*σ1*p)/2
    uBg=r-1/2*(σ1*σ1+σ2*σ2)/2+1/2*σBg*σBg
    Bg0=(S1*S2)**(1/2)
    d1=(np.log(Bg0/K)+(uBg+1/2*σBg*σBg)*T)/(σBg*np.sqrt(T))
    d2=d1-σBg*np.sqrt(T)
    if ot=='call':
        C=np.exp(-r*T)*(norm.cdf(d1,loc=0,scale=1)*Bg0*np.exp(uBg*T)-K*norm.cdf(d2,loc=0,scale=1))
        return C
    if ot=='put':
        P=np.exp(-r*T)*(K*norm.cdf(-d2,loc=0,scale=1)-norm.cdf(-d1,loc=0,scale=1)*Bg0*np.exp(uBg*T))
        return P
def GeometricBasketOptionExtension(sList,σList,r,T,K,pMatrix,ot):
    n=len(sList)
    tmp=0
    for i in range(n):
        for j in range(n):
            tmp+=σList[i]*σList[j]*pMatrix[i][j]
    σBg=np.sqrt(tmp)/n
    uBg=r-1/2*sum(np.square(σList))/n+1/2*σBg*σBg
    tmp=1
    for i in range(n):
        tmp*=sList[i]
    Bg0=(tmp)**(1/n)
    d1=(np.log(Bg0/K)+(uBg+1/2*σBg*σBg)*T)/(σBg*np.sqrt(T))
    d2=d1-σBg*np.sqrt(T)
    if ot=='call':
        C=np.exp(-r*T)*(norm.cdf(d1,loc=0,scale=1)*Bg0*np.exp(uBg*T)-K*norm.cdf(d2,loc=0,scale=1))
        return C
    if ot=='put':
        P=np.exp(-r*T)*(K*norm.cdf(-d2,loc=0,scale=1)-norm.cdf(-d1,loc=0,scale=1)*Bg0*np.exp(uBg*T))
        return P
def ArithmeticAsianOption(S,σ,r,T,K,n,ot,M,controlMethod):
    np.random.seed(seed)
    X=np.random.standard_normal((M,n))
    arithPayoff=np.zeros(M)
    geoPayoff=np.zeros(M)
    geo = GeometricAsianOption(S,σ,r,T,K,n,ot)
    drift = np.exp((r-0.5*σ**2)*(T/n))
    for i in range(M):
        growthFactor=drift*np.exp(σ*np.sqrt(T/n)*X[i][0])
        Spath=np.zeros(n)
        Spath[0]=S*growthFactor
        for j in range(1,n):
            growthFactor=drift*np.exp(σ*np.sqrt(T/n)*X[i][j])
            Spath[j]=Spath[j-1]*growthFactor
        arithMean = np.mean(Spath)
        geoMean = np.exp((1/n)*sum(np.log(Spath)))
        arithPayoffTmp=0
        geoPayoffTmp=0
        if ot=='call':
            arithPayoffTmp=max(arithMean-K,0)
            geoPayoffTmp=max(geoMean-K,0)
        else:
            arithPayoffTmp=max(K-arithMean,0)
            geoPayoffTmp=max(K-geoMean,0)
        arithPayoff[i] = np.exp(-r*T)*arithPayoffTmp
        geoPayoff[i] = np.exp(-r*T)*geoPayoffTmp
    if controlMethod=='none':
        Pmean = np.mean(arithPayoff);
        Pstd = np.std(arithPayoff);
        confmc = [Pmean-1.96*Pstd/np.sqrt(M), Pmean+1.96*Pstd/np.sqrt(M)]
        return (Pmean,Pstd,confmc)
    elif controlMethod=='control':
        #control variate
        covXY = np.mean(arithPayoff * geoPayoff)- np.mean(arithPayoff) * np.mean(geoPayoff);
        theta= covXY/np.var(geoPayoff);
        Z = arithPayoff + theta * (geo - geoPayoff);
        Zmean = np.mean(Z);
        Zstd = np.std(Z);
        confcv = [Zmean-1.96*Zstd/np.sqrt(M), Zmean+1.96*Zstd/np.sqrt(M)]
        return (Zmean,Zstd,confcv)
def ArithmeticBasketOption(S1,S2,σ1,σ2,r,T,K,p,ot,M,controlMethod):
    drift1 = np.exp((r-0.5*σ1**2)*(T))
    drift2 = np.exp((r-0.5*σ2**2)*(T))
    np.random.seed(seed)
    X=np.random.standard_normal(M)
    Y=np.random.standard_normal(M)
    arithPayoff=np.zeros(M)
    geoPayoff=np.zeros(M)
    geo = GeometricBasketOption(S1,S2,σ1,σ2,r,T,K,p,ot)
    for i in range(M):
        growthFactorX=drift1*np.exp(σ1*np.sqrt(T)*X[i])
        growthFactorY=drift2*np.exp(σ2*np.sqrt(T)*Y[i])
        Stx=S1*growthFactorX
        Sty=S2*growthFactorY
        arithMean = (Stx+Sty)/2
        geoMean = (Stx*Sty)**(1/2)
        arithPayoffTmp=0
        geoPayoffTmp=0
        if ot=='call':
            arithPayoffTmp=max(arithMean-K,0)
            geoPayoffTmp=max(geoMean-K,0)
        else:
            arithPayoffTmp=max(K-arithMean,0)
            geoPayoffTmp=max(K-geoMean,0)
        arithPayoff[i] = np.exp(-r*T)*arithPayoffTmp
        geoPayoff[i] = np.exp(-r*T)*geoPayoffTmp
    if controlMethod=='none':
        Pmean = np.mean(arithPayoff);
        Pstd = np.std(arithPayoff);
        confmc = [Pmean-1.96*Pstd/np.sqrt(M), Pmean+1.96*Pstd/np.sqrt(M)]
        return (Pmean,Pstd,confmc)
    elif controlMethod=='control':
        #control variate
        covXY = np.mean(arithPayoff * geoPayoff)- np.mean(arithPayoff) * np.mean(geoPayoff);
        theta= covXY/np.var(geoPayoff);
        Z = arithPayoff + theta * (geo - geoPayoff);
        Zmean = np.mean(Z);
        Zstd = np.std(Z);
        confcv = [Zmean-1.96*Zstd/np.sqrt(M), Zmean+1.96*Zstd/np.sqrt(M)]
        return (Zmean,Zstd,confcv)
def ArithmeticBasketOptionExtension(SList,σList,r,T,K,pMatrix,ot,M,controlMethod):
    n=len(SList)
    driftList=np.ones(n)
    for i in range(n):
        driftList[i]=np.exp((r-0.5*σList[i]**2)*(T))
    np.random.seed(seed)
    X=np.random.standard_normal((M,n))
    arithPayoff=np.zeros(M)
    geoPayoff=np.zeros(M)
    geo = GeometricBasketOptionExtension(SList,σList,r,T,K,pMatrix,ot)
    for i in range(M):
        StList=np.ones(n)
        for j in range(n):
            growthFactorj=driftList[j]*np.exp(σList[j]*np.sqrt(T)*X[i][j])
            StList[j]=SList[j]*growthFactorj
        arithMean = np.mean(StList)
        geoMean = np.exp((1/n)*sum(np.log(StList)))
        arithPayoffTmp=0
        geoPayoffTmp=0
        if ot=='call':
            arithPayoffTmp=max(arithMean-K,0)
            geoPayoffTmp=max(geoMean-K,0)
        else:
            arithPayoffTmp=max(K-arithMean,0)
            geoPayoffTmp=max(K-geoMean,0)
        arithPayoff[i] = np.exp(-r*T)*arithPayoffTmp
        geoPayoff[i] = np.exp(-r*T)*geoPayoffTmp
    if controlMethod=='none':
        Pmean = np.mean(arithPayoff);
        Pstd = np.std(arithPayoff);
        confmc = [Pmean-1.96*Pstd/np.sqrt(M), Pmean+1.96*Pstd/np.sqrt(M)]
        return (Pmean,Pstd,confmc)
    elif controlMethod=='control':
        #control variate
        covXY = np.mean(arithPayoff * geoPayoff)- np.mean(arithPayoff) * np.mean(geoPayoff);
        theta= covXY/np.var(geoPayoff);
        Z = arithPayoff + theta * (geo - geoPayoff);
        Zmean = np.mean(Z);
        Zstd = np.std(Z);
        confcv = [Zmean-1.96*Zstd/np.sqrt(M), Zmean+1.96*Zstd/np.sqrt(M)]
        return (Zmean,Zstd,confcv)
def AmericanOption(S,σ,r,T,K,N,ot):
    dt=T/N
    u=np.exp(σ*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    St = np.zeros((N+1,N+1))
    St[0,0] = S
    for i in range(1,N+1):  #模拟每个节点的价格
        for a in range(i):
            St[a,i] = St[a,i-1] * u
            St[a+1,i] = St[a,i-1] * d
    Sv = np.zeros_like(St)
    if ot == "call":    
        S_intrinsic = np.maximum(St-K,0)
    else:
        S_intrinsic = np.maximum(K-St,0)
    Sv[:,-1] = S_intrinsic[:,-1]
    for i in range(N-1,-1,-1): #反向倒推每个节点的价值
        for a in range(i+1):        
            Sv[a,i] = max((Sv[a,i+1] * p + Sv[a+1,i+1] * (1-p))/np.exp(r*dt),S_intrinsic[a,i])
    return Sv[0,0]
def KIKOputOption(S, σ, r, T, K, L, U, N, M,R):
    dt=T/N
    values = []
    sequencer = qmc.Sobol(d=N, seed=seed)
    # uniform samples
    X = np.array(sequencer.random(n=M))
    # standard normal samples
    Z = norm.ppf(X)
    samples = (r - 0.5 * σ * σ) * dt + σ * np.sqrt(dt) * Z
    df_samples = pd.DataFrame(samples)
    df_samples_cumsum = df_samples.cumsum(axis=1)
    df_stocks = S * np.exp(df_samples_cumsum)
    for ipath in df_stocks.index.to_list():
        ds_path_local = df_stocks.loc[ipath, :]
        price_max = ds_path_local.max()
        price_min = ds_path_local.min()
        if price_max >= U: # knock-out happened
            knockout_time = ds_path_local[ds_path_local>= U].index.to_list()[0]
            payoff = R * np.exp(-knockout_time * r * dt)
            values.append(payoff)
        elif price_min <= L: # knock-in happend
            final_price = ds_path_local.iloc[-1]
            payoff = np.exp(- r * T) * max(K - final_price, 0)
            values.append(payoff)
        else: # no knock-out, no knock-in
            values.append(0)
    value = np.mean(values)
    std = np.std(values)
    conf_interval_lower = value - 1.96 * std / np.sqrt(M)
    conf_interval_upper = value + 1.96 * std / np.sqrt(M)
    confcv = [conf_interval_lower, conf_interval_upper]
    return (value,confcv)

#class BasketValueManager():
 #   def __init__(self):

class OptionLayoutManager():
    def __init__(self):
        qf=QFormLayout()
        self.qf=qf
        self.index=0
        #e1.setMaxLength(4)
        #e1.setAlignment(Qt.AlignRight)
        #e1.setFont(QFont("Arial",20))
        ocb = QComboBox()
        titleList=['Call','Put']
        ocb.addItems(titleList)
        ocb.currentIndexChanged[int].connect(self.changeOptionType)
        
        eS = QLineEdit()
        eS.setValidator(QDoubleValidator(0,1.0,10))
        eσ = QLineEdit()
        eσ.setValidator(QDoubleValidator(0,1.0,10))
        er = QLineEdit()
        er.setValidator(QDoubleValidator(0,1.0,10))
        eq = QLineEdit()
        eq.setValidator(QDoubleValidator(0,1.0,10))
        eT = QLineEdit()
        eT.setValidator(QDoubleValidator(0,1.0,10))
        eK = QLineEdit()
        eK.setValidator(QDoubleValidator(0,1.0,10))
        eL = QLineEdit()
        eL.setValidator(QDoubleValidator(0,1.0,10))
        eU = QLineEdit()
        eU.setValidator(QDoubleValidator(0,1.0,10))
        eR = QLineEdit()
        eR.setValidator(QDoubleValidator(0,1.0,10))
        
        en = QLineEdit()
        en.setValidator(QIntValidator(1,2147483647))
        epath = QLineEdit()
        epath.setValidator(QIntValidator(1,2147483647))
        emethod = QCheckBox("Geometric Option Value")
        
        ess=QLineEdit()
        evs=QLineEdit()
        eps=QPlainTextEdit()

        self.optionType='call'
        self.ocb=ocb
        self.eS=eS
        self.eσ=eσ
        self.er=er
        self.eq=eq
        self.eT=eT
        self.eK=eK
        self.eL=eL
        self.eU=eU
        self.eR=eR
        self.en=en
        self.epath=epath
        self.emethod=emethod
        self.ess=ess
        self.evs=evs
        self.eps=eps
        
        ocb.hide()
        en.hide()
        epath.hide()
        emethod.hide()
        eL.hide()
        eU.hide()
        eR.hide()
        ess.hide()
        evs.hide()
        eps.hide()
        
        qf.addRow("Option type",ocb)
        qf.addRow("Spot price", eS)
        qf.addRow("Spot price list(separated by space)",ess)   
        qf.addRow("Volatility",eσ)
        qf.addRow("Volatility list(separated by space)",evs)
        qf.addRow("Correlation matrix(separated by space)",eps)
        qf.addRow("Risk-free interest rate",er)
        qf.addRow("Repo rate",eq)
        qf.addRow("Time to maturity (in years)",eT)
        qf.addRow("Strike",eK)
        qf.addRow("Lower barrier",eL)
        qf.addRow("Upper barrier",eU)
        qf.addRow("Cash rebate",eR)
        qf.addRow("Observation times or number of steps",en)
        qf.addRow("Number of paths",epath)
        qf.addRow("Control variate",emethod)
        
        button=QPushButton("Calculate")
        button.clicked.connect(self.calculate)
        qf.addRow("",button)

        #e4.textChanged.connect(textchanged)

        eCall = QLineEdit("")
        eCall.setReadOnly(True)
        self.eCall=eCall
        qf.addRow("Call price",eCall)
        
        eCallCfi = QLineEdit("")
        eCallCfi.setReadOnly(True)
        self.eCallCfi=eCallCfi
        qf.addRow("95% confidence interval",eCallCfi)
        eCallCfi.hide()
        
        ePut = QLineEdit("")
        ePut.setReadOnly(True)
        self.ePut=ePut
        qf.addRow("Put price",ePut)
        
        ePutCfi = QLineEdit("")
        ePutCfi.setReadOnly(True)
        self.ePutCfi=ePutCfi
        qf.addRow("95% confidence interval",ePutCfi)
        ePutCfi.hide()
        
        eIV = QLineEdit("")
        eIV.setReadOnly(True)
        self.eIV=eIV
        qf.addRow("Implied volatility",eIV)
        eIV.hide()
        
        
    def changeOptionType(self, i):
        if i==1:
            self.optionType='put'
            self.ePut.show()
            self.eCall.hide()
        else:
            self.optionType='call'
            self.ePut.hide()
            self.eCall.show()
            
    def changeStrToFloatList(self,str):
        str=str.split()
        str= list(map(float, str))
        return str
    def changeStrToMatrix(self,str):
        str=str.split()
        strList= list(map(float, str))
        n2=len(strList)
        n=int(np.sqrt(n2))
        arr=np.array(strList)
        arr=arr.reshape(n,n)
        return arr
    def calculate(self):
        try:
            if self.index==0:
                S=float(self.eS.text())
                σ=float(self.eσ.text())
                r=float(self.er.text())
                q=float(self.eq.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                callPrice=EuropeanOption(S,σ,r,q,T,K,'call')
                putPrice=EuropeanOption(S,σ,r,q,T,K,'put')
                self.eCall.setText(str(callPrice))
                self.ePut.setText(str(putPrice))
            if self.index==1:
                S=float(self.eS.text())
                r=float(self.er.text())
                q=float(self.eq.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                if self.optionType=='call':
                    optionValue=float(self.eCall.text())
                    IV=find_vol_newton(S,r,q,T,K,optionValue,'call')
                    self.eIV.setText(str(IV))
                else:
                    optionValue=float(self.ePut.text())
                    IV=find_vol_newton(S,r,q,T,K,optionValue,'put')
                    self.eIV.setText(str(IV))
            if self.index==2:
                S=float(self.eS.text())
                σ=float(self.eσ.text())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                n=int(self.en.text())
                callPrice=GeometricAsianOption(S,σ,r,T,K,n,'call')
                putPrice=GeometricAsianOption(S,σ,r,T,K,n,'put')
                self.eCall.setText(str(callPrice))
                self.ePut.setText(str(putPrice))
            if self.index==3:
                S=float(self.eS.text())
                σ=float(self.eσ.text())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                n=int(self.en.text())
                M=int(self.epath.text())
                controlMethod="none"
                if self.emethod.isChecked():
                    controlMethod="control"
                callValue=ArithmeticAsianOption(S,σ,r,T,K,n,'call',M,controlMethod)
                putValue=ArithmeticAsianOption(S,σ,r,T,K,n,'put',M,controlMethod)
                self.eCall.setText(str(callValue[0]))
                self.ePut.setText(str(putValue[0]))
                self.eCallCfi.setText(str(callValue[2]))
                self.ePutCfi.setText(str(putValue[2]))
            if self.index==4:
                S=float(self.eS.text())
                σ=float(self.eσ.text())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                n=int(self.en.text())
                callPrice=AmericanOption(S,σ,r,T,K,n,'call')
                putPrice=AmericanOption(S,σ,r,T,K,n,'put')
                self.eCall.setText(str(callPrice))
                self.ePut.setText(str(putPrice))
            if self.index==5:
                S=float(self.eS.text())
                σ=float(self.eσ.text())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                L=float(self.eL.text())
                U=float(self.eU.text())
                R=float(self.eR.text())
                n=int(self.en.text())
                M=int(self.epath.text())
                putPrice=KIKOputOption(S, σ, r, T, K, L, U, n, M,R)
                self.ePut.setText(str(putPrice[0]))
                self.ePutCfi.setText(str(putPrice[1]))
            if self.index==6:
                SList=self.changeStrToFloatList(self.ess.text())
                σList=self.changeStrToFloatList(self.evs.text())
                pMatrix=self.changeStrToMatrix(self.eps.toPlainText())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                callPrice=GeometricBasketOptionExtension(SList,σList,r,T,K,pMatrix,'call')
                putPrice=GeometricBasketOptionExtension(SList,σList,r,T,K,pMatrix,'put')
                self.eCall.setText(str(callPrice))
                self.ePut.setText(str(putPrice))
            if self.index==7:
                SList=self.changeStrToFloatList(self.ess.text())
                σList=self.changeStrToFloatList(self.evs.text())
                pMatrix=self.changeStrToMatrix(self.eps.toPlainText())
                r=float(self.er.text())
                T=float(self.eT.text())
                K=float(self.eK.text())
                M=int(self.epath.text())
                controlMethod="none"
                if self.emethod.isChecked():
                    controlMethod="control"
                callPrice=ArithmeticBasketOptionExtension(SList,σList,r,T,K,pMatrix,'call',M,controlMethod)
                putPrice=ArithmeticBasketOptionExtension(SList,σList,r,T,K,pMatrix,'put',M,controlMethod)
                self.eCall.setText(str(callPrice[0]))
                self.ePut.setText(str(putPrice[0]))
                self.eCallCfi.setText(str(callPrice[2]))
                self.ePutCfi.setText(str(putPrice[2]))
        except Exception:
            print('异常')
    def changePage(self,index):
        self.index=index
        if index==0:
            self.ocb.hide()
            self.eS.show()
            self.eσ.show()
            self.er.show()
            self.eq.show()
            self.eT.show()
            self.eK.show()
            self.en.hide()
            self.epath.hide()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.hide()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==1:
            self.ocb.show()
            self.eS.show()
            self.eσ.hide()
            self.er.show()
            self.eq.show()
            self.eT.show()
            self.eK.show()
            self.en.hide()
            self.epath.hide()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.hide()
            self.eCall.hide()
            self.ePut.hide()
            self.eCall.setReadOnly(False)
            self.ePut.setReadOnly(False)
            if self.optionType=='call':
                self.eCall.show()
            else:
                self.ePut.show()
            self.eIV.show()
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==2:
            self.ocb.hide()
            self.eS.show()
            self.eσ.show()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.show()
            self.epath.hide()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.hide()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==3:
            self.ocb.hide()
            self.eS.show()
            self.eσ.show()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.show()
            self.epath.show()
            self.emethod.show()
            self.eCallCfi.show()
            self.ePutCfi.show()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==4:
            self.ocb.hide()
            self.eS.show()
            self.eσ.show()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.show()
            self.epath.hide()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.hide()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==5:
            self.ocb.hide()
            self.eS.show()
            self.eσ.show()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.show()
            self.epath.show()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.show()
            self.eIV.hide()
            self.eCall.hide()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.show()
            self.eU.show()
            self.eR.show()
            self.ess.hide()
            self.evs.hide()
            self.eps.hide()
        if index==6:
            self.ocb.hide()
            self.eS.hide()
            self.eσ.hide()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.hide()
            self.epath.hide()
            self.emethod.hide()
            self.eCallCfi.hide()
            self.ePutCfi.hide()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.show()
            self.evs.show()
            self.eps.show()
        if index==7:
            self.ocb.hide()
            self.eS.hide()
            self.eσ.hide()
            self.er.show()
            self.eq.hide()
            self.eT.show()
            self.eK.show()
            self.en.hide()
            self.epath.show()
            self.emethod.show()
            self.eCallCfi.show()
            self.ePutCfi.show()
            self.eIV.hide()
            self.eCall.show()
            self.eCall.setReadOnly(True)
            self.ePut.show()
            self.ePut.setReadOnly(True)
            self.eL.hide()
            self.eU.hide()
            self.eR.hide()
            self.ess.show()
            self.evs.show()
            self.eps.show()
class MyWindow():
    def __init__(self):
        app = QApplication(sys.argv)
        window=QWidget()
        # 设置标题
        window.setWindowTitle('ComBox例子')
        # 设置初始界面大小
        window.resize(800, 800)
        #self.move(300, 300)    #移动到300，300这个桌面的坐标上
        window.setWindowTitle('OptionCalculator') #设置窗口的标题    
        
        # 实例化QComBox对象
        cb = QComboBox()
        titleList=['European Option','Implied Volatility','Geometric Asian Option', 'Arithmetic Asian Option','American Option','KIKO-Put Option','Geometric Basket Option','Arithmetic Basket Option']
        cb.addItems(titleList)

        # 信号
        #cb.currentIndexChanged[str].connect(self.print_value) # 条目发生改变，发射信号，传递条目内容
        cb.currentIndexChanged[int].connect(self.changePage)  # 条目发生改变，发射信号，传递条目索引
        #cb.highlighted[str].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目内容
        #cb.highlighted[int].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目索引

        vbox = QVBoxLayout()
        vbox.addWidget(cb)
        optionLayoutManager=OptionLayoutManager()
        vbox.addLayout(optionLayoutManager.qf)
        self.vbox=vbox
        self.optionLayoutManager=optionLayoutManager
        
        window.setLayout(vbox)
        self.curIndex=0
        self.titleList=titleList
        self.window=window
        window.show()
        sys.exit(app.exec_())

    def changePage(self, i):
        self.optionLayoutManager.changePage(i)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()