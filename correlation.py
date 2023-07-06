import cv2
import numpy as np
import matplotlib.pyplot as plt
vid = cv2.VideoCapture("dharneesh_video.mp4")
temp=cv2.imread("sat.png") 
print(temp.size)
temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY) 

#old one.
# print(temp.shape)
#inital : 67,65
#58,46
temp=cv2.resize(temp,(58,46))

print(temp.shape)

# print(temp.shape)

x=0.05

while(True):
    ret, frame = vid.read()
    print(frame.shape)
    
    #old frame
    #576,736.
    # frame=cv2.resize(frame,(256,256))
    frame=cv2.resize(frame,(512,512))
    if frame is None:
        break
    fr=frame
    fr=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    S=fr.shape
    F=temp.shape
    R=S[0]+F[0]-1
    C=S[1]+F[1]-1
    Z=np.zeros((R,C))
    for i in range(S[0]):
        for j in range(S[1]):
            Z[i+np.int32((F[0]-1)/2),j+np.int32((F[1]-1)/2)]=fr[i,j]
            
    p1=int(0)
    p2=int(0)
    mini=10000000000
    for i in range(200,S[0]-200):
        for j in range(200,S[1]-200):
            k1=Z[i:i+F[0],j:j+F[1]]
            l=np.sum(np.multiply(k1,temp))
            m=np.sum(np.multiply(k1,k1))
            n=np.sum(np.multiply(temp,temp))
            if(m+n-2*l<mini):
                p1=i
                p2=j
                mini=m+n-2*l
                gcur=k1
    print(p1," ",p2,mini)
    if(x<0.25):
        x+=(1/1000)
    temp=np.add((1-x)*temp,x*gcur)
    cv2.rectangle(frame,(p1-10,p2-10),(p1+10,p2+10),(255,0,0),1)
    cv2.imshow('frame', frame)


      
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()