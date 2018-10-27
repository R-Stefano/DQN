import numpy as np

class replayMemory():
    def __init__(self, sizeMemory, sampleSize, image_height, image_width, num_frames):
        self.memory=[]
        self.memorySize=sizeMemory #How many transitions store
        self.sampleSize=sampleSize #How many transitions (s,a,r,s1) pick. The minibatch size
        self.imageWidth=image_width
        self.imageHeight=image_height
        self.frames=num_frames

    def addExperience(self,s,a,r,d,s1):
        #If the replay memory is full, delete the first saved transition
        if (len(self.memory)>self.memorySize):
            del self.memory[0]
        
        #anchor for improving convergence
        if r<0:
            d=True

        #Convert d True/False to 1/0
        if d:
            d=1
        else:
            d=0

        #Add the transition to the list of transitions. The replay memory.
        self.memory.append([s,a,r,d,s1])

    def sampleExperience(self):
        #Generate randomly 32 integers between 0 and N_elements in the replay memory
        idxs=[np.random.randint(0,len(self.memory)) for i in range(self.sampleSize)]

        s=np.zeros((self.sampleSize, self.imageHeight, self.imageWidth, self.frames))
        a=[]
        r=[]
        d=[]
        s1=np.zeros((self.sampleSize, self.imageHeight, self.imageWidth, self.frames))
        
        #Retrieve the transitions from the replay memory
        for idx, i in enumerate(idxs):
            s[idx]=self.memory[i][0]/255.
            a.append(self.memory[i][1])
            r.append(self.memory[i][2])
            d.append(self.memory[i][3])
            s1[idx]=self.memory[i][4]/255.

        return [s,a,r,d,s1]
