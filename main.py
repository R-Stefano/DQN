import tensorflow as tf 
import numpy as np 
import gym
from gym import wrappers

#USED TO INCREASE RENDEERING WINDOW ON RETINA SCREENS
from gym.envs.classic_control import rendering

from replayMemory import replayMemory
from DQN import DQN

#USED TO INCREASE RENDEERING WINDOW ON RETINA SCREENS
viewer = rendering.SimpleImageViewer()

#SET UP ENVIRONMENT
game='PongNoFrameskip-v4'#'BreakoutNoFrameskip-v4'#'SpaceInvadersNoFrameskip-v4'#'FrostbiteNoFrameskip-v4'
#breakout:400, spaceInvaders:1976, frostbite:328 #maybe also some different game not atari?
env=gym.make(game)

#Name used to save the data
name="DQN_"+game

#SETUP LISTS FOR DEBUGGING
episodes_reward=[]
episodes_qvalue=[]
global_step=0
episode=0
bestResult=-1e5
lastResult=0

#HYPERPARAMETERS
state_size = env.observation_space.shape #return list 3 elements [width, height, channels(RGB or B/W)]
#print(env.unwrapped.get_action_meanings())
num_actions=env.action_space.n 

syncTarget=1000 #sync target network with prediction network every timesteps #PAPER:10000

#record every 50 episodes
env=wrappers.Monitor(env, "recording/"+name, video_callable=lambda episode_id: episode_id%50==0)

initializeReplayBuffer=10000#experience 10000 timesteps before start training #PAPER:50000
repBufferSize=200000 #PAPER: 1000000
sampleSize=32 #PAPER:32
startingEpsilon=1.0 #Starting epsilon #PAPER:1.0
endEpsilon=0.01 #PAPER: 0.1
epsilonDecay=10**5#Global steps required for decay epsilon from start to end #PAPER:10**6
learn_rate=0.0001#PAPER: 0.00025

factor=(endEpsilon - startingEpsilon)/epsilonDecay
frameskip=4


#Shape of the frame before feeding it
width=84    
height=84

#USED TO INCREASE RENDEERING WINDOW ON RETINA SCREENS
def upsample_rendering(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print ("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def startTraining():
    global episode
    global global_step
    global lastResult
    global bestResult
    print("\n\n FILLING REPLAY BUFFER... \n\n")
    initializationExperiences=0
    while(initializationExperiences<initializeReplayBuffer):
        env.reset()
        f,_,_,lives=env.step(1)#action fire to start the game
        #execute N no-op actions at the beginning
        for i in range(np.random.randint(0,10)):
            f,_,_,_=env.step(0)

        numberLives=lives['ale.lives']
        state=DQN.inputPreprocess(f)
        d=False

        while not(d):
            a=np.random.randint(0,num_actions)

            r=0
            for i in range(frameskip):
                f1, rew, d, lives=env.step(a)
                r += rew

                if d:
                    r=-1
                    break 
            
            if numberLives>lives['ale.lives']:
                r=-1
                numberLives=lives['ale.lives']

            #Reward clipping
            if r>0:
                r=1
            elif r<0:
                r=-1   
            
            newState=DQN.inputPreprocess(f1)

            memory.addExperience(state,a,r, d, newState)

            initializationExperiences+=1
            state=newState

    print("\n\n STARTING TRAINING.. \n\n") 
    while(lastResult<400):
        episode_reward=0
        episode_qvalues=[]

        #At the beginning of each episode reset the game
        env.reset()
        #Execute action: "FIRE" to start the game
        f,_,_,lives=env.step(1)
        #Keep track of the number of lives that the agent has in an episode
        numberLives=lives['ale.lives']
        #execute 10 no-op actions at the beginning of the game
        for i in range(np.random.randint(0,10)):
            f,_,_,_=env.step(0)

        #Empty the state representation
        DQN.resetObservationState()

        #Process the frame in order to output the state representation: [84,84,4]
        state=DQN.inputPreprocess(f)
        d=False

        while not(d):
            #every 50 episodes render the game
            if episode%50==0:
                rgb=env.render('rgb_array')
                upsample=upsample_rendering(rgb, 4, 4)
                viewer.imshow(upsample)

            #Feed the state to the model which selects the action to execute
            a, qvalue=DQN.actionSelection(state)
            
            #Repeat the action for frameskip frames
            r=0
            for i in range(frameskip):
                f1, rew, d, lives=env.step(a)
                r += rew

                if d:
                    r=-1
                    break 

            if numberLives>lives['ale.lives']:
                r=-1
                numberLives=lives['ale.lives']

            #Reward clipping
            if r>0:
                r=1
            elif r<0:
                r=-1

            newState=DQN.inputPreprocess(f1)
            
            #Add new transitions to replay memory
            memory.addExperience(state,a,r, d, newState)

            if(global_step%4==0):
                DQN.training(memory.sampleExperience())

            #epsilon annealing from 1 to 0.1 in 1000000 steps
            if global_step<=epsilonDecay:
                DQN.epsilon= (factor*global_step ) + startingEpsilon
            else:
                DQN.epsilon=endEpsilon

            #Update Target network every 10000 TRAINING STEPS (40000 steps)
            if (global_step%(4*syncTarget)==0 and global_step!=0):
                print("\n\nGlobal step", global_step, "Updating target network..\n\n")
                DQN.updateTargetNetwork()

            #Every 50k steps save filters values
            if global_step%50000==0:
                summ=DQN.sess.run(DQN.mergeFilters)

                DQN.writeOps.add_summary(summ, global_step=global_step)
            

            state=newState
            global_step +=1
            episode_reward+= r
            episode_qvalues.append(qvalue)

        #EPISODE ENDED
        print("\nEnded episode:", episode,"Global step:", global_step,"\n")

        #Store the averaged Q_value of the episode
        avgQVal=np.mean(episode_qvalues)
        #Store the total reward of the episode
        episodes_reward.append(episode_reward)
        #Compute the average total reward of the last 100 episodes
        lastResult=np.mean(episodes_reward[-100:])

        #Feed the averaged reward to the agent method in order to show the stats on tensorboard
        summ=DQN.sess.run(DQN.mergeEpisodeData, feed_dict={DQN.averagedReward:lastResult,
                                                           DQN.PHEpsilon:DQN.epsilon,
                                                           DQN.avgQValue : avgQVal})
        #Save the stats for tensorboard
        DQN.writeOps.add_summary(summ, global_step=episode)

        if lastResult>bestResult:
            print("\n")
            print("Saving model..")
            print("\n")
            DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode, rewards=episodes_reward[-100:])

            bestResult=lastResult

        episode+=1

        
    DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode, rewards=episodes_reward[-100:])

if __name__ == '__main__':
    with tf.Session() as sess:
        try:
            DQN=DQN(sess, num_actions=num_actions, num_frames=4, width=width, height=height, lr=learn_rate, startEpsilon=startingEpsilon, folderName=name)
            memory=replayMemory(sizeMemory=repBufferSize, sampleSize=sampleSize, image_height=height ,image_width=width, num_frames=4)

            res=input("Do you want to load the model? [y/n]")
            if res.lower()=="y":
                DQN.save_restore_Model(restore=True)
                episodes_reward=DQN.episode_Rewards.eval().tolist()
                global_step=DQN.global_step.eval()
                episode=DQN.episode.eval()

            startTraining()
        except (KeyboardInterrupt, SystemExit):
            print("Program shut down, saving the model..")
            DQN.save_restore_Model(restore=False, globa_step=global_step, episode=episode, rewards=episodes_reward[-100:])
            print("\n\nModel saved!\n\n")
            raise


    
        
        