import tensorflow as tf 
import numpy as np 
from scipy.misc import imresize

class DQN():

    def __init__(self, session, num_actions, num_frames, width, height, lr, startEpsilon, folderName):
        self.sess=session
        self.num_actions=num_actions
        self.folderName=folderName
        self.global_step=tf.get_variable("global_step", initializer=0, trainable=False)
        self.episode=tf.get_variable("episode", initializer=0, trainable=False)
        self.episode_Rewards=tf.get_variable("avg_Rew", initializer=tf.zeros((100,)), trainable=False)

        #HYPERPARAMETERS
        self.discount=0.99
        self.imageWidth=width
        self.imageHeight=height
        self.frames=num_frames #number of frames to stack to produce the input for the network
        self.state=[] #used to create the observation state to feed into the network

        self.learning_rate=lr 
        self.epsilon=startEpsilon
        
        #BUILD GRAPH AND INITIALIZE VARIABLES
        self.buildTargetNetwork()
        self.buildPredictionNetwork()
        self.buildTraining()
        self.buildwriteOps()


        self.sess.run(tf.global_variables_initializer())
        self.writeOps=tf.summary.FileWriter('results/'+self.folderName, self.sess.graph)
        self.saver=tf.train.Saver()

    def buildTargetNetwork(self):
        with tf.variable_scope("Target_network"):
            self.Target_X=tf.placeholder(tf.float32, shape=[None,self.imageWidth, self.imageHeight, self.frames], name="input")

            #take as input 4 channels (the 4 frames) and output 16 filters
            self.Target_filter_l1=tf.get_variable('filter_l1', [8,8,self.frames,32], initializer=tf.contrib.layers.xavier_initializer())
            self.Target_bias_l1=tf.get_variable('bias_l1', 32, initializer=tf.contrib.layers.xavier_initializer())
            self.Target_conv1=tf.nn.conv2d(self.Target_X, filter=self.Target_filter_l1, strides=[1,4,4,1], padding="SAME", name="conv_1")
            self.Target_out_l1=tf.nn.relu(tf.nn.bias_add(self.Target_conv1, self.Target_bias_l1))

            self.Target_filter_l2=tf.get_variable('filter_l2', [4,4,32,64], initializer=tf.contrib.layers.xavier_initializer())
            self.Target_bias_l2=tf.get_variable('bias_l2', 64, initializer=tf.contrib.layers.xavier_initializer())
            self.Target_conv2=tf.nn.conv2d(self.Target_out_l1, filter=self.Target_filter_l2, strides=[1,2,2,1], padding="SAME", name="conv_2")
            self.Target_out_l2=tf.nn.relu(tf.nn.bias_add(self.Target_conv2, self.Target_bias_l2))

            self.Target_filter_l3=tf.get_variable('filter_l3', [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
            self.Target_bias_l3=tf.get_variable('bias_l3', 64, initializer=tf.contrib.layers.xavier_initializer())
            self.Target_conv3=tf.nn.conv2d(self.Target_out_l2, filter=self.Target_filter_l3, strides=[1,1,1,1], padding="SAME", name="conv_3")
            self.Target_out_l3=tf.nn.relu(tf.nn.bias_add(self.Target_conv3, self.Target_bias_l3))

            #Flat the filters matricies in order to feed them to the fully connected layer
            #Retrieve tensor feature map's shape Target_out_l3
            shape=self.Target_out_l3.get_shape().as_list()
            dimension=shape[1]*shape[2]*shape[3]#Compute the dimension of the flatted vector
            #Flatting Target_out_l3
            self.Target_flatted=tf.reshape(self.Target_out_l3, [-1, dimension])

            self.Target_W4=tf.get_variable("weights_4", [dimension,512], initializer=tf.contrib.layers.xavier_initializer())
            self.Target_B4=tf.get_variable('bias_4', 512, initializer=tf.contrib.layers.xavier_initializer())
            self.Target_out_l4=tf.nn.relu(tf.nn.bias_add(tf.matmul(self.Target_flatted, self.Target_W4),self.Target_B4))

            self.Target_W5=tf.get_variable("weights_5", [512,self.num_actions], initializer=tf.contrib.layers.xavier_initializer())
            self.Target_B5=tf.get_variable('bias_5', self.num_actions, initializer=tf.contrib.layers.xavier_initializer())
            #Output a q(s',a') value for each possible action
            self.Target_output=tf.nn.bias_add(tf.matmul(self.Target_out_l4, self.Target_W5), self.Target_B5)

    def buildPredictionNetwork(self):
        with tf.variable_scope("Prediction_network"):
            #Input is a vector of size [batch_size,84,84,4]
            self.Pred_X=tf.placeholder(tf.float32, shape=[None,self.imageWidth, self.imageHeight, self.frames], name="input")

            #take as input 4 channels (the 4 frames) and output 16 filters
            self.Pred_filter_l1=tf.get_variable('filter_l1', [8,8,self.frames,32], initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_bias_l1=tf.get_variable('bias_l1', 32, initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_conv1=tf.nn.conv2d(self.Pred_X, filter=self.Pred_filter_l1, strides=[1,4,4,1], padding="SAME", name="conv_1")
            self.Pred_out_l1=tf.nn.relu(tf.nn.bias_add(self.Pred_conv1, self.Pred_bias_l1))

            self.Pred_filter_l2=tf.get_variable('filter_l2', [4,4,32,64], initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_bias_l2=tf.get_variable('bias_l2', 64, initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_conv2=tf.nn.conv2d(self.Pred_out_l1, filter=self.Pred_filter_l2, strides=[1,2,2,1], padding="SAME", name="conv_2")
            self.Pred_out_l2=tf.nn.relu(tf.nn.bias_add(self.Pred_conv2, self.Pred_bias_l2))

            self.Pred_filter_l3=tf.get_variable('filter_l3', [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_bias_l3=tf.get_variable('bias_l3', 64, initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_conv3=tf.nn.conv2d(self.Pred_out_l2, filter=self.Pred_filter_l3, strides=[1,1,1,1], padding="SAME", name="conv_3")
            self.Pred_out_l3=tf.nn.relu(tf.nn.bias_add(self.Pred_conv3, self.Pred_bias_l3))

            #Flat the filters matricies in order to feed them to the fully connected layer
            #Retrieve tensor feature map's shape Pred_out_l3
            shape=self.Pred_out_l3.get_shape().as_list() 
            dimension=shape[1]*shape[2]*shape[3] #Compute the dimension of the flatted vector
            #Flatting Pred_out_l3
            self.Pred_flatted=tf.reshape(self.Pred_out_l3, [-1, dimension])

            self.Pred_W4=tf.get_variable("weights_4", [dimension,512], initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_B4=tf.get_variable('bias_4', 512, initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_out_l4=tf.nn.relu(tf.nn.bias_add(tf.matmul(self.Pred_flatted, self.Pred_W4),self.Pred_B4))

            self.Pred_W5=tf.get_variable("weights_5", [512,self.num_actions], initializer=tf.contrib.layers.xavier_initializer())
            self.Pred_B5=tf.get_variable('bias_5', self.num_actions, initializer=tf.contrib.layers.xavier_initializer())
            #Output a q(s,a) value for each possible action
            self.Pred_output=tf.nn.bias_add(tf.matmul(self.Pred_out_l4, self.Pred_W5),self.Pred_B5)

    def updateTargetNetwork(self):
        #Retrieve prediction network's parameters
        pred_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Prediction_network")

        #Retrieve target network's parameters
        target_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Target_network")

        #Substitute the target network's parameter with the correspondent prediction network's parameter
        for pred_param, target_param in zip(pred_params, target_params):
            updateWeights=tf.placeholder(tf.float32, name="weights")
            target_param.assign(updateWeights).eval({updateWeights:pred_param.eval()})

    def buildTraining(self):
        self.input_Actions=tf.placeholder(tf.int32, name="actions_input")
        self.input_Targets=tf.placeholder(tf.float32, name="targets_input")

        #Every action is an integer between 0 and num_Actions. We encode them. matrix: [32,num_Actions]
        self.one_hot_actions=tf.one_hot(self.input_Actions, self.num_actions, name="encoding_actions")

        #For every state we obtain a single q value (the one used in the environment) matrix: [32,1]
        self.qsa=tf.reduce_sum(self.Pred_output * self.one_hot_actions, axis=1, name="computing_prediction")

        self.diff=self.input_Targets - self.qsa
        
        #HUBER LOSS
        self.use_linear_term=tf.cast((tf.abs(self.diff) > 1.0), tf.float32)
        self.quadratic_term=tf.square(self.diff)/2
        self.linear_term=tf.abs(self.diff)-1/2

        self.huber_loss=self.use_linear_term * self.linear_term + (1-self.use_linear_term) * self.quadratic_term

        #compute the average error
        self.loss=tf.reduce_mean(self.huber_loss, name="computing_loss")

        #apply AdamOptimizer
        self.opt=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def buildwriteOps(self):
        #episodeData
        self.averagedReward=tf.placeholder(tf.float32, name="Episode_Average_Reward")
        self.PHEpsilon=tf.placeholder(tf.float32, name="Epsilon_value")
        self.avgQValue=tf.placeholder(tf.float32, name="Avg_QValue")
        self.mergeEpisodeData=tf.summary.merge([tf.summary.scalar("Average_Reward_Episode", self.averagedReward),
                                                tf.summary.scalar("epsilonValue", self.PHEpsilon),
                                                tf.summary.scalar("Average_Q_Value", self.avgQValue)])

        #Filters images
        [shape1, shape2, shape3]=[self.Pred_filter_l1.get_shape().as_list(),self.Pred_filter_l2.get_shape().as_list(),self.Pred_filter_l3.get_shape().as_list()]
        self.mergeFilters=tf.summary.merge([tf.summary.image("Filters_L1", tf.reshape(tf.transpose(self.Pred_filter_l1,(2,0,3,1)), (shape1[-1]*shape1[-2], shape1[0], shape1[1], 1)), max_outputs=12),
                                            tf.summary.image("Filters_L2", tf.reshape(tf.transpose(self.Pred_filter_l2,(2,0,3,1)), (shape2[-1]*shape2[-2], shape2[0], shape2[1], 1)), max_outputs=12),
                                            tf.summary.image("Filters_L3", tf.reshape(tf.transpose(self.Pred_filter_l3,(2,0,3,1)), (shape3[-1]*shape3[-2], shape3[0], shape3[1], 1)), max_outputs=12),
                                            tf.summary.histogram("Filters_L1", self.Pred_filter_l1),
                                            tf.summary.histogram("Biases_L1", self.Pred_bias_l1),
                                            tf.summary.histogram("Filters_L2", self.Pred_filter_l2),
                                            tf.summary.histogram("Biases_L2", self.Pred_bias_l2),
                                            tf.summary.histogram("Filters_L3", self.Pred_filter_l3),
                                            tf.summary.histogram("Biases_L3", self.Pred_bias_l3),
                                            tf.summary.histogram("Weights_L4", self.Pred_W4),
                                            tf.summary.histogram("Biases_L4", self.Pred_B4),
                                            tf.summary.histogram("Weights_L5", self.Pred_W5),
                                            tf.summary.histogram("Biases_L5", self.Pred_B5)])

    def training(self, experiences):
        s,a,r,d,s1=experiences
        #Use the list of next states (s1) to compute all possible q(s',a'). 
        #It returns a matrix [32, num_actions]
        nextQsa=self.sess.run(self.Target_output, feed_dict={self.Target_X: s1})

        #For each next state (s1) retrieve max q(s',a'). 
        #maxNextQsa shape:[32,1]
        maxNextQsa=np.max(nextQsa, axis=1)

        #Convert the list of 32 elements to array [32,1]
        d=np.array(d)

        #Compute the target r + discount * max(q(s',a')). 
        #If d=1, so s1 is the terminal state use just r: (1-1)=0 -> target=0 + r
        #target is a matrix [32,1]
        target=((1-d)*self.discount * maxNextQsa) + r
        
        _=self.sess.run(self.opt, feed_dict={self.Pred_X:s, 
                                             self.input_Actions:a, 
                                             self.input_Targets:target})
        

    def actionSelection(self, inputState):
        #The state is reshaped [84,84,4] -> [1,84,84,4] and normalized. 
        #The network returns a vector [1,num_actions]
        actionDistrib=self.sess.run(self.Pred_output, feed_dict={self.Pred_X: np.expand_dims((inputState/255.), axis=0)})

        #e-greedy policy to select the action
        if np.random.rand()<self.epsilon:
            action=np.random.randint(0,self.num_actions)
        else:
            action=np.argmax(actionDistrib)

        return action, np.max(actionDistrib)

    def inputPreprocess(self, input_frame):
        #Convert RGB to grayscale [210,160,3] -> [210,160,1]
        grayFrame=np.mean(input_frame, axis=2)
        #Downsampling the image [210,160,1] -> [110,84,1]
        downsampledFrame=imresize(grayFrame, [110,84])
        #try and error to set the right height crop for the image in order that it is 84x84
        gap=20
        processedFrame=(downsampledFrame[gap:self.imageHeight+gap][:]).astype(np.uint8)

        if(len(self.state)==4):
            del self.state[0]
            self.state.append(processedFrame)
        else:
            while(len(self.state)!=4):
                self.state.append(processedFrame)
        
        #return the state [4,84,84] as [84,84,4]
        return np.transpose(self.state, (1,2,0))

    def resetObservationState(self):
        self.state=[]

    def save_restore_Model(self, restore, globa_step=None, episode=None, rewards=None):
        if restore:
            self.saver.restore(self.sess, "myModel/"+self.folderName+"/graph.ckpt")
        else:
            self.saveStats(globa_step, episode, rewards)
            self.saver.save(self.sess, "myModel/"+self.folderName+"/graph.ckpt")
    
    def saveStats(self, globa_step, episode, rewards):
        op1=self.global_step.assign(globa_step)
        op2=self.episode.assign(episode)

        if(len(rewards)<100):
            pad=np.full((100,), np.mean(rewards))
            pad[-len(rewards):]=rewards
        else:
            pad=rewards
        op3=self.episode_Rewards.assign(pad)

        _=self.sess.run([op1,op2,op3])


