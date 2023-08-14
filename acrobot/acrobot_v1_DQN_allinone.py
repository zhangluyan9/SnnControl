import gym
from matplotlib import animation
import matplotlib.pyplot as plt
from numpy import cos, pi, sin

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import catSNN
import catCuda
import math

CNN = False
SNN = True
Quantized_activation_level = 1
Integer_Weight = False
HDC = True
MPC = False
add_noise = [True,0.8,3]
quantized_weight = [True,8]
quantized_input = [True,8]
nbit = quantized_weight[1]
scale_factor = [-1,-1]
print("CNN",CNN,"SNN",SNN,"Quantized_activation_level",Quantized_activation_level,"HDC",HDC,"add_noise",add_noise,"quantized_weight",quantized_weight)
def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    y=x
    
    x = (1- 2.0**(1-nbit))*x
    #x = torch.clamp(x,0,1)
    #x = torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))#/2.0**(1-nbit)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    #x = x*
    #x = torch.clamp(x,2.0**(1-nbit)-1,1-2.0**(1-nbit))
    #print(y,x)
    #print(torch.min(torch.round(torch.div(x, 2.0**(1-nbit)))))
    #print(x)
    return x
def quantize_to_bit_(x, nbit):
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    #print(x)
    return x

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class Quantization_integer_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=1000):
        ctx.constant = constant
        #return torch.floor(tensor)
        x = tensor
        #x = torch.sign(x)
        x_ = torch.where(x>=0 , torch.div(torch.ceil(torch.mul(x, 1)), 1), x)
        x = torch.where(x_<0 , torch.div(torch.floor(torch.mul(x_, 1)), 1), x_)
        #print(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_integer = Quantization_integer_.apply



class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        #print(torch.max(torch.abs(x)))
        #if add_noise[0]:
        #    noise = add_noise[1]*torch.tensor(np.random.randn(x.shape[0],x.shape[1])).type(torch.FloatTensor)
        #    x+=noise
        if CNN:
            if Integer_Weight:
                self.fc1.weight.data = Quantization_integer(self.fc1.weight.data)
            x = self.fc1(x)
            #train_cq
            x = torch.clamp(x,0,1)
            #x = Quantization_(x,Quantized_activation_level)

            #train_relu
            #x = F.relu(x)
            out = self.fc2(x)
        if SNN:
            if Integer_Weight:
                self.fc1.weight.data = Quantization_integer(self.fc1.weight.data)
            #print(x)
            if quantized_weight[0]:
                factor = torch.max(torch.abs(self.fc1.weight))
                if factor<torch.max(torch.abs(self.fc1.bias)):
                    factor = torch.max(torch.abs(self.fc1.bias))
                self.fc1.weight.data /= factor
                self.fc1.weight.data = nn.Parameter(quantize_to_bit_(self.fc1.weight.data, quantized_weight[1]))
                #print(self.fc1.weight.data,torch.min(self.fc1.weight.data))

                self.fc1.bias.data/=factor
                self.fc1.bias.data= nn.Parameter(quantize_to_bit_(self.fc1.bias.data, quantized_weight[1]))
                #print(self.fc1.weight.data)
            
            #print(factor,quantized_weight,quantized_weight[2])
                #
                if scale_factor[0]<0:
                    scale_factor[0] = factor
                #print( factor,scale_factor[0])
                #print(scale_factor[0])

            if quantized_input[0]:
                #faco = torch.max(torch.abs(x))
                faco = 28.274334
                #print(faco)
                x/= faco
                x = quantize_to_bit_(x,quantized_input[1])
                x = F.linear(x,self.fc1.weight.data,2.0**(quantized_input[1]-1)*self.fc1.bias.data/faco)

            else:
                x = self.fc1(x)
                
            #x = x

            threshold = 0.999
            if quantized_weight[0]:
                threshold = 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*threshold/scale_factor[0]
            if quantized_input[0]:
                threshold = 2.0**(quantized_input[1]-1)*(1-2.0**(1-quantized_input[1]))*threshold/faco

            #generate spike train
            spikes_data = [x for _ in range(Quantized_activation_level)]
            out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
            out = catCuda.getSpikes(out, threshold)
            out_hdc = out.clone()
            sum_addition = torch.sum(out)/64
            
            out = out.cpu()
            #reshape
            out_ = out.clone()
            out_ = out_.reshape(out_.shape[0],out_.shape[2],out_.shape[1])
            for i in range(out.shape[0]):
                out_[i] = out[i].mT
            
            if quantized_weight[0]:
                factor = torch.max(torch.abs(self.fc2.weight))
                if factor<torch.max(torch.abs(self.fc2.bias)):
                    factor = torch.max(torch.abs(self.fc2.bias))
                self.fc2.weight.data /= factor
                self.fc2.weight.data = nn.Parameter(quantize_to_bit_(self.fc2.weight.data, quantized_weight[1]))

                self.fc2.bias.data/=factor
                self.fc2.bias.data= nn.Parameter(quantize_to_bit_(self.fc2.bias.data, quantized_weight[1]))
                #print(self.fc1.weight.data)
            #print(factor,quantized_weight,quantized_weight[2])

            out_s = self.fc2(out_[0][0].reshape(1,64))

            for i in range(Quantized_activation_level-1):
                out_s+=self.fc2(out_[0][i+1])
            if SNN and not HDC:
                return out_s,sum_addition
            if HDC and SNN:
                return out_s,out_hdc

        return out

class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions) # nit two nets
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.n_actions = n_actions
        self.n_states = n_states
        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((2000, 2 * n_states + 1 + 1))  # s, s', a, r
        self.cost = []  # 记录损失值
        self.done_step_list = []

    def choose_action(self, state, epsilon):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)
        if np.random.uniform() < epsilon:
            if SNN:
                action_value,sum_addition = self.eval_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()[0]
                return action,sum_addition
            else:
                action_value = self.eval_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()[0] # d the max value in softmax layer. before .data, it is a tensor
        else:
            action = np.random.randint(0, self.n_actions)
        # print("action=", action)
        return action

    def store_transition(self, state, action, reward, next_state):
        # print("<store_transition>")
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % 2000  # 满了就覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("<learn>")
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
            # print("update eval to target")
        self.learn_step_counter += 1

        sample_index = np.random.choice(2000, 16)  
        memory = self.memory[sample_index, :]  
        state = torch.FloatTensor(memory[:, :self.n_states])
        action = torch.LongTensor(memory[:, self.n_states:self.n_states + 1])
        reward = torch.LongTensor(memory[:, self.n_states + 1:self.n_states + 2])
        next_state = torch.FloatTensor(memory[:, self.n_states + 2:])

        q_eval = self.eval_net(state).gather(1, action) 
        q_next = self.target_net(next_state).detach()
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1) # label
        #ddqn
        #q_target = reward + 0.9 * self.eval_net(next_state).detach().max(1)[0].unsqueeze(1) # label

        loss = self.loss(q_eval, q_target)  # td error
        self.cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  
    def save_model(self):
        torch.save(self.eval_net.state_dict(), "acrobot_v1_DQN_1_1_small_0425_quan_0529_.pt")
    def load_model(self):
        #for param_tensor in self.eval_net.state_dict():
        #    print(param_tensor, "\t", self.eval_net.state_dict()[param_tensor].size())
        #acrobot_v1_DQN_1_1_small
        #acrobot_v1_DQN_1_1_small_0425
        #acrobot_v1_DQN_1_1_small_0425_quan
        if CNN:
            self.eval_net.load_state_dict(torch.load("acrobot_v1_DQN_1_1_small_0425_quan.pt"), strict=False)
        if SNN:
            self.eval_net.load_state_dict(torch.load("acrobot_v1_DQN_1_1_small_0425_quan.pt"), strict=False)



if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    # train
    counter = 0
    done_step = 0
    max_done_step = 0
    #num = 20
    num = 000
    negative_reward = -10.0
    positive_reward = 10.0
    x_bound = 1.0
    state = env.reset()
    model = DQN(
        n_states=6,
        n_actions=3
    )  # 算法模型
    model.cost.clear()
    model.done_step_list.clear()
    model.load_model()
    count= 0
    for i in range(num):
        # env.render()
        epsilon = 0.9 + i / num * (0.95 - 0.9)
        # epsilon = 0.9
        if type(state) == tuple:
            state = state[0]
        action = model.choose_action(state, epsilon)
        # print('action = %d' % action)
        state_old = state
        #print(env.step(action))
        state, reward, done, info,_ = env.step(action)
        s1 = np.arccos(state[0])
        s1_ = np.arcsin(state[1])
        if cos(s1)==state[0] and sin(s1) == state[1]:
            theta1 = s1
        else:
            theta1 = s1_

        s2 = np.arccos(state[2])
        s2_ = np.arcsin(state[3])

        if cos(s2)==state[2] and sin(s2) == state[3]:
            theta2 = s2
        else:
            theta2 = s2_
        done_f = -cos(theta1)-cos(theta1+theta2)

        if done_f>1:
            count+=1
            #print(done_f)
            reward = 10000
        #print(np.arccos(state[0]),cos(np.arccos(state[0])),state[0])
        #-cos(s[0]) - cos(s[1] + s[0]) > 1.0
        model.store_transition(state_old, action, reward, state)
        if (i > 2000 and counter % 10 == 0):
            model.learn()
            counter = 0
        counter += 1
        done_step += 1

        if (done):
            # print("reset env! done_step = %d, epsilon = %lf" % (done_step, epsilon))
            if (done_step > max_done_step):
                max_done_step = done_step
            state = env.reset()
            model.done_step_list.append(done_step)
            done_step = 0
        #if i%100==0:
        #    print(i/100)
    print(count)
    #model.save_model()
    #model.load_model()  
    #print("reccurent time = %d, max done step = %d, final done step = %d" % (retime, max_done_step, model.done_step_list[-1]))
    # test
    if HDC:
        model_ =  Net(6,3)
        model =  Net(6,3)
        model_.load_state_dict(torch.load("acrobot_v1_DQN_1_1_small_0425_quan.pt"), strict=True)
        model.load_state_dict(torch.load("acrobot_v1_DQN_1_1_small_0425_quan.pt"), strict=True)
        represent_ = torch.zeros((3,64*Quantized_activation_level)).cuda()
        """
        for tt in range(100):
            state = env.reset()
            sum_re = 0
            state_collect  = []
            action_collect = []
            for _ in range(1000):
                if type(state) == tuple:
                    state = state[0]
                state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)

                action,output_ = model_(state)
                output_ = output_.reshape(64*Quantized_activation_level)
                action = torch.argmax(action)
                action = int(action)
                represent_[action]+=output_
                state, reward, done, info,_ = env.step(action)
                if not done:
                    sum_re+=reward
                if done:
                    state = env.reset()
                    #print("test try again")
                    break
        my_ones = torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        my_zeros = -1*torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        print(torch.max(represent_))
        represent_ = torch.where(represent_<=2500, my_zeros, represent_)
        represent_ = torch.where(represent_>=2500 , my_ones, represent_)
        print(represent_)
        """
        represent_ = torch.tensor([
        [ 1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1., -1., -1.,  1., -1.,  1.,
         -1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,
          1.,  1., -1.,  1.,  1., -1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1.,
         -1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,
         -1., -1.,  1., -1.,  1.,  1.,  1.,  1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1.,  1., -1.,
         -1., -1., -1.,  1., -1., -1.,  1.,  1., -1., -1., -1., -1., -1., -1.,
         -1., -1.,  1., -1., -1.,  1., -1., -1.,  1.,  1.,  1., -1., -1., -1.,
          1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1., -1., -1.,
          1.,  1.,  1.,  1., -1., -1., -1., -1.]]).cuda()
        #print(represent_.shape)
        
        #reward_s_ = 0
        repear_t=1
        if add_noise[0]:
            repear_t=15

        for xtimes in range(repear_t+1):
            #sum_re = 0
            reward_s_ = 0
            for tt in range(100):
                state = env.reset()
                sum_re = 0
                state_collect  = []
                action_collect = []
                count =0
                for _ in range(500):
                    if type(state) == tuple:
                        state = state[0]
                    if add_noise[0]:
                        s1 = np.arccos(state[0])
                        s1_ = np.arcsin(state[1])
                        if cos(s1)==state[0] and sin(s1) == state[1]:
                            theta1 = s1
                        else:
                            theta1 = s1_

                        s2 = np.arccos(state[2])
                        s2_ = np.arcsin(state[3])

                        if cos(s2)==state[2] and sin(s2) == state[3]:
                            theta2 = s2
                        else:
                            theta2 = s2_


                        v1 = state[4]
                        v2 = state[5]
                        #print(s1)
                        if add_noise[2]==0:
                            theta1+= 0.1*(xtimes)*np.random.randn()
                            theta2+=0.1*(xtimes)*np.random.randn()
                            v1+=0.1*4*(xtimes)*np.random.randn()
                            v2+=0.1*9*(xtimes)*np.random.randn()
                        if add_noise[2]==1:
                            theta1+= 0.1*(xtimes)*np.random.uniform(-1,1,theta1.shape)
                            theta2+=0.1*(xtimes)*np.random.uniform(-1,1,theta2.shape)
                            v1+=0.1*4*(xtimes)*np.random.uniform(-1,1,v1.shape)
                            v2+=0.1*9*(xtimes)*np.random.uniform(-1,1,v2.shape)


                        if add_noise[2]==2:
                            #noise = 0.1*(xtimes)*np.random.poisson(1, state.shape[0])
                            theta1+= 0.1*(xtimes)*np.random.poisson(1, theta1.shape)
                            theta2+=0.1*(xtimes)*np.random.poisson(1, theta2.shape)
                            v1+=0.1*4*(xtimes)*np.random.poisson(1, v1.shape)
                            v2+=0.1*9*(xtimes)*np.random.poisson(1, v2.shape)
                            #print(noise)
                        if add_noise[2]==3:
                            #print(theta1,theta2,v1,v2)
                            theta1+= np.cumsum(np.random.normal(0, 0.1*(xtimes), size=theta1.shape))
                            theta2+=np.cumsum(np.random.normal(0, 0.1*(xtimes), size=theta2.shape))
                            
                            v1+=np.cumsum(np.random.normal(0, 0.4*(xtimes), size=v1.shape))
                            v2+=np.cumsum(np.random.normal(0, 0.9*(xtimes), size=v2.shape))
                            #print(theta1,theta2,v1,v2)
                            theta1 = theta1[0]
                            theta2 = theta2[0]
                            v1 = v1[0]
                            v2 = v2[0]

                        #theta1+=0.1*(xtimes)*np.random.randn()
                        #theta2+=0.1*(xtimes)*np.random.randn()

                        #v1+=0.1*4*(xtimes)*np.random.randn()
                        #v2+=0.1*9*(xtimes)*np.random.randn()
                        #print(s1)
                        state = [cos(theta1),sin(theta1),cos(theta2),sin(theta2),v1,v2]
                        #print(state)
                        state = np.array(state)
                        state = state.astype('float32')
                        #print(state)
                    state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)

                    action,output_ = model_(state)
                    output_ = output_.reshape(64*Quantized_activation_level)
                    output_minones = -1* torch.ones(output_.shape[0]).cuda()
                    output = torch.where(output_==0 , output_minones, output_)
                    #origin
                    testing_inference = torch.mm(output.reshape(1,output.shape[0]),represent_.T)
                    out = torch.argmax(testing_inference, dim=1)
                    out = int(out)
                    action = out

                    if MPC:
                        mpc_steps = 5
                        action_ = action
                        state_ = state
                        done_ = False
                        #print(action)
                        
                        for i in range(mpc_steps):
                            state_ = my_step_(state_,action_)
                            if state_[0]>2.4 or state_[0]<-2.4 or state_[2]>0.209 or state_[2]<-0.209:
                                done_ = True
                                #print(tt)
                                break
                            else:
                                done_ = False

                            action_,output_ = model(torch.tensor(state_))
                            output_ = output_.reshape(64*Quantized_activation_level)
                            output_minones = -1* torch.ones(output_.shape[0])
                            output = torch.where(output_==0 , output_minones, output_)
                            testing_inference = torch.mm(output.reshape(1,output.shape[0]),represent_.T)
                            out = torch.argmax(testing_inference, dim=1)
                            action_ = int(out)

                        if done_:
                            #print(action)
                            action = 1-action
                            #print(action)
                            #print(i)

                    state, reward, done, info,_ = env.step(action)

                    if not done:
                        reward_s_+=reward
                    if done:
                        state = env.reset()
                        #print("test try again")
                        break
                #print(sum_re)
                #reward_s_+sum_re
                #print(sum_re)
            print(reward_s_/100)

    else:
        repear_t=0
        if add_noise[0]:
            repear_t=15
        for xtimes in range(repear_t+1):
            #print(xtimes)
            state_collect  = []
            action_collect = []
            reward_a = 0
            ite = 100
            additions_s = 0
            
            for ite_ in range(ite):
                state = env.reset()
                #print("####",state)
                sum_re = 0
                step_ = 0
                #number_it = 500*(mmmm+1)
                for t_ in range(500):
                    if type(state) == tuple:
                        state = state[0]
                    if add_noise[0]:
                        s1 = np.arccos(state[0])
                        s1_ = np.arcsin(state[1])
                        if cos(s1)==state[0] and sin(s1) == state[1]:
                            theta1 = s1
                        else:
                            theta1 = s1_

                        s2 = np.arccos(state[2])
                        s2_ = np.arcsin(state[3])

                        if cos(s2)==state[2] and sin(s2) == state[3]:
                            theta2 = s2
                        else:
                            theta2 = s2_


                        v1 = state[4]
                        v2 = state[5]
                        #print(s1)
                        
                        if add_noise[2]==0:
                            theta1+= 0.1*(xtimes)*np.random.randn()
                            theta2+=0.1*(xtimes)*np.random.randn()
                            v1+=0.1*4*(xtimes)*np.random.randn()
                            v2+=0.1*9*(xtimes)*np.random.randn()
                        if add_noise[2]==1:
                            theta1+= 0.1*(xtimes)*np.random.uniform(-1,1,theta1.shape)
                            theta2+=0.1*(xtimes)*np.random.uniform(-1,1,theta2.shape)
                            v1+=0.1*4*(xtimes)*np.random.uniform(-1,1,v1.shape)
                            v2+=0.1*9*(xtimes)*np.random.uniform(-1,1,v2.shape)


                        if add_noise[2]==2:
                            #noise = 0.1*(xtimes)*np.random.poisson(1, state.shape[0])
                            theta1+= 0.1*(xtimes)*np.random.poisson(1, theta1.shape)
                            theta2+=0.1*(xtimes)*np.random.poisson(1, theta2.shape)
                            v1+=0.1*4*(xtimes)*np.random.poisson(1, v1.shape)
                            v2+=0.1*9*(xtimes)*np.random.poisson(1, v2.shape)
                            #print(noise)
                        if add_noise[2]==3:
                            #print(theta1,theta2,v1,v2)
                            theta1+= np.cumsum(np.random.normal(0, 0.1*(xtimes), size=theta1.shape))
                            theta2+=np.cumsum(np.random.normal(0, 0.1*(xtimes), size=theta2.shape))
                            
                            v1+=np.cumsum(np.random.normal(0, 0.4*(xtimes), size=v1.shape))
                            v2+=np.cumsum(np.random.normal(0, 0.9*(xtimes), size=v2.shape))
                            #print(theta1,theta2,v1,v2)
                            theta1 = theta1[0]
                            theta2 = theta2[0]
                            v1 = v1[0]
                            v2 = v2[0]

                        #print(s1)
                        state = [cos(theta1),sin(theta1),cos(theta2),sin(theta2),v1,v2]
                    #print(state)

                    #print("before",state)
                    #print(np.random.rand(4))
                    #state+= 0.14*np.random.rand(4)
                    #print("after",state)
                    a =  torch.tensor(state)
                    a = a.type(torch.FloatTensor)
                    #print(a,a.type)
                    if SNN:
                        action,additions = model.choose_action(state, 1.0)
                        additions_s+=additions
                    else: 
                        action = model.choose_action(state, 1.0)


                    #print(a,a.type)
                    action_collect.append(action)
                    state_collect.append(state)

                    #print(state,action)
                    
                    state, reward, done, info,_ = env.step(action)
                    step_+=1
                    if not done:
                        sum_re+=reward
                    if done:
                        #step_+=1
                        state = env.reset()
                        #print("test try again")
                        break
                reward_a+=sum_re
                
                #print(sum_re)
                #print(step_)
            print(reward_a/ite)
            #print(reward_a/ite,additions_s/(ite*500*Quantized_activation_level))

            #state_collect = np.array(state_collect)
            #action_collect = np.array(action_collect)

            
            #print(state_collect.shape)
            #print(action_collect.shape)
            #np.savez("cartpole_training_data_0327_500_1000.npz", train=state_collect, label=action_collect)
            #print(sum_re)
            env.close()
