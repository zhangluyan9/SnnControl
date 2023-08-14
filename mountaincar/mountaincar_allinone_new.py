import gym
from matplotlib import animation
import matplotlib.pyplot as plt
from numpy import cos, pi, sin

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import catSNN
import catCuda

CNN = False
SNN = True
Quantized_activation_level = 1
Integer_Weight = False
HDC = False
MPC = False
add_noise = [True,0.06,0]
quantized_input = [True,32]
quantized_weight = [True,32]
scale_factor = [-1,-1,-1]

factor_ = 1

print("CNN",CNN,"SNN",SNN,"Quantized_activation_level",Quantized_activation_level,"HDC",HDC,"add_noise",add_noise,"quantized_weight",quantized_weight)
def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    x = torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))#/2.0**(1-nbit)
    #print(x)
    return x

def quantize_to_bit_(x, nbit):
    #if nbit == 32:
    #    return x
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    #print(1/(2.0**(1-nbit)))
    #x =  torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))
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
def get_mem_p(out,fc):
                fc = fc
                out_ = out.clone()
                out_ = out_.reshape(out_.shape[0],out_.shape[2],out_.shape[1])
                for i in range(out.shape[0]):
                    out_[i] = out[i].mT
                out_s = fc(out_[0][0].reshape(1,24))
                #V
                for i in range(Quantized_activation_level-1):
                    out_s+=fc(out_[0][i+1])
                out_s = out_s/Quantized_activation_level
                return out_s
class Net(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        if CNN:
            x = self.fc1(state)
            #x = torch.clamp(x,0,1)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)

        if SNN:
            if Integer_Weight:
                self.fc1.weight.data = Quantization_integer(self.fc1.weight.data)
            #print(x)
            #print(self.fc1.weight.data)
            if quantized_weight[0]:
                factor = torch.max(torch.abs(self.fc1.weight))
                if factor<torch.max(torch.abs(self.fc1.bias)):
                    factor = torch.max(torch.abs(self.fc1.bias))
                self.fc1.weight.data /= factor
                self.fc1.weight.data = nn.Parameter(quantize_to_bit_(self.fc1.weight.data, quantized_weight[1]))

                self.fc1.bias.data/=factor
                self.fc1.bias.data= nn.Parameter(quantize_to_bit_(self.fc1.bias.data, quantized_weight[1]))
                #print(self.fc1.weight.data)
                
                if scale_factor[0]<0:
                    scale_factor[0] = factor
                    #print(scale_factor)
            #print(scale_factor[0])
            x = state
            if quantized_input[0]:
                #faco = torch.max(torch.abs(x))
                faco = torch.max(torch.abs(x))
                #print(faco)
                #print(faco)
                x/= faco
                x = quantize_to_bit_(x,quantized_input[1])
                x = F.linear(x,self.fc1.weight.data,2.0**(quantized_input[1]-1)*self.fc1.bias.data/faco)

            else:
                x = self.fc1(x) 

            threshold = 0.999
            if quantized_weight[0]:
                threshold = 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*threshold/scale_factor[0]
            if quantized_input[0]:
                threshold = 2.0**(quantized_input[1]-1)*(1-2.0**(1-quantized_input[1]))*threshold/faco

            #x = self.fc1(state)
            #print(factor,quantized_weight,quantized_weight[2])
            
            #print(factor)
            #x = x

            #generate spike train
            spikes_data = [x for _ in range(Quantized_activation_level)]
            out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
            """
            if quantized_weight[0]:
                #(1-2.0**(1-nbit))
                #out = catCuda.getSpikes(out, 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*0.999/scale_factor[0])
                #print(2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))* 0.2/scale_factor[0])
                out = catCuda.getSpikes(out, 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*0.999/scale_factor[0])
            else:
                out = catCuda.getSpikes(out, 0.999)
            """
            out = catCuda.getSpikes(out, threshold)
            reward_l1 = torch.sum(out)/24

            #print(out.shape)
            out = out.cpu()
            #reshape
            out_ = out.clone()
            out_ = out_.reshape(out_.shape[0],out_.shape[2],out_.shape[1])

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
                if scale_factor[1]<0:
                    scale_factor[1] = factor

            for i in range(out.shape[0]):
                out_[i] = out[i].mT
            out_s = self.fc2(out_[0][0].reshape(1,24))
            #V
            for i in range(Quantized_activation_level-1):
                out_s+=self.fc2(out_[0][i+1])
            out_s = out_s/Quantized_activation_level

            spikes_data = [out_s for _ in range(Quantized_activation_level)]
            out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
            #out = catCuda.getSpikes(out, 0.999)

            if quantized_weight[0]:
                out = catCuda.getSpikes(out, 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*0.999/scale_factor[1])
                #out = catCuda.getSpikes(out, 0.999/scale_factor[1])
            else:
                out = catCuda.getSpikes(out, 0.999)
            out_hdc = out.clone()

            reward_l2 = torch.sum(out)/24

            if quantized_weight[0]:
                factor = torch.max(torch.abs(self.fc3.weight))
                if factor<torch.max(torch.abs(self.fc3.bias)):
                    factor = torch.max(torch.abs(self.fc3.bias))
                self.fc3.weight.data /= factor
                self.fc3.weight.data = nn.Parameter(quantize_to_bit_(self.fc3.weight.data, quantized_weight[1]))

                self.fc3.bias.data/=factor
                self.fc3.bias.data= nn.Parameter(quantize_to_bit_(self.fc3.bias.data, quantized_weight[1]))
                #print(self.fc1.weight.data)
            #print(factor,quantized_weight,quantized_weight[2])
                if factor>scale_factor[2]:
                    scale_factor[2] = factor
            #print(scale_factor)
            out = out.cpu()
            #reshape
            #out = get_mem_p(out, self.fc3)
            
            out_ = out.clone()
            out_ = out_.reshape(out_.shape[0],out_.shape[2],out_.shape[1])
            for i in range(out.shape[0]):
                out_[i] = out[i].mT
            #print(out_.shape)

            out_s = self.fc3(out_[0][0].reshape(1,24))
            #V
            for i in range(Quantized_activation_level-1):
                out_s+=self.fc3(out_[0][i+1])
            #out_s = out_s/Quantized_activation_level
            if HDC and SNN:
                return out_s, out_hdc
            if SNN and not HDC:
                return out_s,reward_l1,reward_l2
            
        


        
class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions) # nit two nets
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=5e-4 )
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
                action_value,sum_addition_1,sum_addition_2 = self.eval_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()[0]
                return action,sum_addition_1,sum_addition_2
            else:
                action_value = self.eval_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()[0] # d the max value in softmax layer. before .data, it is a tensor
        else:
            action = np.random.randint(0, self.n_actions)
        # print("action=", action)
        return action

        
    def act(self, state, eps=0.):
        if type(state) == tuple:
            state = state[0]
        #print(state.shape)
        state = torch.from_numpy(state).float().unsqueeze(0)
        #print(state.shape)
        self.eval_net.eval()
        with torch.no_grad():
            action_values = self.eval_net(state)
        self.eval_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(4))

    def store_transition(self, state, action, reward, next_state):
        # print("<store_transition>")
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % 2000  
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
        self.optimizer.zero_grad()  
        loss.backward()  
        self.optimizer.step()  
    def save_model(self):
        torch.save(self.eval_net.state_dict(), "lunar_v1_DQN_1_1_.pt")
    def load_model(self):
        for param_tensor in self.eval_net.state_dict():
            print(param_tensor, "\t", self.eval_net.state_dict()[param_tensor].size())
        if SNN:
            #checkpoint_64_1_cq_q_level4
            #checkpoint_64_1_1_0430
            if Quantized_activation_level==1:
                self.eval_net.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q1.pth"), strict=False)
            if Quantized_activation_level==2:
                self.eval_net.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q2_.pth"), strict=False)
            if Quantized_activation_level==4:
                self.eval_net.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q4.pth"), strict=False)
        if CNN:
            self.eval_net.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_relu.pth"), strict=False)


class Net_hdc(nn.Module):
    def __init__(self, n_states, n_actions,noise=True):
        super(Net_hdc, self).__init__()
        self.fc1 = nn.Linear(n_states, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, n_actions)
        self.noise_ = noise

    def forward(self, x):
        if add_noise[0]:
            noise = add_noise[1]*torch.tensor(np.random.randn(x.shape[0])).type(torch.FloatTensor)
            x+=noise
            
        if Integer_Weight:
            self.fc1.weight.data = Quantization_integer(self.fc1.weight.data)
        #x = x.type(torch.DoubleTensor)
        """
        x = self.fc1(x)
        y = torch.clamp(x,0,1)
        y = Quantization_(y,Quantized_activation_level)

        #generate spike train
        spikes_data = [x for _ in range(Quantized_activation_level)]
        out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
        out = catCuda.getSpikes(out, 0.999)
        out = out.cpu()
        #print(out.shape)
        out = out.mT
        #print(out.shape)
        out_s = self.fc2(out[0])
        for i in range(Quantized_activation_level-1):
            out_s+=self.fc2(out[i+1])
        """
        x = self.fc1(x)
        #print(x)
        # train with cq
        y = torch.clamp(x,0,1)
        y = Quantization_(y,Quantized_activation_level)

        #generate spike train
        spikes_data = [x for _ in range(Quantized_activation_level)]
        out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
        out = catCuda.getSpikes(out, 0.999)
        #print(out.shape)
        out = out.cpu()
        out = out.mT
        #reshape
        out_s = self.fc2(out[0])
        #V
        for i in range(Quantized_activation_level-1):
            out_s+=self.fc2(out[i+1])
        out_s = out_s/Quantized_activation_level

        spikes_data = [out_s for _ in range(Quantized_activation_level)]
        out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
        out = catCuda.getSpikes(out, 0.999)
        out = out.cpu()
        out_hdc = out.clone()
        out = out.mT
        #print(out_.shape)
        out_s = self.fc3(out[0])
        #V
        for i in range(Quantized_activation_level-1):
            out_s+=self.fc3(out[i+1])
        out_s = out_s/Quantized_activation_level
    

        return out_s,out_hdc

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    # train
    counter = 0
    done_step = 0
    max_done_step = 0
    #num = 20
    num = 0000
    negative_reward = -10.0
    positive_reward = 10.0
    x_bound = 1.0
    state = env.reset()
    model = DQN(
        n_states=2,
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
        #s1 = np.arccos(state[0])
        #s2 = np.arccos(state[2])
        #done_f = -cos(s1)-cos(s1+s2)
        #if done_f>1:
            #count+=1
            #print(done_f)
            #reward = 10000
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
    #print("reccurent time = %d, max done step = %d, final done step = %d" % (retime, max_done_step, model.done_step_list[-1]))
    # test

    if HDC:
        model_ =  Net(2,3)
        #model =  Net_hdc(8,4,noise=False)
        if Quantized_activation_level==1:
            model_.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q1.pth"), strict=False)
        if Quantized_activation_level==2:
            model_.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q2_.pth"), strict=False)
        if Quantized_activation_level==4:
            model_.load_state_dict(torch.load("New_DQN_MountainCar-v0_128_mean_1_q4.pth"), strict=False)
        """
        represent_ = torch.zeros((4,24*Quantized_activation_level))

        for tt in range(500):
            state = env.reset()
            sum_re = 0
            state_collect  = []
            action_collect = []
            for ll in range(200):
                if type(state) == tuple:
                    state = state[0]
                action,output_ = model_(torch.tensor(state))
                output_ = output_.reshape(24*Quantized_activation_level)
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
        my_ones = torch.ones(represent_.shape[0],represent_.shape[1])
        my_zeros = -1*torch.ones(represent_.shape[0],represent_.shape[1])
        print(torch.max(represent_),torch.mean(represent_))
        #print(represent_)
        represent_ = torch.where(represent_<=1000, my_zeros, represent_)
        represent_ = torch.where(represent_>=1000 , my_ones, represent_)
        print(represent_)
        """
        represent_= torch.tensor([[-1.,  1.,  1., -1.,  1., -1., -1., -1., -1., -1., -1., -1.,  1., -1.,
          1.,  1., -1., -1.,  1.,  1., -1., -1., -1.,  1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [ 1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1.,  1.,  1., -1.,  1.,  1.,
          1.,  1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
         -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]]).cuda()
        for loop_n in range(15):
            loop_n = loop_n*0.01
            reward_s_ = 0
            scores = 0   
            min = 999
            max = -999     
            t_=100        # list containing scores from each episode
            for i_episode in range(t_):
                state = env.reset()
                score = 0
                c = 0
                d=0
                for t in range(200):
                    if type(state) == tuple:
                        state = state[0]
                    if add_noise[0]:
                        #print(state)
                        #*1*np.random.randn()
                        if add_noise[2]==0:
                            noise = loop_n*np.random.randn(state.shape[0])
                        if add_noise[2]==1:
                            noise = loop_n*np.random.uniform(-1,1,state.shape[0])
                        if add_noise[2]==2:
                            noise = loop_n*np.random.poisson(1, state.shape[0])
                            #print(noise)
                        if add_noise[2]==3:
                            noise = np.cumsum(np.random.normal(0, loop_n, size=state.shape[0]))
                        #noise = add_noise[1]*np.random.randn(state.shape[0])
                        #print(noise,state)
                        #print(noise.shape,state.shape)
                        state+=noise

                    state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)
                    #print(state.shape)
                    #action_ = model.act(state)
                    action,output_ = model_(state)
                    action = torch.argmax(action)
                    action = int(action)
                    output_ = output_.reshape(24*Quantized_activation_level)
                    output_minones = -1* torch.ones(output_.shape[0]).cuda()
                    output = torch.where(output_==0 , output_minones, output_)
                    #origin
                    testing_inference = torch.mm(output.reshape(1,output.shape[0]),represent_.T)
                    out = torch.argmax(testing_inference, dim=1)
                    out = int(out)
                    #print(out,action_,action)
                    next_state, reward, done,info, _ = env.step(out)
                    state = next_state
                    score += reward
                    if done:
                        break 
                if score<min:
                    min = score
                if score>max:
                    max = score
                scores+=score
                #print(d,c)
            print(scores/t_)
        #print(d,c)
        """
        for i_episode in range(10):
            state = env.reset()
            sum_re = 0
            state_collect  = []
            action_collect = []
            count =0
            for t in range(400):
                if type(state) == tuple:
                    state = state[0]
   
                action,output_ = model_(torch.tensor(state))
                action = torch.argmax(action)
                action = int(action)
                output_ = output_.reshape(24*Quantized_activation_level)
                output_minones = -1* torch.ones(output_.shape[0])
                output = torch.where(output_==0 , output_minones, output_)
                #origin
                testing_inference = torch.mm(output.reshape(1,output.shape[0]),represent_.T)
                out = torch.argmax(testing_inference, dim=1)
                out = int(out)
                #action = out

                state, reward, done, info,_ = env.step(action)

                if not done:
                    sum_re+=reward
                if done:
                    state = env.reset()
                    #print("test try again")
                    break
            #print(sum_re)
            reward_s_+=sum_re
            #print(sum_re)
        print("sumre",reward_s_/10)
        """

    else:    
        for loop_n in range(15):
            loop_n = loop_n*0.01
            min = 999
            max = -999
            scores = 0  
            additions_s_l1=0
            additions_s_l2=0      
            t_=100         # list containing scores from each episode
            for i_episode in range(t_):
                state = env.reset()
                score = 0
                for t in range(200):
                    if type(state) == tuple:
                        state = state[0]
                    if add_noise[0]:
                        #*1*np.random.randn()
                        #noise = np.random.uniform(low, high, signal.shape)
                        if add_noise[2]==0:
                            noise = loop_n*np.random.randn(state.shape[0])
                        if add_noise[2]==1:
                            noise = loop_n*np.random.uniform(-1,1,state.shape[0])
                        if add_noise[2]==2:
                            noise = loop_n*np.random.poisson(1, state.shape[0])
                            #print(noise)
                        if add_noise[2]==3:
                            noise = np.cumsum(np.random.normal(0, loop_n, size=state.shape[0]))
                        #print(noise,state)
                        #print(noise.shape,state.shape)
                        state+=noise

                    if SNN:
                        action,additions_l1,additions_l2 = model.choose_action(state, 1.0)
                        additions_s_l1+=additions_l1
                        additions_s_l2+=additions_l2
                    else: 
                        action = model.choose_action(state, 1.0)
                    next_state, reward, done,info, _ = env.step(action)
                    state = next_state
                    score += reward
                    if done:
                        break 
                if score<min:
                    min = score
                if score>max:
                    max = score
                scores+=score
            print(scores/t_)
        #print(scores/t_,additions_s_l1/(t_*200*Quantized_activation_level),additions_s_l2/(t_*200*Quantized_activation_level))

    
    #print(state_collect.shape)
    #print(action_collect.shape)
    #np.savez("cartpole_training_data_0327_500_1000.npz", train=state_collect, label=action_collect)
    #print(sum_re)
    env.close()
