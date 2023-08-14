import gym
from matplotlib import animation
import matplotlib.pyplot as plt

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
action_final = 0
Integer_Weight = False
HDC = False
MPC = False
add_noise = [True,0.06,0]
quantized_input = [True,32]
quantized_weight = [True,32]
scale_factor = [1,1]
print("CNN",CNN,"SNN",SNN,"Quantized_activation_level",Quantized_activation_level,"HDC",HDC,"add_noise",add_noise,"quantized_weight",quantized_weight)

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, Quantized_activation_level):
        ctx.constant = Quantized_activation_level
        return torch.div(torch.floor(torch.mul(tensor, Quantized_activation_level)), Quantized_activation_level)

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

class Quantization_train_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=4):
        ctx.constant = constant
        #return torch.floor(tensor)
        x = (1/(2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*0.999/constant))*tensor
        #print(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_train_weight = Quantization_train_weight.apply

def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    x = torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))#/2.0**(1-nbit)
    #print(x)
    return x

def quantize_to_bit_(x, nbit):
    if nbit == 32:
        return x
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    #print(x)
    return x
    #return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))

def quantize_to_bit_1(x, nbit):
    #if nbit == 32:
    #    return x
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    #print(x)
    return x
    #return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        #if add_noise[0]:
        #    noise = add_noise[1]*torch.tensor(np.random.randn(x.shape[0],x.shape[1])).type(torch.FloatTensor)
        #    x+=noise
        if CNN:

            x = self.fc1(x)
            x = F.relu(x)
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

                self.fc1.bias.data/=factor
                self.fc1.bias.data= nn.Parameter(quantize_to_bit_(self.fc1.bias.data, quantized_weight[1]))
                #print(self.fc1.weight.data)
                if factor>scale_factor[0]:
                    scale_factor[0] = factor


            if quantized_input[0]:
                faco = 4.8
                x/= faco
                x = quantize_to_bit_1(x,quantized_input[1])
                x = F.linear(x,self.fc1.weight.data,2.0**(quantized_input[1]-1)*self.fc1.bias.data/faco)

            else:
                x = self.fc1(x)      

            threshold = 0.999
            if quantized_weight[0]:
                threshold = 2.0**(quantized_weight[1]-1)*(1-2.0**(1-quantized_weight[1]))*threshold/scale_factor[0]
            if quantized_input[0]:
                threshold = 2.0**(quantized_input[1]-1)*(1-2.0**(1-quantized_input[1]))*threshold/faco

            #generate spike train
            spikes_data = [x for _ in range(Quantized_activation_level)]
            out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
            out = catCuda.getSpikes(out,0.999)
            sum_addition = torch.sum(out)
            out_hdc = out.clone()
            #print(out)
            #print(sum_addition,out.shape)
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
                if factor>scale_factor[1]:
                    scale_factor[1] = factor

            out_s = self.fc2(out_[0][0].reshape(1,10))
            for i in range(Quantized_activation_level-1):
                out_s+=self.fc2(out_[0][i+1])
            if SNN and not HDC:
                return out_s,sum_addition
            if SNN and HDC:
                return out_s,out_hdc
        return out


def my_step_(state,action):
    x, x_dot, theta, theta_dot = state
    polemass_length = 0.5*0.1
    masspole = 0.1
    total_mass = 0.1+1.0
    gravity = 9.8
    tau = 0.02
    gravity = 9.8
    """
    self.masscart = 1.0
    self.masspole = 0.1
    self.total_mass = self.masspole + self.masscart
    self.length = 0.5  # actually half the pole's length
    self.polemass_length = self.masspole * self.length
    self.force_mag = 10.0
    self.tau = 0.02  # seconds between state updates
    """
    force =10.0 if action == 1 else -10.0
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (0.5 * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    #temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
    #thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
    #xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    state = (x, x_dot, theta, theta_dot)
    return (np.array(state, dtype=np.float32))

class DQN:
    def __init__(self, n_states, n_actions):
        #print("<DQN init>")
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions) # nit two nets
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.00001)
        self.n_actions = n_actions
        self.n_states = n_states
        self.learn_step_counter = 0  
        self.memory_counter = 0  
        self.memory = np.zeros((2000, 2 * n_states + 1 + 1))  # s, s', a, r
        self.cost = []  
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
        torch.save(self.eval_net.state_dict(), "cartpole_v1_quantized_0526.pt")
    def load_model(self):
        #for param_tensor in self.eval_net.state_dict():
        #    print(param_tensor, "\t", self.eval_net.state_dict()[param_tensor].size())
        if CNN:
            #cartpole_v1_DQN_1
            self.eval_net.load_state_dict(torch.load("cartpole_v1_DQN_1.pt"), strict=False)
        if SNN:
            self.eval_net.load_state_dict(torch.load("cartpole_v1_DQN_1_0327_10.pt"), strict=False)
        if SNN and Integer_Weight:
            self.eval_net.load_state_dict(torch.load("cartpole_v1_DQN_1_bnn_integerweight.pt"), strict=False)


class Net_hdc(nn.Module):
    def __init__(self, n_states, n_actions,noise=True):
        super(Net_hdc, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.noise_ = noise

    def forward(self, x):
        #if add_noise[0] and self.noise_:
        #    noise = add_noise[1]*torch.tensor(np.random.randn(x.shape[0])).type(torch.FloatTensor)
        #    x+=noise
        if Integer_Weight:
            self.fc1.weight.data = Quantization_integer(self.fc1.weight.data)
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

        return out_s,out

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    # train
    counter = 0
    done_step = 0
    max_done_step = 0
    num = 00000
    #num = 200000
    negative_reward = -10.0
    positive_reward = 10.0
    x_bound = 1.0
    state = env.reset()
    model = DQN(
        n_states=4,
        n_actions=2
    )  # 算法模型
    model.cost.clear()
    model.done_step_list.clear()
    model.load_model()
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
        x, x_dot, theta, theta_dot = state
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # x_threshold 4.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        if (abs(x) > x_bound):
            r1 = 0.5 * negative_reward
        else:
            r1 = negative_reward * abs(x) / x_bound + 0.5 * (-negative_reward)
        if (abs(theta) > env.theta_threshold_radians):
            r2 = 0.5 * negative_reward
        else:
            r2 = negative_reward * abs(theta) / env.theta_threshold_radians + 0.5 * (-negative_reward)
        reward = r1 + r2
        if (done) and (done_step < 499):
            reward += negative_reward
        # print("x = %lf, r1 = %lf, theta = %lf, r2 = %lf" % (x, r1, theta, r2))
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
    #model.save_model()
    #model.load_model()  # 误差曲线
    #print("reccurent time = %d, max done step = %d, final done step = %d" % (retime, max_done_step, model.done_step_list[-1]))
    # test

    if HDC:
        model_ =  Net(4,2)
        model_.load_state_dict(torch.load("cartpole_v1_DQN_1_0327_10.pt"), strict=True)
        represent_ = torch.zeros((2,10*Quantized_activation_level))
        """
        for tt in range(100):
            state = env.reset()
            sum_re = 0
            state_collect  = []
            action_collect = []
            for _ in range(2000):
                if type(state) == tuple:
                    state = state[0]
                state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)
                action,output_ = model_(torch.tensor(state))
                output_ = output_.reshape(10*Quantized_activation_level)
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
        print(torch.max(represent_))
        represent_ = torch.where(represent_<=3000, my_zeros, represent_)
        represent_ = torch.where(represent_>=3000 , my_ones, represent_)
        print(represent_)
        """
        represent_ = torch.tensor([[ 1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1.],
        [ 1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1., -1.]]).cuda()
        print(represent_)
        
        for loop_n in range(15):
            reward_s_ = 0
            loop_n = (loop_n)*0.01
            for tt in range(100):
                state = env.reset()
                sum_re = 0
                state_collect  = []
                action_collect = []
                count =0
                for steps_ in range(2000):
                    if type(state) == tuple:
                        state = state[0]
                    state_withnoise = state.copy()
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
                    state_mpc = state.copy()
                    #print('statempc',state_mpc)
                    
                    state = torch.unsqueeze(torch.FloatTensor(state), 0)  # (1,2)
                    action,output_ = model_(state)
                    output_ = output_.reshape(10*Quantized_activation_level)
                    output_minones = -1* torch.ones(output_.shape[0]).cuda()
                    output = torch.where(output_==0 , output_minones, output_)
                    #origin
                    testing_inference = torch.mm(output.reshape(1,output.shape[0]),represent_.T)
                    out = torch.argmax(testing_inference, dim=1)
                    out = int(out)
                    action = out

                    #print(action)
                    done_ = False
                    
                    #print(action_final,action)
                    #print(action_final,action)
                    state, reward, done, info,_ = env.step(action)
                    #print('real',state)

                    if not done:
                        sum_re+=reward

                    if done:
                        state = env.reset()
                        #print("test try again")
                        break
                #print(sum_re)
                reward_s_+=sum_re
                #print(sum_re)
            print("sumre",reward_s_/100)

    else:
        for loop_n in range(15):
            loop_n = loop_n*0.01
            #print(loop_n)
            reward_ave = 0
            times = 100
            additions_s = 0
            noi = [0,0,0,0]
            for _ in range(times):
                state = env.reset()
                #print("####",state)
                sum_re = 0
                for _ in range(2000):
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

                    a =  torch.tensor(state)
                    a = a.type(torch.FloatTensor)
                    #print(a,a.type)
                    if SNN:
                        action,additions = model.choose_action(state, 1.0)
                        additions_s+=additions
                    else: 
                        action = model.choose_action(state, 1.0)
                    #print(state,action)
                    
                    state, reward, done, info,_ = env.step(action)
                    if not done:
                        sum_re+=reward
                    if done:
                        state = env.reset()
                        #print("test try again")
                        break
                reward_ave+=sum_re
            noi[0]/=2000*times
            noi[1]/=2000*times
            noi[2]/=2000*times
            noi[3]/=2000*times
            print(reward_ave/times)

    
    #print(state_collect.shape)
    #print(action_collect.shape)
    #np.savez("cartpole_training_data_0327_500_1000.npz", train=state_collect, label=action_collect)
    #print(sum_re)
    env.close()
