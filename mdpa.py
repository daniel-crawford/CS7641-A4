# MDP Assignment
import matplotlib.pyplot as plt
import numpy as np
import mdptoolbox
from mdptoolbox.example import rand as randMDP
from mdptoolbox.example import forest as forestMDP
from mdptoolbox.mdp import PolicyIteration as PI
from mdptoolbox.mdp import ValueIteration as VI
from mdptoolbox.mdp import QLearning as QL


#NUM_STATES = 1000
#NUM_ACTIONS = 10
#DISCOUNT = 0.99
#MAX_ITERATIONS = 10000
#VI_EPSILON = 0.01

#FM_states = 10
#FM_r1 = 6
#FM_r2 = 2
#FM_p = 0.1

#P1 - Random MDP
#mdp = randMDP(NUM_STATES, NUM_ACTIONS)

#P2 - Forest Management
#mdp = forestMDP(S = FM_states, r1 = FM_r1, r2 = FM_r2, p = FM_p) 

'''  
pi = PI(transitions = mdp[0],
        reward = mdp[1],
        discount = DISCOUNT,
        max_iter = MAX_ITERATIONS,
        eval_type = 'iterative')
pi.run()

vi = VI(transitions = mdp[0],
        reward = mdp[1],
        discount = DISCOUNT,
        epsilon = VI_EPSILON,
        max_iter = MAX_ITERATIONS)
vi.run()

ql = QL(transitions = mdp[0],
        reward = mdp[1],
        discount = DISCOUNT,
        n_iter = max(MAX_ITERATIONS,10000))

ql.run()
'''

'''       
#How Many Iterations to converge:
pi_iters, vi_iters = [],[]
for s in range(1,2001):
    for a in range(1,2001):
          
        pi = PI(transitions = mdp[0],
                reward = mdp[1],
                discount = DISCOUNT,
                max_iter = MAX_ITERATIONS,
                eval_type = 'iterative')
        pi.run()

        vi = VI(transitions = mdp[0],
                reward = mdp[1],
                discount = DISCOUNT,
                epsilon = VI_EPSILON,
                max_iter = MAX_ITERATIONS)
       vi.run()

        pi_iters.append(pi.iter)
        vi_iters.append(vi.iter)

'''


'''
def randMDPiterateVI(s, a, d, m, c):
    mdp = randMDP(s, a)
    vi = VI(transitions = mdp[0],
        reward = mdp[1],
        discount = d,
        epsilon = c,
        max_iter = m)
    vi.run()
        
    return vi.iter

def randMDPiteratePI(s, a, d, m):
    mdp = randMDP(s, a)
    pi = PI(transitions = mdp[0],
                reward = mdp[1],
                discount = d,
                max_iter = m,
                eval_type = 'iterative')
    pi.run()
    return pi.iter

discounts = [0.99,0.999]
#convergence_criteria = [0.01,0.001,0.0001]
convergence_criteria = [0.001,0.0001]
MAX_ITERATIONS = 10000


for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for discount in discounts:
        print('Discount:', discount,'--------------------------')
        states = np.linspace(0,500, 21)
        actions = np.linspace(0,500, 21)
        iters = []
        for s in states:
            s = int(max(2,s))
            print('State:', s)
            for a in actions:
                a = max(2,a)
                act = int(min(s,a))
                #print(s)
                iters.append(randMDPiterateVI(s,act, discount, MAX_ITERATIONS, cc))
                

        fig, ax = plt.subplots()
        s, a = np.meshgrid(states, actions)
        i = np.array(iters).reshape(21,21)
        c = ax.pcolormesh(s, a, i)
        fig.colorbar(c, ax = ax)
        plt.xlabel("Number of States")
        plt.ylabel("Number of Actions")
        plt.legend()
        s = ''.join(['Value Iteration: ','Iterations To Converge - ', 'Discount: ', str(discount), ' Convergence: ',str(cc)])
        plt.title(s)
        l = ''.join(['Value_Iteration_','Iterations_To_Converge_', '_Discount_', str(discount),'_Convergence_',str(cc)])
        plt.savefig(''.join([l,'.png']))
        plt.close()
'''
'''
FM_p = 0.1
def forestMDPiterateVI(s, d, m, c, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,FM_p)
    vi = VI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            epsilon = c,
            max_iter = m)
    vi.run()
    return vi.iter

def forestMDPiteratePI(s, d, m, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,p)
    vi = PI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            max_iter = m)
    vi.run()
    return vi.iter

discounts = [0.9,0.99,0.999]
convergence_criteria = [0.01, 0.001,0.0001]
for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for discount in discounts:
        print('Discount:', discount,'--------------------------')
        states = np.linspace(0,500, 21)
        actions = np.linspace(0,500, 21)
        iters = []
        for s in states:
            s = int(max(2,s))
            print('State:', s)
            for a in actions:
                a = max(2,a)
                act = int(min(s,a))
                #print(s)
                iters.append(forestMDPiterateVI(s,act, discount, MAX_ITERATIONS, cc))
                

        fig, ax = plt.subplots()
        s, a = np.meshgrid(states, actions)
        i = np.array(iters).reshape(21,21)
        c = ax.pcolormesh(s, a, i)
        fig.colorbar(c, ax = ax)
        plt.xlabel("Number of States")
        plt.ylabel("Number of Actions")
        plt.legend()
        s = ''.join(['Value Iteration: ','Time To Converge - ', 'Discount: ', str(discount), ' Convergence: ',str(cc)])
        plt.title(s)
        l = ''.join(['Value_Iteration_','Time_To_Converge_', '_Discount_', str(discount),'_Convergence_',str(cc)])
        plt.savefig(''.join([l,'.png']))
        plt.close()
'''
'''
discounts = [0.9]
for discount in discounts:
    print('Discount:', discount,'--------------------------')
    states = np.linspace(0,500, 21)
    actions = np.linspace(0,500, 21)
    iters = []
    for s in states:
        s = int(max(2,s))
        print('State:', s)
        for a in actions:
            a = max(2,a)
            act = int(min(s,a))
            #print(s)
            iters.append(randMDPiteratePI(s,act, discount, MAX_ITERATIONS))
            

    fig, ax = plt.subplots()
    s, a = np.meshgrid(states, actions)
    i = np.array(iters).reshape(21,21)
    c = ax.pcolormesh(s, a, i)
    fig.colorbar(c, ax = ax)
    plt.xlabel("Number of States")
    plt.ylabel("Number of Actions")
    plt.legend()
    s = ''.join(['Policy Iteration: ','Time To Converge - ', 'Discount: ', str(discount)])
    plt.title(s)
    l = ''.join(['Policy_Iteration_','Time_To_Converge_', '_Discount_', str(discount)])
    plt.savefig(''.join([l,'.png']))
    plt.close()
'''
'''
MAX_ITERATIONS = 10000
def randMDPiterateVI(s, a, d, m, c):
    mdp = randMDP(s, a)
    vi = VI(transitions = mdp[0],
        reward = mdp[1],
        discount = d,
        epsilon = c,
        max_iter = m)
    vi.run()
        
    return list(vi.policy)

def randMDPiteratePI(s, a, d, m):
    mdp = randMDP(s, a)
    pi = PI(transitions = mdp[0],
                reward = mdp[1],
                discount = d,
                max_iter = m,
                eval_type = 'iterative')
    pi.run()
    return list(pi.policy)
'''
'''
MAX_ITERATIONS = 10000
def policySim(s, d, m, c):
    mdp = forestMDP(int(s),2,1,0.1)
    
    vi = VI(transitions = mdp[0],
        reward = mdp[1],
        discount = d,
        epsilon = c,
        max_iter = m)
    vi.run()

    pi = PI(transitions = mdp[0],
        reward = mdp[1],
        discount = d,
        max_iter = m,
        eval_type = 'iterative')
    pi.run()

    p = np.array(list(pi.policy))
    v = np.array(list(vi.policy))
    return sum(p==v)/len(p)
    

discounts = [0.9,0.99,0.999]
convergence_criteria = [0.01,0.001,0.0001]
for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for discount in discounts:
        print('Discount:', discount,'--------------------------')
        states = np.linspace(0,100, 21)
        
        sims = []
        for s in states:
            s = int(max(2,s))
            print('State:', s)
            for a in actions:
                a = max(2,a)
                act = int(min(s,a))
                #print(s)
                #v = np.array(randMDPiterateVI(s,act, discount, MAX_ITERATIONS, cc))
                #p = np.array(randMDPiteratePI(s,act, discount, MAX_ITERATIONS))
                sims.append(policySim(s,discount, MAX_ITERATIONS, cc))
                

        fig, ax = plt.subplots()
        s, a = np.meshgrid(states, actions)
        i = np.array(sims).reshape(21,21)
        c = ax.pcolormesh(s, a, i)
        fig.colorbar(c, ax = ax)
        plt.xlabel("Number of States")
        plt.ylabel("Number of Actions")
        plt.legend()
        s = ''.join(['Forest Policy Similarity: ' 'Discount: ', str(discount), ' Convergence: ',str(cc)])
        plt.title(s)
        l = ''.join(['Forest_Policy_Sim_', '_Discount_', str(discount),'_Convergence_',str(cc)])
        plt.savefig(''.join([l,'.png']))
        plt.close()

'''
'''
FM_p = 0.1
def forestMDPiterateVI(s, d, m, c, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,FM_p)
    vi = VI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            epsilon = c,
            max_iter = m)
    vi.run()
    return vi.iter

def forestMDPiteratePI(s, d, m, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,p)
    vi = PI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            max_iter = m)
    vi.run()
    return vi.iter

#P2 - Forest Management
#mdp = forestMDP(S = FM_states, r1 = FM_r1, r2 = FM_r2, p = FM_p) 
discounts = [0.9]
convergence_criteria = [0.01]
m = 10000
iters = []
for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for d in discounts:
        print('Discount:', d,'--------------------------')
        probs = np.linspace(0,1, 21)
        states = np.linspace(2,102, 21)
        iters = []
        for s in states:
            for p in probs:
                r1 = 1.0
                r2 = 2.0
                i = forestMDPiteratePI(s, d, m, r1, r2, p)
                iters.append(i)
                

        fig, ax = plt.subplots()
        s, a = np.meshgrid(states, probs)
        i = np.array(iters).reshape(21,len(probs))
        c = ax.pcolormesh(s, a, i)
        fig.colorbar(c, ax = ax)
        plt.xlabel("Number of States")
        plt.ylabel("Reward Ratio, Rw/Rx")
        plt.legend()
        s = ''.join(['Probability Impact Forest MDP Policy Iteration: ','Iterations To Converge - ', 'Discount: ', str(d), ' Convergence: ',str(cc)])
        plt.title(s)
        l = ''.join(['Prob_Forest_Policy_Iteration_','Iterations_To_Converge_', '_Discount_', str(d),'_Convergence_',str(cc)])
        plt.savefig(''.join([l,'.png']))
        plt.close()
'''

'''
def forestMDPiterateQL(s, d, m, rw, rc) :
    mdp = forestMDP(int(s),rw,rc)
    ql = QL(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            n_iter = m)
    ql.run()
    return max(ql.V)

def forestMDPiterateVI(s, d, m, c, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,p)
    vi = VI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            epsilon = c,
            max_iter = m)
    vi.run()
    return max(vi.V)

def forestMDPiteratePI(s, d, m, rw, rc, p):
    mdp = forestMDP(int(s),rw,rc,p)
    pi = PI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            max_iter = m)
    pi.run()
    return max(pi.V)

discounts = [0.999]
convergence_criteria = [0.0001]

pi_max = []
vi_max = []
ql_max = []

m = 10000
for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for d in discounts:
        print('Discount:', d,'--------------------------')
        states = np.linspace(2,102, 21)
        for s in states:
            print('State:', s)
            pi_max.append(forestMDPiteratePI(s, d, 10000, 2, 1, 0.1))
            vi_max.append(forestMDPiterateVI(s, d, 10000, cc, 2, 1, 0.1))
            ql_max.append(forestMDPiterateQL(s, d, 10000, 2, 1))

        plt.plot(states, pi_max, label = "PI")
        plt.plot(states, vi_max, label = "VI")
        plt.plot(states, ql_max, label = "QL")
        plt.xlabel("Number of States")
        plt.ylabel("Max Value")
        plt.legend()
        plt.title("Forest Management Max Value - STRICT")
        plt.savefig('STRICT_Forest_Management_Max_Value.png')
        plt.close()
'''
'''
def randMDPmaxValues(s, a, d, m, c) :
    mdp = forestMDP(int(s),2,1)
    ql = QL(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            n_iter = m)
    ql.run()

    vi = VI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            epsilon = c,
            max_iter = m)
    vi.run()
    
    pi = PI(transitions = mdp[0],
            reward = mdp[1],
            discount = d,
            max_iter = m)
    pi.run()
    
    #return max(pi.V), max(vi.V), max(ql.V)
    viql_sim = sum(np.array(vi.policy)==np.array(ql.policy))/s
    piql_sim = sum(np.array(pi.policy)==np.array(ql.policy))/s
    pivi_sim = sum(np.array(pi.policy)==np.array(vi.policy))/s

    return viql_sim, piql_sim, pivi_sim


discounts = [0.999]
convergence_criteria = [0.0001]

pi_max = []
vi_max = []
ql_max = []

m = 10000
for cc in convergence_criteria:
    print("CC:",cc,"+++++++++++++++++++++++++++++++++++")
    for d in discounts:
        print('Discount:', d,'--------------------------')
        states = np.linspace(2,502, 21)
        for s in states:
            print('State:', s)
            p, v, q = randMDPmaxValues(int(s),int(s),d, m, cc)
            pi_max.append(p)
            vi_max.append(v)
            ql_max.append(q)

        plt.plot(states, pi_max, label = "VI-QL Similarity")
        plt.plot(states, vi_max, label = "PI-QL Similarity")
        plt.plot(states, ql_max, label = "PI-VI Similatity")
        plt.xlabel("Number of States")
        plt.ylabel("Similiarity")
        plt.legend()
        plt.title("Forest MDP Policy Similarity - STRICT")
        plt.savefig('STRICT_SIM_Forest_MDP_Max_Value.png')
        plt.close()
'''
