# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for _ in range(self.iterations):
            states = self.mdp.getStates()
            new_values = util.Counter()
            for s in states:
                if(self.mdp.isTerminal(s)):
                    new_values[s] = 0
                else:
                    actions = self.mdp.getPossibleActions(s)
                    maxQ = float('-inf')
                    for a in actions:
                        maxQ = max(maxQ, self.getQValue(s, a))
                    new_values[s] = maxQ
            self.values = new_values.copy()
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        probs = self.mdp.getTransitionStatesAndProbs(state, action)
        sum_probs = 0
        for (nextState, prob) in probs:
            sum_probs += prob * (self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])
        return sum_probs

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if(self.mdp.isTerminal(state)):
            return None
        actions = self.mdp.getPossibleActions(state)
        QValues = util.Counter()
        for a in actions:
            QValues[a] = self.getQValue(state, a)
        return QValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        i = 0
        for _ in range(self.iterations):
            s = states[i % len(states)]
            if(self.mdp.isTerminal(s)):
                self.values[s] = 0
            else:
                actions = self.mdp.getPossibleActions(s)
                maxQ = float('-inf')
                for a in actions:
                    maxQ = max(maxQ, self.getQValue(s, a))
                self.values[s] = maxQ
            i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.predecessors = dict()
        for s in mdp.getStates():
            self.predecessors[s] = set()
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors of all states
        states = self.mdp.getStates()
        for state in states:
            for state_pred in states:
                actions = self.mdp.getPossibleActions(state_pred)
                for a in actions:
                    probs = self.mdp.getTransitionStatesAndProbs(state_pred, a)
                    for (nextState, prob) in probs:
                        if(prob > 0 and nextState == state):
                            self.predecessors[state].add(state_pred)

        # Initialize empty priority queue
        q = util.PriorityQueue()

        # Iterate over non terminan states
        for state in states:
            if(not self.mdp.isTerminal(state)):
                actions = self.mdp.getPossibleActions(state)
                diff = abs(self.getValue(state) - max(self.getQValue(state, a) for a in actions))
                q.push(state, -1*diff)
        i = 0
        while(i < self.iterations and not q.isEmpty()):
            s = q.pop()
            # Update value of s
            if(not self.mdp.isTerminal(s)):
                actions = self.mdp.getPossibleActions(s)
                maxQ = float('-inf')
                for a in actions:
                    maxQ = max(maxQ, self.getQValue(s, a))
                self.values[s] = maxQ
            for p in self.predecessors[s]:
                actions = self.mdp.getPossibleActions(p)
                diff = abs(self.getValue(p) - max(self.getQValue(p, a) for a in actions))
                if(diff > self.theta):
                    q.update(p, -1*diff)
            i += 1