# multiAgents.py
# --------------
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


from ast import Tuple
import collections
from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
from game import Agent
import heapq


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == "Stop":
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            """The larger distance, the lower eval value"""
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if tempDistance > distance:
                distance = tempDistance

        """eval value is the largest distance"""
        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="betterEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        pacmanAgentIndex = 0
        legal_actions = gameState.getLegalActions(pacmanAgentIndex)
        max_utility = float("-inf")
        max_action = None
        scores = []
        for action in legal_actions:
            successor = gameState.generateSuccessor(pacmanAgentIndex, action)
            current_utility = self.value(successor, 1, 0)
            if current_utility > max_utility or max_action == None:
                max_utility = current_utility
                max_action = action
            scores.append(current_utility)
        print(
            f"""
        Legal action: {legal_actions}
        Scores: {scores}
        Taken action: {max_action}
        """
        )

        return max_action

    def value(self, state: GameState, agentIndex: int, depth: int):
        """Return current state utility if is terminal"""
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            """Max agent = 0"""
            return self.max_value(state, agentIndex, depth)
        else:
            """Min agent >= 1"""
            return self.min_value(state, agentIndex, depth)

    def max_value(self, state: GameState, agentIndex: int, depth: int):

        legal_actions = state.getLegalActions(agentIndex)
        max_utility = float("-Inf")
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(successor, next_agent_index, next_depth)
            max_utility = max(current_utility, max_utility)

        return max_utility

    def min_value(self, state: GameState, agentIndex: int, depth: int):

        legal_actions = state.getLegalActions(agentIndex)
        min_utility = float("Inf")
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(successor, next_agent_index, next_depth)
            min_utility = min(current_utility, min_utility)

        return min_utility


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        pacmanAgentIndex = 0
        legal_actions = gameState.getLegalActions(pacmanAgentIndex)
        max_utility = float("-inf")
        max_action = None
        scores = []
        initAlpha = float("-inf")
        initBeta = float("inf")
        for action in legal_actions:
            successor = gameState.generateSuccessor(pacmanAgentIndex, action)
            current_utility = self.value(successor, 1, 0, initAlpha, initBeta)
            if current_utility > max_utility or max_action == None:
                max_utility = current_utility
                max_action = action
            scores.append(current_utility)
        print(
            f"""
        Legal action: {legal_actions}
        Scores: {scores}
        Taken action: {max_action}
        """
        )
        # if scores.count(max_utility) == len(legal_actions):
        #     return random.choice(legal_actions)
        return max_action

    def value(self, state: GameState, agentIndex: int, depth: int, alpha, beta):
        """Return current state utility if is terminal"""
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            """Max agent = 0"""
            return self.max_value(state, agentIndex, depth, alpha, beta)
        else:
            """Min agent >= 1"""
            return self.min_value(state, agentIndex, depth, alpha, beta)

    def max_value(self, state: GameState, agentIndex: int, depth: int, alpha, beta):

        legal_actions = state.getLegalActions(agentIndex)
        max_utility = float("-Inf")
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(
                successor, next_agent_index, next_depth, alpha, beta
            )

            max_utility = max(current_utility, max_utility)

            if max_utility >= beta:
                return max_utility
            alpha = max(alpha, max_utility)

        return max_utility

    def min_value(self, state: GameState, agentIndex: int, depth: int, alpha, beta):

        legal_actions = state.getLegalActions(agentIndex)
        min_utility = float("Inf")
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(
                successor, next_agent_index, next_depth, alpha, beta
            )

            min_utility = min(current_utility, min_utility)

            if min_utility <= alpha:
                return min_utility
            beta = min(beta, min_utility)

        return min_utility


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        pacmanAgentIndex = 0
        legal_actions = gameState.getLegalActions(pacmanAgentIndex)
        max_utility = float("-inf")
        max_action = None
        scores = []
        for action in legal_actions:
            successor = gameState.generateSuccessor(pacmanAgentIndex, action)
            current_utility = self.value(successor, 1, 0)
            if current_utility > max_utility or max_action == None:
                max_utility = current_utility
                max_action = action
            scores.append(current_utility)
        print(
            f"""
        Legal action: {legal_actions}
        Scores: {scores}
        Taken action: {max_action}
        """
        )

        return max_action

    def value(self, state: GameState, agentIndex: int, depth: int):
        """Return current state utility if is terminal"""
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            """Max agent = 0"""
            return self.max_value(state, agentIndex, depth)
        else:
            """Min agent >= 1"""
            return self.exp_value(state, agentIndex, depth)

    def max_value(self, state: GameState, agentIndex: int, depth: int):

        legal_actions = state.getLegalActions(agentIndex)
        max_utility = float("-Inf")
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(successor, next_agent_index, next_depth)
            max_utility = max(current_utility, max_utility)

        return max_utility

    def exp_value(self, state: GameState, agentIndex: int, depth: int):

        legal_actions = state.getLegalActions(agentIndex)
        utilities = []
        next_agent_index = agentIndex + 1
        next_depth = depth
        """Reset agent index"""
        if next_agent_index == state.getNumAgents():
            next_agent_index = 0
            next_depth += 1

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=agentIndex, action=action)

            current_utility = self.value(successor, next_agent_index, next_depth)
            utilities.append(current_utility)

        return sum(utilities) / len(utilities)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    score = 0

    """Penalize if lose"""
    if currentGameState.isLose():
        return float("-Inf")

    """
    The ultimate goal is win
    So i'll give it inf score if win
    """
    if currentGameState.isWin():
        score += 10000

    pacmanPos = currentGameState.getPacmanPosition()

    """Calculate closest food distance"""
    currentFoods = currentGameState.getFood()
    foodList = currentFoods.asList()
    closestFood = 0
    if foodList:
        closestFood = min([manhattanDistance(pacmanPos, food) for food in foodList])

    """Calculate closest big dot distance"""
    currentCapsules = currentGameState.getCapsules()
    closestCapsule = 0
    if currentCapsules:
        closestCapsule = min(
            [manhattanDistance(pacmanPos, caps) for caps in currentCapsules]
        )

    noFoodLeft = len(foodList)

    noBigDotLeft = len(currentCapsules)

    # active ghosts are ghosts that aren't scared.
    scaredGhosts, activeGhosts = [], []
    for ghost in currentGameState.getGhostStates():
        if not ghost.scaredTimer:
            activeGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)

    def getManhattanDistances(ghosts):
        return map(
            lambda g: util.manhattanDistance(pacmanPos, g.getPosition()),
            ghosts,
        )

    distanceToClosestScaredGhost = 0
    distanceToClosestActiveGhost = float("inf")
    if activeGhosts:
        distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))

    if distanceToClosestActiveGhost == 0:
        distanceToClosestActiveGhost = float("inf")

    if scaredGhosts:
        distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))

    # (
    #     closestCapsule,
    #     closestFood,
    #     distanceToClosestScaredGhost,
    #     distanceToClosestActiveGhost,
    # ) = closestStuff(currentGameState)

    # if distanceToClosestActiveGhost == 0:
    #     distanceToClosestActiveGhost = float("inf")
    scaredScore = (
        90  # ghost = 200 point
        if distanceToClosestScaredGhost <= 2 and len(scaredGhosts) != 0
        else len(scaredGhosts) * 90
    )
    scores = np.array(
        [
            1 * currentGameState.getScore(),
            -4.1 * closestFood,
            -9.2 * closestCapsule,
            -1.1 * 1 / distanceToClosestActiveGhost,
            -3.4 * distanceToClosestScaredGhost,
            +1 * scaredScore,
            -5.3 * noBigDotLeft,
            -2.6 * noFoodLeft,
        ]
    )
    score += np.sum(scores)

    # if pacmanPos == (2, 5):
    #     a = 3
    # if pacmanPos == (3, 5):
    #     a = 3
    if "205.72499999" in str(score):
        print(
            f"""Game Score: {currentGameState.getScore()}
    Closest Food: {closestFood}
    Closest Capusle: {closestCapsule}
    Closest active ghost: {distanceToClosestActiveGhost}
    Closest scared ghost: {distanceToClosestScaredGhost}
    Number of big dot left: {noBigDotLeft}
    Number of food left: {noFoodLeft}
    Number Scared ghosts: {len(scaredGhosts)}
    Pacman Pos: {pacmanPos}
    """
        )
        print(f"Score1: {score}\n")
    # print(f"Number Scared ghosts: {len(scaredGhosts)}")

    return score


# Abbreviation
better = betterEvaluationFunction


def contestEvaluationFunc(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    def closest_dot(cur_pos, food_pos):
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def closest_ghost(cur_pos, ghosts):
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        num_ghosts = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                num_ghosts += 1
        return scores

    def food_stuff(cur_pos, food_positions):
        food_distances = []
        for food in food_positions:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return sum(food_distances)

    def num_food(cur_pos, food):
        return len(food)

    def closest_capsule(cur_pos, caps_pos):
        capsule_distances = []
        for caps in caps_pos:
            capsule_distances.append(util.manhattanDistance(caps, cur_pos))
        return min(capsule_distances) if len(capsule_distances) > 0 else 9999999

    def scaredghosts(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if (
                ghost.scaredTimer > 8
                and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 4
            ):
                scoreslist.append(scores + 50)
            if (
                ghost.scaredTimer > 8
                and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 3
            ):
                scoreslist.append(scores + 60)
            if (
                ghost.scaredTimer > 8
                and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 2
            ):
                scoreslist.append(scores + 70)
            if (
                ghost.scaredTimer > 8
                and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 1
            ):
                scoreslist.append(scores + 90)
            # if ghost.scaredTimer > 0 and util.manhattanDistance(ghost.getPosition(), cur_pos) < 1:
        #              scoreslist.append(scores + 100)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def ghostattack(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer == 0:
                scoreslist.append(
                    scores - util.manhattanDistance(ghost.getPosition(), cur_pos) - 10
                )
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def scoreagent(cur_pos, food_pos, ghost_states, caps_pos, score):
        if closest_capsule(cur_pos, caps_pos) < closest_ghost(cur_pos, ghost_states):
            return score + 40
        if closest_dot(cur_pos, food_pos) < closest_ghost(cur_pos, ghost_states) + 3:
            return score + 20
        if closest_capsule(cur_pos, caps_pos) < closest_dot(cur_pos, food_pos) + 3:
            return score + 30
        else:
            return score

    capsule_pos = currentGameState.getCapsules()
    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    # score = score * 2 if closest_dot(pacman_pos, food) < closest_ghost(pacman_pos, ghosts) + 3 else score
    # score = score * 1.5 if closest_capsule(pacman_pos, capsule_pos) < closest_dot(pacman_pos, food) + 4 else score
    score += scoreagent(pacman_pos, food, ghosts, capsule_pos, score)
    score += scaredghosts(ghosts, pacman_pos, score)
    score += ghostattack(ghosts, pacman_pos, score)
    score -= 0.35 * food_stuff(pacman_pos, food)
    return score


def closestStuff(initialState: GameState):
    """Calculate the real distance from pacman position to goal position"""
    initPacmanPos = initialState.getPacmanPosition()
    capsulesPos = initialState.getCapsules()
    foodsPos = initialState.getFood().asList()
    scaredGhostsPos, activeGhostsPos = [], []
    for ghost in initialState.getGhostStates():
        temp = ghost.getPosition()
        pos = (int(temp[0]), int(temp[1]))
        if not ghost.scaredTimer:
            activeGhostsPos.append(pos)
        else:
            scaredGhostsPos.append(pos)

    closestCapsule = 0
    closestFood = 0
    closestScaredGhost = 0
    closestActiveGhost = 0

    def assignValue(
        pacmanPos, closestCapsule, closestFood, closestScaredGhost, closestActiveGhost
    ):
        if closestCapsule == 0 and pacmanPos in capsulesPos:
            closestCapsule = manhattanDistance(initPacmanPos, pacmanPos)
        if closestFood == 0 and pacmanPos in foodsPos:
            closestFood = manhattanDistance(initPacmanPos, pacmanPos)
        if closestScaredGhost == 0 and pacmanPos in scaredGhostsPos:
            closestScaredGhost = manhattanDistance(initPacmanPos, pacmanPos)
        if closestActiveGhost == 0 and pacmanPos in activeGhostsPos:
            closestActiveGhost = manhattanDistance(initPacmanPos, pacmanPos)
        return closestCapsule, closestFood, closestScaredGhost, closestActiveGhost

    def isEndState():
        capsuleCheck = foodCheck = scaredCheck = activeCheck = False
        if not capsulesPos or closestCapsule != 0:
            capsuleCheck = True
        if not foodsPos or closestFood != 0:
            foodCheck = True
        if not scaredGhostsPos or closestScaredGhost != 0:
            scaredCheck = True
        if not activeGhostsPos or closestActiveGhost != 0:
            activeCheck = True
        return activeCheck and scaredCheck and foodCheck and capsuleCheck

    pacmanPos = initialState.getPacmanPosition()

    startState = initialState.deepCopy()

    # this queue's element is an array of state transition from the root
    # to the latest node in the  array.
    frontier = collections.deque([[startState]])

    # initialize the actions queue
    # in which element is an array contain the action need to be performed to reach the state
    # correspond to the state which has the same index in frontier queue.
    actions = collections.deque([[0]])

    # initialize the exploredSet which will contain the state tuple
    # that the algorithm explored.
    exploredSet = set()

    # initialize the variable to count the number
    # of states explored
    states_count = 0

    # loop until all states have been explored or result is found.
    while len(frontier) != 0:
        # get the oldest state transition array
        node = frontier.popleft()

        # get the oldest action correspond to the node above
        node_action = actions.popleft()
        currentState = node[-1]
        # extract player position
        pacmanPos = currentState.getPacmanPosition()
        (
            closestCapsule,
            closestFood,
            closestScaredGhost,
            closestActiveGhost,
        ) = assignValue(
            pacmanPos,
            closestCapsule,
            closestFood,
            closestScaredGhost,
            closestActiveGhost,
        )
        # increase the number of states explored
        states_count += 1

        # if the solution is found.
        if isEndState():
            # set the solution equal to the currently exploring node's actions.
            # and break the loop
            break

        # if the current node hasn't been explored
        if currentState not in exploredSet:
            # mark it as explored
            exploredSet.add(currentState)

            # get available actions of the current state
            availableActions = currentState.getLegalActions(0)

            # loop through the availableActions
            for action in availableActions:
                # foreach action, get the next state
                newState = currentState.generateSuccessor(0, action)
                # enqueue new state into frontier
                frontier.append(node + [newState])

                # enqueue the corresponding action into actions
                actions.append(node_action + [action[-1]])

    return closestCapsule, closestFood, closestScaredGhost, closestActiveGhost
