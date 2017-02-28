package solver;

import java.io.IOException;
import java.util.*;

import problem.Fridge;
import problem.Matrix;
import problem.ProblemSpec;

public class SeanMSolver implements OrderingAgent {
	/*
	 * This solver uses two different methods, depending upon the size of the fridge:
	 * 
	 * For smaller fridges, we use policy iteration for an MDP with:
	 *  - state space = all possible configs of fridge
	 *  - action space = all possible maximum purchases from s
	 *  - transition function = chance of user putting us in state s' after action a on s
	 *  - reward = expected cost of using a at state s
	 *  - policy starting with greedy purchases matching immediate best option
	 *  - value function starting at estimated greedy cost for each state
	 *  
	 * For bigger fridges we use similar to the above, but for monte carlo tree search,
	 * allowing for online calculation.
	 */
	private ProblemSpec spec;
	private Fridge fridge;
	private List<Matrix> probabilities;
	
	/*
	 * COMMON
	 */
	private final boolean isBigFridge; // We only use monte carlo for bigger fridges
	
	// constants
	private final double futureDiscount; // How much to favour current vs future tests
	
	// randomisers
	private Random r;
	private List<Integer> randomList;
	
	// Pregenerated/persistent values to save on calculations
	private final List<List<Double>> expectedWants; // Expected wants for j values of type i
	private final List<List<Double>> expectedFails; // Same as the above, but with failures
	
	/*
	 * POLICY ITERATION
	 */
	private List<List<Integer>> possibleStates; // All possible states (BIG)
	// Current value/policy info - null results meaning not yet calculated
	// S->A; non-listed are assumed to be greedy action
	private final Map<List<Integer>, List<Integer>> curPolicy;
	// Value for given state
	private final Map<List<Integer>, Double> valueFunc;
	// Reward for given result state (expected cost from this state)
	private final Map<List<Integer>, Double> reward;
	// Mapping of result state to a list of possible outcomes states from this state,
	// along with the probability of each outcome occurring (sums to 1).
	// if null is returned, means not yet calculated
	private final Map<List<Integer>, Map<List<Integer>, Double>> transitionInfo;
	
	/*
	 * MONTE CARLO
	 */
	// Balance between explore and exploit
	private final double cBalance = 40;
	//Level needed for factor to be less than threshold
	private int levelNeeded;
	// Known number of max actions from state
	private final Map<List<Integer>, Long> actionNumMax;
	// Current tree status
	private MCTSTreeInfo<List<Integer>, List<Integer>> treeInfo;
	
	// Constructor
	public SeanMSolver(ProblemSpec spec) throws IOException {
		System.out.println();
		System.out.println("SOLVER: COMP3702 Assignment 2 - Sean Manson 42846413");
		System.out.println("SOLVER: Initialising...");
		
		// Pointers
		this.spec = spec;
		fridge = spec.getFridge();
		probabilities = spec.getProbabilities();
		
		// Fridge type
		if (fridge.getName().equals("tiny") || fridge.getName().equals("small") ||
				fridge.getName().equals("medium")) {
			isBigFridge = false;
		} else {
			isBigFridge = true;
		}
		
		// Randomisers
		r = new Random();
		randomList = new ArrayList<>();
		for (int i = 0; i < fridge.getMaxTypes(); i++) {
			randomList.add(i);
		}
		
		// Other
		futureDiscount = spec.getDiscountFactor();
		
		// Persistent precalculated values, to save time
		expectedWants = new ArrayList<List<Double>>();
		expectedFails = new ArrayList<List<Double>>();
		for (int k = 0; k < fridge.getMaxTypes(); k++) {
			List<Double> expected = new ArrayList<Double>();
			List<Double> fails = new ArrayList<Double>();
			for (int i = 0; i < fridge.getCapacity(); i++) {
				List<Double> prob = probabilities.get(k).getRow(i);
				double expectedNum = 0;
				for (int j = 1; j < prob.size(); j++) {
					expectedNum += j * prob.get(j);
				}
				expected.add(expectedNum);
				
				double diff = expectedNum - i;
				fails.add((diff > 0) ? diff : 0);
			}
			expectedWants.add(expected);
			expectedFails.add(fails);
		}
		
		// Avoid wasting memory
		if (!isBigFridge) {
			// Policy iteration things
			curPolicy = new HashMap<>();
			valueFunc = new HashMap<>();
			reward = new HashMap<>();
			transitionInfo = new HashMap<>();
			actionNumMax = null;
		} else {
			// Monte Carlo things
			curPolicy = null;
			valueFunc = null;
			reward = null;
			transitionInfo = null;
			actionNumMax = new HashMap<>();
			levelNeeded = spec.getNumWeeks();
		}
		
	}
	
	// Public methods (implement ordering agent)
	public void doOfflineComputation() {
		System.out.println();
		System.out.println("SOLVER: Computing offline portion of solver");
		long startTime = System.currentTimeMillis();
		List<Integer> empty = new ArrayList<>(Collections.nCopies(fridge.getMaxTypes(), 0));
		
		if (!isBigFridge) {
			System.out.println("SOLVER: Medium or smaller fridge - using policy iteration");
			
			// Build list of states
			System.out.println("SOLVER: Beginning offline iteration process");
			possibleStates = possibleStatesGen();
			
			// Begin iteration
			doPolicyIteration(empty, 5 * 60 * 1000 - 20);
		} else {
			System.out.println("SOLVER: Large/Super fridge - using monte carlo search");
			
			// Compute and generate for an empty inventory for a while
			System.out.println("SOLVER: The AI will now use offline time to prepare a wise initial choice. This process takes about 4 minutes. Feel free to sit back and relax.");
			startTime = System.currentTimeMillis();
			treeInfo = new MCTSTreeInfo<>(cBalance);
			monteCarloTreeSearch(empty, 4 * 60 * 1000 - 20);
		}
		
		// Fin
		System.out.println("SOLVER: Offline computation completed. (Took " + (System.currentTimeMillis() - startTime)/1000. + " secs)");
	}
	
	public List<Integer> generateShoppingList(List<Integer> inventory, int numWeeksLeft) {
		System.out.println();
		System.out.println("SOLVER: Generating shopping list");
		long startTime = System.currentTimeMillis();
		List<Integer> shopping;
		
		if (!isBigFridge) {
			// Policy iterate for a while for this state
			doPolicyIteration(inventory, 59 * 1000);
			
			// Return best policy
			shopping = getCurPolicyFor(inventory);
		} else {
			System.out.println("SOLVER: Running online monte carlo tree search for 59 seconds...");
			
			// Update info/reset tree
			levelNeeded = numWeeksLeft + 1;
			if (numWeeksLeft != spec.getNumWeeks()) {
				treeInfo = new MCTSTreeInfo<>(cBalance);
			}
			
			// Get a tree search
			shopping = monteCarloTreeSearch(inventory, 59 * 1000);
		}
		
		// Fin
		System.out.println("SOLVER: Generated. (Took " + (System.currentTimeMillis() - startTime)/1000. + " secs)");
		return shopping;
	}
	
	
	
	
	/* 
	 * ------------------------------------------------------------------------
	 * 	Policy Iteration functions
	 * ------------------------------------------------------------------------
	 */
	
	/*
	 * Generates a list of all possible states
	 */
	private List<List<Integer>> possibleStatesGen() {
		List<List<Integer>> statesPossible = new ArrayList<>();
		
		// Iterate through, adding new list each time
		List<Integer> curState = new ArrayList<>(Collections.nCopies(fridge.getMaxTypes(), 0));
		while (true) {
			// Increment
			int incIndex = 0;
			int valHere;
			boolean fin = true;
			do {
				if (incIndex >= curState.size()) {
					// We've gone through all possible values, return
					return statesPossible;
				}
				
				// Increment
				valHere = curState.get(incIndex);
				if (valHere == fridge.getMaxItemsPerType()) {
					curState.set(incIndex, 0);
					fin = false;
				} else {
					curState.set(incIndex, valHere + 1);
					fin = true;
				}
				incIndex++;
			} while (!fin);
			
			// Add this state if sum is right
			if (sumList(curState) <= fridge.getCapacity())
				statesPossible.add(new ArrayList<>(curState));
		}
	}
	
	/*
	 * Runs a full policy iteration process for the given amount of time,
	 * starting from the given state.
	 */
	private void doPolicyIteration(List<Integer> state, long timeout) {
		long endTime = System.currentTimeMillis() + timeout;
		
		// We endlessly try these until we get a consistent result, or time runs out
		while (true) {
			// Policy evaluation
			
			// Policy evaluation
			for (List<Integer> s : possibleStates) {
				if (System.currentTimeMillis() >= endTime)
					return;
				
				// Update value
				List<Integer> policy = getCurPolicyFor(s);
				updateVal(s, policy);
			}
			
			// Policy improvement
			boolean updatedThisRun = false;
			for (List<Integer> s : possibleStates) {
				// Ensure time constraint
				if (System.currentTimeMillis() >= endTime)
					return;
				
				// Update policy
				if (updatePol(s)) {
					updatedThisRun = true;
				}
			}
			
			// If finished, exit
			if (!updatedThisRun)
				return;
		}
	}
	
	/*
	 * Updates the value of the given state with given policy. (bellman update)
	 * 
	 * Returns diff of change to value
	 */
	private double updateVal(List<Integer> state, List<Integer> policy) {
		// Get result state of performing this policy
		List<Integer> resultState = new ArrayList<Integer>(state.size());
		for (int i = 0; i < state.size(); i++) {
			resultState.add(state.get(i) + policy.get(i));
		}
		
		// Find reward of this result state
		double r = getRewardFor(resultState);
		
		// Have we set this value before? if not, we're done
		Double prevVal = valueFunc.get(state);
		if (prevVal == null) {
			valueFunc.put(state, r);
			return Double.MAX_VALUE;
		}
		
		// Otherwise, need to calculate value
		Map<List<Integer>, Double> transitions = getTransitionsFrom(resultState);
		double valSum = 0.;
		for (Map.Entry<List<Integer>, Double> t : transitions.entrySet()) {
			// For each reachable state s' with probability
			valSum += t.getValue() * getValueFor(t.getKey());
		}
		
		// Get final value
		double newVal = r + futureDiscount * valSum;
		valueFunc.put(state, newVal);
		
		return Math.abs(newVal - prevVal);
	}
	
	
	/*
	 * Updates the policy for the given state by looking one step ahead
	 * and choosing the best action (out of all actions possible from this state)
	 * 
	 * Returns true if updated
	 */
	private boolean updatePol(List<Integer> state) {
		List<Integer> bestAction = null;
		Double bestReward = null;
		for (List<Integer> action : getPossibleActionsFrom(state)) {
			// Get result
			List<Integer> resultState = new ArrayList<Integer>(state.size());
			for (int i = 0; i < state.size(); i++) {
				resultState.add(state.get(i) + action.get(i));
			}
			
			// Value for result
			double r = getRewardFor(resultState);
			
			// One step ahead
			Map<List<Integer>, Double> transitions = getTransitionsFrom(resultState);
			double valSum = 0.;
			for (Map.Entry<List<Integer>, Double> t : transitions.entrySet()) {
				// For each reachable state s' with probability
				valSum += t.getValue() * getValueFor(t.getKey());
			}
			
			// Get final value
			double newRew = r + futureDiscount * valSum;
			if (bestReward == null || newRew > bestReward) {
				bestReward = newRew;
				bestAction = action;
			}
		}
		
		// Update
		if (bestAction == null)
			return false;
		List<Integer> prevPol = getCurPolicyFor(state);
		curPolicy.put(state, bestAction);
		
		return prevPol == null || !prevPol.equals(bestAction);
	}
	
	/*
	 * Gets the value for the given state using the current policy, setting
	 * up a new value if not defined.
	 */
	private double getValueFor(List<Integer> state) {
		// Get policy for this state
		List<Integer> policy = getCurPolicyFor(state);
		
		// Get result state of performing this policy
		List<Integer> resultState = new ArrayList<Integer>(state.size());
		for (int i = 0; i < state.size(); i++) {
			resultState.add(state.get(i) + policy.get(i));
		}
		
		// Get value
		Double val = valueFunc.get(resultState);
		if (val == null) {
			val = getRewardFor(resultState);
			valueFunc.put(state, val);
		}
		return val;
	}
	
	/*
	 * Returns current policy for given state.
	 */
	private List<Integer> getCurPolicyFor(List<Integer> state) {
		List<Integer> policy = curPolicy.get(state);
		if (policy == null) {
			policy = getGreedyAction(state, neededToFill(state));
			curPolicy.put(state, policy);
		}
		
		return policy;
	}
	
	/*
	 * Returns reward for given result state, setting it to its value as the
	 * expected cost of the state.
	 */
	private double getRewardFor(List<Integer> resultState) {
		Double r = reward.get(resultState);
		if (r == null) {
			// Cost is given by spec cost * number of expected failures
			double fails = 0;
			for (int k = 0; k < resultState.size(); k++) {
				int i = resultState.get(k);
				fails += expectedFails.get(k).get(i);
			}
			r = spec.getCost() * fails;
		}
		return r;
	}
	
	/*
	 * Returns and populates transition info with possible transitions from
	 * the given state with probs.
	 */
	private Map<List<Integer>, Double> getTransitionsFrom(List<Integer> resultState) {
		Map<List<Integer>, Double> transitions = transitionInfo.get(resultState);
		if (transitions != null) {
			return transitions;
		}
		
		// Need to populate transitions
		transitions = new HashMap<>();
		List<Integer> curFinalState = new ArrayList<>(Collections.nCopies(resultState.size(), 0));
		
		// Go through all possible final states and find their properties
		while (true) {
			// Find overall probability of getting this final state
			double totalProb = 1.0;
			for (int k = 0; k < curFinalState.size(); k++) {
				// Find probability of having this many wants
				int remaining = curFinalState.get(k);
				int wants = resultState.get(k) - remaining;
				double prob = probabilities.get(k).getRow(resultState.get(k)).get(wants);
				
				// Add probability of more wants if empty
				if (remaining == 0) {
					for (int moreWants = wants + 1; moreWants <= fridge.getMaxItemsPerType(); moreWants++) {
						prob += probabilities.get(k).getRow(resultState.get(k)).get(moreWants);
					}
				}
				totalProb *= prob;
			}
			
			// Put in list if possible transition
			if (totalProb > 0) {
				transitions.put(new ArrayList<Integer>(curFinalState), totalProb);
			}
			
			// Increment
			int incIndex = 0;
			int numHere;
			boolean fin = true;
			do {
				if (incIndex >= curFinalState.size()) {
					// We've gone through all possible values, populate and return
					transitionInfo.put(resultState, transitions);
					return transitions;
				}
				
				numHere = curFinalState.get(incIndex);
				if (numHere >= resultState.get(incIndex)) {
					curFinalState.set(incIndex, 0);
					fin = false;
				} else {
					curFinalState.set(incIndex, numHere + 1);
					fin = true;
				}
				incIndex++;
			} while (!fin);
		}
	}
	
	/*
	 * Returns all actions possible to make from a given state.
	 * We only consider actions buying as many items as possible.
	 */
	private List<List<Integer>> getPossibleActionsFrom(List<Integer> state) {
		double needed = neededToFill(state);
		List<List<Integer>> possActions = new ArrayList<>();
		
		// List of maximum values at each spot
		List<Integer> maxHere = new ArrayList<>();
		for (int i = 0; i < state.size(); i++) {
			maxHere.add(fridge.getMaxItemsPerType() - state.get(i));
		}
		
		// Iterate through, adding new list each time
		List<Integer> curAction = new ArrayList<>(Collections.nCopies(state.size(), 0));
		while (true) {
			// Increment
			int incIndex = 0;
			int addingHere;
			boolean fin = true;
			do {
				if (incIndex >= curAction.size()) {
					// We've gone through all possible values, return
					return possActions;
				}
				
				// Add this action if sum is right
				if (sumList(curAction) == needed)
					possActions.add(new ArrayList<>(curAction));
				
				// Increment
				addingHere = curAction.get(incIndex);
				if (addingHere == maxHere.get(incIndex)) {
					curAction.set(incIndex, 0);
					fin = false;
				} else {
					curAction.set(incIndex, addingHere + 1);
					fin = true;
				}
				incIndex++;
			} while (!fin);
		}
	}
	
	
	
	
	/* 
	 * ------------------------------------------------------------------------
	 * 	Monte Carlo functions
	 * ------------------------------------------------------------------------
	 */
	
	/*
	 * Run the monte carlo tree search starting from the given state for the
	 * given number of seconds
	 */
	private List<Integer> monteCarloTreeSearch(List<Integer> state, long timeout) {
		long endTime = System.currentTimeMillis() + timeout;
		
		// Find actions until time runs out
		while (System.currentTimeMillis() < endTime) {
			searchSingleLevel(state, 0);
		}
		
		// Return argmax action of all states
		List<Integer> bestAction = treeInfo.argmax(state);
		if (bestAction == null) { // If no action is found, do nothing (should never happen)
			bestAction = new ArrayList<>(Collections.nCopies(fridge.getMaxTypes(), 0));
		}
		
		// Finished this week
		return bestAction;
	}
	
	/*
	 * Perform a single level of the monte carlo search, updating bestAction
	 * and bestActionReward until
	 */
	private double searchSingleLevel(List<Integer> state, int level) {
		if (level >= levelNeeded) {
			return 0;
		}
		
		// Get all unused actions at this state
		List<Integer> action = getUnusedAction(state);
		if (action != null) {
			// If we have an action as yet unused, run a simulation to estimate
			double estimateForQ = 0;
			for (int i = 0; i < 3; i++) {
				estimateForQ += estimateValue(state, action, level);
			}
			estimateForQ /= 3;
			
			// Set this as the value for making A from S, and tell parent
			treeInfo.setQ(state, action, estimateForQ);
			
			// Increment visits
			treeInfo.incrementN(state);
			treeInfo.incrementN(state, action);
			
			return estimateForQ;
		} else {
			// If all actions are used up, get best action uct
			action = treeInfo.uct(state);
			
			// Get random resulting state by simulating this
			//<resulting state, reward>
			List<Integer> resultState = new ArrayList<Integer>(state.size());
			double resultVal;
			//resultVal = simulateSingleStep(state, action, resultState);
			resultVal = simulateSingleStepExpectedFails(state, action, resultState);
			
			// Set new Q value for this node by searching below
			double oldQ = treeInfo.getQ(state, action);
			double newQ = resultVal + futureDiscount * searchSingleLevel(resultState, level + 1);
			int oldN = treeInfo.getN(state, action);
			newQ = (oldQ * oldN + newQ)/(oldN + 1);
			
			treeInfo.setQ(state, action, newQ);
			
			// Increment visits
			treeInfo.incrementN(state);
			treeInfo.incrementN(state, action);
			
			return newQ;
		}
	}
	
	/*
	 * Returns a random action (list of shopping) from the given state which
	 * has not been used yet, or null if all actions have been used.
	 * 
	 * Possible actions are actions which add up to max types of items filling
	 * up the state to capacity.
	 */
	private List<Integer> getUnusedAction(List<Integer> state) {
		int needed = neededToFill(state);
		
		// See if we've used up all possible actions
		Long possibleActions = actionNumMax.get(state);
		if (possibleActions == null) { // If we haven't looked up this yet
			possibleActions = numActionsFromState(state, needed);
			actionNumMax.put(state, possibleActions);
		}
		if (possibleActions != null && treeInfo.actionNum(state) >= possibleActions) {
			return null;
		}
		
		// Prefer more spread-out actions first
		List<Integer> action = null;
		do {
			action = getRandomGoodAction(state, needed);
		} while (treeInfo.actionUsed(state, action));
		
		return action;
	}
	
	/*
	 * Estimate the value given (actually penalty) by running the given
	 * action from the given state. Runs over N weeks.
	 * 
	 * Most code taken from the simulator.
	 * 
	 * Uses a uniformly random shopping list for its policy.
	 */
	private double estimateValue(List<Integer> state, List<Integer> action, int level) {
		// Get starting fridge contents
		List<Integer> simFridge = new ArrayList<Integer>(state);
		for (int i = 0; i < action.size(); i++) {
			simFridge.set(i, state.get(i) + action.get(i));
		}
		
		// Run simulation for constant weeks
		List<Integer> randomAction;
		List<Integer> wants;
		double totalPenalty = 0;
		for (int week = 0; week < levelNeeded; week++) {
			// For non-initial weeks, fill fridge as neccessary
			if (week != 0) {
				randomAction = getGreedyAction(simFridge, neededToFill(simFridge));
				for (int i = 0; i < action.size(); i++) {
					simFridge.set(i, simFridge.get(i) + randomAction.get(i));
				}
			}
			
			// Get user wants and sample, adding failures
			wants = sampleUserWants(simFridge);
			int numFailures = 0;
			for (int i = 0; i < wants.size(); i++) {
				int net = simFridge.get(i) - wants.get(i);
				if (net < 0) {
					simFridge.set(i, 0);
					numFailures -= net;
				} else {
					simFridge.set(i, net);
				}
			}
			
			// Calculate and add to penalty
			double penalty = spec.getCost() * numFailures;
			totalPenalty += Math.pow(futureDiscount, week) * penalty;
		}
		
		return totalPenalty;
	}
	
	/*
	 * Simulates one step at random from the given state, putting the new state
	 * in the EMPTY list given in result state. Returns the penalty generated
	 * by moving to this state.
	 * 
	 * Steps are simulated by finding the expected number of wants and choosing
	 * the result rounded for these.
	 */
	private double simulateSingleStepExpectedFails(List<Integer> state, List<Integer> action,
			List<Integer> resultState) {
		// Find initial values in intermediate result state
		for (int i = 0; i < state.size(); i++) {
			resultState.add(state.get(i) + action.get(i));
		}
		
		// Find total fails expected for each item
		int numFails = 0;
		for (int k = 0; k < resultState.size(); k++) {
			// Expected wants for this item amount
			int i = resultState.get(k);
			double expectedWant = expectedWants.get(k).get(i);
			int roundExpectedWant = (int) Math.round(expectedWant);
			if (roundExpectedWant <= i) {
				resultState.set(k, i - roundExpectedWant);
			} else {
				resultState.set(k, 0);
				numFails ++;
			}
		}
		
		// Calculate penalty from number of failures
		return spec.getCost() * numFails;
	}
	
	/*
	 * Obtains a random action for the given state, similar to the above,
	 * except without going over the max items used by the user per week.
	 */
	private List<Integer> getRandomGoodAction(List<Integer> state, int needed) {
		List<Integer> action = new ArrayList<>(Collections.nCopies(state.size(), 0));
		
		for (int i = 0; i < needed; i++) {
			Collections.shuffle(randomList);
			for (int randInd : randomList) {
				int valueHere = state.get(randInd) + action.get(randInd);
				if (valueHere < fridge.getMaxItemsPerType()) {
					action.set(randInd, action.get(randInd) + 1);
					break;
				}
			}
		}
		
		return action;
	}
	
	/*
	 * Given a state, calculates the number of actions consisting of needed
	 * items which can be made from that state, without going over the max
	 * types per item.
	 * 
	 * Returns null for 'infinity', meaning far too many numbers to count
	 */
	private Long numActionsFromState(List<Integer> state, int needed) {
		TruncPolynomial genFunction = null;
		if (needed == 0) {
			return 1L;
		}
		
		// Return infinity once numbers are too big to ever count
		if (needed + state.size() > 30) {
			return null;
		}
		
		// For each item
		for (int item : state) {
			// Get how many can be filled in this slot
			int numFillable = fridge.getMaxItemsPerType() - item;
			if (numFillable <= 0) {
				continue;
			}
			
			// Create a polynomial of all 1 with length matching the space here
			TruncPolynomial thisItemGen = new TruncPolynomial(needed + 1);
			thisItemGen.setCoeff(1, numFillable + 1);
			if (genFunction == null) {
				genFunction = thisItemGen;
			} else {
				genFunction = genFunction.multiply(thisItemGen);
			}
		}
		
		// Actions should be the coefficient for the needed term
		return genFunction.getCoeff(needed);
	}
	
	// THE FOLLOWING ARE TAKEN FROM Simulator.java
	/**
	 * Uses the currently loaded stochastic model to sample user wants.
	 * Note that user wants may exceed the inventory
	 * @param state The inventory
	 * @return User wants as list of item quantities
	 */
	public List<Integer> sampleUserWants(List<Integer> state) {
		List<Integer> wants = new ArrayList<Integer>();
		for (int k = 0; k < fridge.getMaxTypes(); k++) {
			int i = state.get(k);
			List<Double> prob = probabilities.get(k).getRow(i);
			wants.add(sampleIndex(prob));
		}
		return wants;
	}
	
	/**
	 * Returns an index sampled from a list of probabilities
	 * @precondition probabilities in prob sum to 1
	 * @param prob
	 * @return an int with value within [0, prob.size() - 1]
	 */
	public int sampleIndex(List<Double> prob) {
		double sum = 0;
		double rand = r.nextDouble();
		for (int i = 0; i < prob.size(); i++) {
			sum += prob.get(i);
			if (sum >= rand) {
				return i;
			}
		}
		return -1;
	}
	
	
	
	
	/* 
	 * ------------------------------------------------------------------------
	 * 	Common helper methods
	 * ------------------------------------------------------------------------
	 */
	
	/*
	 * Gets a greedy action - one which minimises the immediate expected fails.
	 */
	private List<Integer> getGreedyAction(List<Integer> state, int needed) {
		List<Integer> action = new ArrayList<>(Collections.nCopies(state.size(), 0));
		
		// Do the following for as many as we need
		for (int a = 0; a < needed; a++) {
			double highestFailDecrease = 0; // Greatest decrease in expected fails
			int highestFailDecreaseInd = -1;
			
			// For all items
			for (int k = 0; k < state.size(); k++) {
				if (state.get(k) >= fridge.getMaxItemsPerType()) {
					continue;
				}
				
				// Find the probabilities of failure with and without
				int numCur = state.get(k) + action.get(k);
				
				// Get expected number of fails for this and when incremented
				double failsCur = expectedFails.get(k).get(numCur);
				double failsNew = expectedFails.get(k).get(numCur + 1);
				
				double failNumddiff = failsCur - failsNew;
				if (failNumddiff > highestFailDecrease) {
					highestFailDecrease = failNumddiff;
					highestFailDecreaseInd = k;
				}
			}
			
			if (highestFailDecreaseInd != -1) {
				action.set(highestFailDecreaseInd, action.get(highestFailDecreaseInd) + 1);
			} else {
				// Get random fillable index, and add 1 to it
				Collections.shuffle(randomList);
				int indexToBuy = 0;
				for (int randInt : randomList) {
					if (action.get(randInt) < fridge.getMaxItemsPerType()) {
						indexToBuy = randInt;
						break;
					}
				}
				action.set(indexToBuy, action.get(indexToBuy) + 1);
			}
		}
		
		return action;
	}
	
	/*
	 * Returns the amount needed to fill the given fridge to capacity
	 * Requires a list containing less than the capacity of the fridge.
	 */
	private int neededToFill(List<Integer> list) {
		int needed =  fridge.getCapacity() - sumList(list);
		return (needed < fridge.getMaxPurchase()) ? needed : fridge.getMaxPurchase();
	}
	
	/*
	 * Sums all values in the given list.
	 */
	private int sumList(List<Integer> list) {
		int sum = 0;
		for (int i : list) {
			sum += i;
		}
		
		return sum;
	}
	
	
	
	/* 
	 * ------------------------------------------------------------------------
	 * 	Extra classes for monte carlo
	 * ------------------------------------------------------------------------
	 */
	
	// The current state of the performing monte carlo search tree
	private class MCTSTreeInfo<S, A> {
		// Q(s, a) - average estimated reward of performing a at s
		private final Map<Tuple<S, A>, Double> q;
		private final Map<S, Set<A>> q2; // All actions tested on S so far
		
		//c constant used for balancing exploration with exploitation
		private final double cBalance;
		
		// n(s) - number of times s has been visited
		private final Map<S, Integer> n;
		// n(s, a) - number of times a has been performed from s
		private final Map<Tuple<S, A>, Integer> n2;
		
		
		public MCTSTreeInfo(double cBalance) {
			this.q = new HashMap<>();
			this.q2 = new HashMap<>();
			this.n = new HashMap<>();
			this.n2 = new HashMap<>();
			this.cBalance = cBalance;
		}
		
		/*public void debugPrint() {
			System.out.println("Printing current tree status:");
			for (S state : q2.keySet()) {
				for (A action : q2.get(state)) {
					System.out.println("Q(" + state + ", " + action + ") = " + q.get(new Tuple<S, A>(state, action)));
				}
				System.out.println();
			}
		}
		
		public void debugPrint(S state) {
			System.out.println("Printing current tree status for this state:");
			Set<A> s = q2.get(state);
			if (s != null) {
				for (A action : s) {
					System.out.println("Q(" + state + ", " + action + ") = " + q.get(new Tuple<S, A>(state, action)));
				}
			}
			
			System.out.println();
		}*/
		
		public void setQ(S state, A action, double value) {
			q.put(new Tuple<S, A>(state, action), value);
			
			// Ensure action is in the set of all applied actions
			Set<A> actions = q2.get(state);
			if (actions == null) {
				actions = new HashSet<A>();
				q2.put(state, actions);
			}
			actions.add(action);
		}
		
		public void incrementN(S state) {
			Integer prev = n.get(state);
			n.put(state, (prev == null) ? 1 : prev + 1);
		}
		
		public void incrementN(S state, A action) {
			Tuple<S, A> tup = new Tuple<>(state, action);
			Integer prev = n2.get(tup);
			n2.put(tup, (prev == null) ? 1 : prev + 1);
		}
		
		// Number of actions stemming from given state
		public long actionNum(S state) {
			Set<A> actions = q2.get(state);
			if (actions == null) {
				return 0;
			}
			
			return actions.size();
		}
		
		// Whether the given action was ever tested for the given state
		public boolean actionUsed(S state, A action) {
			Set<A> actions = q2.get(state);
			if (actions == null) {
				return false;
			}
			
			return actions.contains(action);
		}
		
		public Double getQ(S state, A action) {
			return q.get(new Tuple<S, A>(state, action));
		}
		
		public Integer getN(S state) {
			Integer ret = n.get(state);
			if (ret == null) {
				return 0;
			} else {
				return ret;
			}
		}
		
		public Integer getN(S state, A action) {
			Integer ret = n2.get(new Tuple<S, A>(state, action));
			if (ret == null) {
				return 0;
			} else {
				return ret;
			}
		}
		
		// Returns the action with the greatest Q(s, a) for given state
		public A argmax(S state) {
			Set<A> actions = q2.get(state);
			if (actions == null) {
				return null;
			}
			
			A bestAction = null;
			Double bestActionValue = null;
			for (A action : actions) {
				Double thisVal = q.get(new Tuple<S, A>(state, action));
				
				if (bestActionValue == null || thisVal > bestActionValue) {
					bestAction = action;
					bestActionValue = thisVal;
				}
			}
			
			return bestAction;
		}
		
		/*
		 * Returns the upper confidence bound at the given state (best action
		 * which has the highest value given by the thing we had in lectures)
		 */
		public A uct(S state) {
			Set<A> actions = q2.get(state);
			if (actions == null) {
				return null;
			}
			
			A bestAction = null;
			Double bestActionValue = null;
			for (A action : actions) {
				// Exploit value at this point
				double exploit = q.get(new Tuple<S, A>(state, action));
				
				// Explore value at this point
				double explore = cBalance * Math.sqrt(Math.log(getN(state))/getN(state, action));
				
				// Choose best value
				double total = exploit + explore;
				if (bestActionValue == null || total > bestActionValue) {
					bestAction = action;
					bestActionValue = total;
				}
			}
			
			return bestAction;
		}
	}
	
	/*
	 * A polynomial 1 + x + x^2 + ... up to x^n, which is used to represent
	 * gaps in the graph. This has the ability to only count up to n values.
	 */
	private class TruncPolynomial {
		private final int maxLen;
		private final long coefficients[];
		TruncPolynomial(int maxLen) {
			this.maxLen = maxLen;
			this.coefficients = new long[maxLen];
		}
		
		// Sets the first k elements to i
		public void setCoeff(int i, int k) {
			for (int j = 0; j < Math.min(k, maxLen); j++) {
				coefficients[j] = i;
			}
		}
		
		public long getCoeff(int i) {
			return coefficients[i];
		}
		
		// Multiply this by another polynomial, returning truncated
		public TruncPolynomial multiply(TruncPolynomial other) {
			TruncPolynomial result = new TruncPolynomial(this.maxLen);
			
			for (int i = 0; i < Math.min(other.maxLen, maxLen); i++) {
				for (int j = 0; j < maxLen - i; j++) {
					result.coefficients[i + j] += other.coefficients[i] * this.coefficients[j];
				}
			}
			
			return result;
		}
		
		@Override
		public String toString() {
			String res = "" + this.coefficients[0];
			for (int i = 1; i < this.coefficients.length; i++) {
				res += " + " + this.coefficients[i] + ((i == 1) ? "x" : "x^" + i);
			}
			return res;
		}
		
		@Override
		public boolean equals(Object o) {
			if (!(o instanceof TruncPolynomial)) {
				return false;
			}
			
			TruncPolynomial other = (TruncPolynomial)o;
			return this.coefficients.equals(other.coefficients);
		}
		
		@Override
		public int hashCode() {
			return this.coefficients.hashCode();
		}
	}
	
	/*
	 * A tuple for any two generic classes
	 */
	private class Tuple<X, Y> {
		public final X a;
		public final Y b;
		public Tuple(X a, Y b) {
			this.a = a;
			this.b = b;
		}
		
		@Override
		public boolean equals(Object o) {
			if (!(o instanceof Tuple)) {
				return false;
			}
			
			Tuple<?, ?> other = (Tuple<?, ?>) o;
			return this.a.equals(other.a) && this.b.equals(other.b);
		}
		
		@Override
		public int hashCode() {
			return this.a.hashCode() + this.b.hashCode() * 7;
		}
	}
}
