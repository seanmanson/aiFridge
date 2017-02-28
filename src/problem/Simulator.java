package problem;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Simulator {
    private Random random = new Random();

	private int currentWeek;
	private ProblemSpec problemSpec;
	private ArrayList<Integer> inventory;
	private ArrayList<List<Integer>> inventoryHistory;
	private ArrayList<List<Integer>> shoppingHistory;
	private ArrayList<List<Integer>> userWantsHistory;
	private ArrayList<Double> penaltyHistory;
	private double totalPenalty = 0;
	private double totalMaxPenalty = 0;
	private Fridge fridge;
	private List<Matrix> probabilities;
	private boolean verbose = true;
	
	/** 
	 * True if you want the fridge to start off being full, with random
	 * initial contents.
	 */
	public static boolean RANDOM_INITIAL_CONTENTS = false;
	
	/**
	 * Constructor
	 * @param problemSpecFilename path to input file
	 * @throws IOException
	 */
	public Simulator(String problemSpecPath) throws IOException {
	    this(new ProblemSpec(problemSpecPath));
	}
	
	/**
	 * Constructor
	 * @param spec A ProblemSpec
	 */
	public Simulator(ProblemSpec spec) {
	    problemSpec = spec;
		fridge = problemSpec.getFridge();
		probabilities = problemSpec.getProbabilities();
	
        reset();
		
		if (verbose) {
			System.out.println("Problem spec loaded.");
			System.out.println("Fridge: " + fridge.getName());
			System.out.println("Discount factor: " + 
					problemSpec.getDiscountFactor());
			System.out.println("Cost per failure: " + problemSpec.getCost());
		}
	}
	
	public void reset() {
	    currentWeek = 1;
	    inventory = new ArrayList<Integer>();
	    inventoryHistory = new ArrayList<List<Integer>>();
	    shoppingHistory = new ArrayList<List<Integer>>();
	    userWantsHistory = new ArrayList<List<Integer>>();
	    penaltyHistory = new ArrayList<Double>();
	    totalPenalty = 0;
	    totalMaxPenalty = 0;
	    
	    for (int i = 0; i < fridge.getMaxTypes(); i++) {
            inventory.add(0);
        }
	    
	    if (RANDOM_INITIAL_CONTENTS) {
            for (int i = 0; i < fridge.getCapacity(); i++) {
                int itemType = random.nextInt(fridge.getMaxTypes());
                inventory.set(itemType, inventory.get(itemType) + 1);
            }
        }

	}
	
	/**
	 * Simulate a week. A runtime exception is thrown if the shopping list is 
	 * invalid e.g. if total items in the current inventory plus the total items
	 * in the shopping list exceeds the fridge capacity. If the shopping list is
	 * valid, the user consumption is sampled and the current week is advanced.
	 * @param shopping List of item quantities to buy.
	 */
	public void simulateStep(List<Integer> shopping) {
		if (verbose && currentWeek > problemSpec.getNumWeeks()) {
			System.out.println("Warning: problem spec num weeks exceeded.");
		}
		
		// Process shopping
		if (shopping.size() != fridge.getMaxTypes()) {
			throw new IllegalArgumentException("Invalid shopping list size");
		}
		ArrayList<Integer> tempState = new ArrayList<Integer>(inventory);
		int totalBought = 0;
		for (int i = 0; i < shopping.size(); i++) {
		    totalBought += shopping.get(i);
			if (shopping.get(i) < 0) {
				throw new IllegalArgumentException("Negative shopping?");
			}
			tempState.set(i, tempState.get(i) + shopping.get(i));
		}
		if (totalBought > fridge.getMaxPurchase()) {
		    throw new IllegalArgumentException("Order too large!");
		}
		
		int sum = 0;
		for (int i : tempState) {
			sum += i;
		}
		if (sum > fridge.getCapacity()) {
			throw new IllegalArgumentException("Fridge capacity exceeded");
		}
		shoppingHistory.add(new ArrayList<Integer>(shopping));
		
		// Sample next inventory
		inventoryHistory.add(inventory);
		inventory = tempState;
		List<Integer> wants = sampleUserWants(inventory);
		int numFailures = 0;
		for (int i = 0; i < wants.size(); i++) {
			int net = inventory.get(i) - wants.get(i);
			if (net < 0) {
				inventory.set(i, 0);
				numFailures -= net;
			} else {
				inventory.set(i, net);
			}
		}
		
		// Add user wants to history
		userWantsHistory.add(wants);
		
		// Calculate penalty
		double penalty = problemSpec.getCost() * numFailures;
		totalPenalty += Math.pow(problemSpec.getDiscountFactor(),
				currentWeek - 1) * penalty;
		penaltyHistory.add(penalty);
		int wantsSum = 0;
		for (int i : wants) {
			wantsSum += i;
		}
//		int minNumFailures = wantsSum - fridge.getCapacity();
//		if (minNumFailures < 0) {
//			minNumFailures = 0;
//		}
		int maxNumFailures = (fridge.getMaxItemsPerType() * fridge.getMaxTypes());
		maxNumFailures -= fridge.getCapacity();
		double maxPenalty = maxNumFailures * problemSpec.getCost();
		totalMaxPenalty += Math.pow(problemSpec.getDiscountFactor(),
				currentWeek - 1) * maxPenalty;
		
		if (verbose) {
			System.out.println();
			System.out.println("Week " + currentWeek);
			List<Integer> startInventory = inventoryHistory.get(
					inventoryHistory.size() - 1);
			List<Integer> postShopping = new ArrayList<Integer>();
			for (int i = 0; i < fridge.getMaxTypes(); i++) {
			    postShopping.add(startInventory.get(i) + shopping.get(i));
			}
			System.out.println("Start inventory: " + startInventory);
			System.out.println("Shopping:        " + shopping);
			System.out.println("Post-shopping:   " + postShopping);
			System.out.println("User wants:      " + wants);
			System.out.println("End inventory:   " + inventory);
			System.out.println("Num failures: " + numFailures);
			System.out.println("Penalty this week: " + penalty);
			System.out.println("Total penalty: " + totalPenalty);
			System.out.println("Maximum penalty this week: " + maxPenalty);
		
			if (currentWeek == problemSpec.getNumWeeks()) {
				System.out.println();
				System.out.println("Total maximum penalty: " + totalMaxPenalty);
				double temp = totalPenalty / totalMaxPenalty;
				System.out.println("Total penalty/total max penalty: " + temp);
			}
		}	
		currentWeek ++;	
	}
	
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
		double r = random.nextDouble();
		for (int i = 0; i < prob.size(); i++) {
			sum += prob.get(i);
			if (sum >= r) {
				return i;
			}
		}
		return -1;
	}
	
	/**
	 * Saves the current shopping history and total penalty to file
	 * @param filename The path to the text file to save to
	 * @throws IOException
	 */
	public void saveOutput(String filename) throws IOException {
		problemSpec.saveOutput(filename, getTotalPenalty(), shoppingHistory);
	}

	/**
	 * Set verbose to true for console output
	 * @param verbose
	 */
	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
	
	/**
	 * Get penalty from history.
	 * @precondition week < currentWeek
	 * @param week The week to retrieve. Week starts at 1. 
	 * @return penalty
	 */
	public double getPenalty(int week) {
		return penaltyHistory.get(week - 1);
	}
	
	/**
	 * Get inventory from history
	 * @precondition week < currentWeek
	 * @param week The week to retrieve. Week starts at 1. 
	 * @return the inventory for that week.
	 */
	public List<Integer> getInventoryAt(int week) {
	    return inventoryHistory.get(week - 1);
	}
	
	/**
     * Get shopping list from history
     * @precondition week < currentWeek
     * @param week The week to retrieve. Week starts at 1. 
     * @return the shopping list for that week.
     */
	public List<Integer> getShoppingAt(int week) {
	    return shoppingHistory.get(week - 1);
	}
	
	 /**
     * Get user request from history
     * @precondition week < currentWeek
     * @param week The week to retrieve. Week starts at 1. 
     * @return the user request for that week.
     */
    public List<Integer> getUserRequestAt(int week) {
        return userWantsHistory.get(week - 1);
    }
    
	
	/**
	 * @return the total penalty so far
	 */
	public double getTotalPenalty() {
		return totalPenalty;
	}
	
	/** 
	 * @return the total maximum penalty so far
	 */
	public double getTotalMaxPenalty() {
		return totalMaxPenalty;
	}

	public int getCurrentWeek() {
		return currentWeek;
	}
	
	public List<Integer> getInventory() {
		return new ArrayList<Integer>(inventory);
	}
}
