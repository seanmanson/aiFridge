package solver;

import java.lang.reflect.Constructor;
import java.util.List;

import problem.ProblemSpec;
import problem.Simulator;

public class Runner {
	/** The default file to read the user model from. */
	public static String DEFAULT_INPUT = "testcases/tiny-v1.txt";
	/** The default file to output to. */
	public static String DEFAULT_OUTPUT = "testcases/output.txt";
	/** The path for the input file. */
	private static String inputPath = null;
	/** The path for the output file. */
	private static String outputPath = null;
	
	/** The default number of simulations to run. */
	public static int DEFAULT_NUM_SIMULATIONS = 3;
	/** The number of simulations to run. */
	private static int numSimulations = 0; 
	
	/** The default solver to use. */
	public static String DEFAULT_SOLVER = "solver.SeanMSolver";
	/** The name of the solver class that will be used. */
	private static String solverName = null;
	
	/** Whether to re-create the solver for every simulation. */
	public static boolean RECREATE_SOLVER = false;

	public static void main(String[] args) throws Exception {
		parseCommandLine(args);
		Class<?> clazz = Class.forName(solverName);
		Constructor<?> ctor = clazz.getConstructor(ProblemSpec.class);
		
		ProblemSpec spec = new ProblemSpec(inputPath);
		
		double totalPenalty = 0;
		double totalMaxPenalty = 0;
		
		Simulator simulator = new Simulator(spec);
		OrderingAgent solver = null;
		if (!RECREATE_SOLVER) {
			solver = (OrderingAgent)ctor.newInstance(spec);
			solver.doOfflineComputation();
		}
		for (int simNo = 0; simNo < numSimulations; simNo++) {
	        
			System.out.printf("Run #%d\n", simNo+1);
			System.out.println("-----------------------------------------------------------");
			
			simulator.reset();
			if (RECREATE_SOLVER) {
				solver = (OrderingAgent)ctor.newInstance(spec);
				solver.doOfflineComputation();
			}
			
			for (int i = 0; i < spec.getNumWeeks(); i++) {
				List<Integer> shopping = solver.generateShoppingList(
						simulator.getInventory(), spec.getNumWeeks() - (i+1));
				simulator.simulateStep(shopping);
			}
			
			totalPenalty += simulator.getTotalPenalty();
			totalMaxPenalty += simulator.getTotalMaxPenalty();
			System.out.println("-----------------------------------------------------------");
		}
		
		simulator.saveOutput(outputPath);
		System.out.printf("Summary statistics from %d runs:\n", numSimulations);
		System.out.println();
		System.out.printf("Total penalty: %f\n", -totalPenalty);
		System.out.printf("Total maximum penalty: %f\n", -totalMaxPenalty);
		System.out.printf("Overall penalty ratio: %f\n", totalPenalty / totalMaxPenalty);
	}
	
	/**
	 * Parses the command line arguments.
	 * 
	 * @param args
	 *            the array of command line arguments.
	 */
	public static void parseCommandLine(String args[]) {
		for (int i = 0; i < args.length; i++) {
			String arg = args[i].trim();
			if (inputPath == null) {
				inputPath = arg;
			} else if (outputPath == null) {
				outputPath = arg;
			} else if (solverName == null) {
			    solverName = arg;
			} else if (numSimulations == 0) {
				numSimulations = Integer.valueOf(arg);
			} 
		}
		if (inputPath == null) {
			inputPath = DEFAULT_INPUT;
		}
		if (outputPath == null) {
			outputPath = DEFAULT_OUTPUT;
		}
		if (solverName == null) {
		    solverName = DEFAULT_SOLVER;
		}
		if (numSimulations == 0) {
			numSimulations = DEFAULT_NUM_SIMULATIONS;
		}
	}

}
