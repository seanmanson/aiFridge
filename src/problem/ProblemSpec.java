package problem;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * This class is used for file I/O
 */
public class ProblemSpec {
	
	/** True iff user stochatic model is currently loaded */
	private boolean modelLoaded = false;
	/** The number of weeks the fridge will be evaluated */
	private int numWeeks;
	/** Penalty for when the user fails to find the product wanted */
	private double cost;
	/** Discount factor */
	private double discountFactor;
	/** The fridge type */
	private Fridge fridge;
	/** The probabilities for the user's consumption behaviour */
	private List<Matrix> probabilities;
	
	public ProblemSpec() {
	}
	
	public ProblemSpec(String specFileName) throws IOException {
	    this();
	    loadInputFile(specFileName);
	}
	
	/**
	 * Loads the user's stochastic model from file
	 * @param filename the path of the text file to load.
	 * @throws IOException
	 * 		if the text file doesn't exist or doesn't meet the assignment
	 *      specifications.
	 */
	public void loadInputFile(String filename) throws IOException {
		modelLoaded = false;
		BufferedReader input = new BufferedReader(new FileReader(filename));
		String line;
		int lineNo = 0;
		Scanner s;
		try {
			line = input.readLine();
			lineNo++;
			s = new Scanner(line);
			numWeeks = s.nextInt();
			s.close();

			line = input.readLine();
			lineNo++;
			s = new Scanner(line);
			cost = s.nextDouble();
			s.close();
			
			line = input.readLine();
			lineNo++;
			s = new Scanner(line);
			discountFactor = s.nextDouble();
			s.close();
			
			line = input.readLine();
			lineNo++;
			fridge = new Fridge(line.trim().toLowerCase());
			
			probabilities = new ArrayList<Matrix>();
			for (int k = 0; k < fridge.getMaxTypes(); k++) {
				double[][] data = new double[fridge.getCapacity() + 1]
						[fridge.getMaxItemsPerType() + 1];
				for (int i = 0; i <= fridge.getCapacity(); i++) {
					line = input.readLine();
					lineNo++;
					double rowSum = 0;
					s = new Scanner(line);
					for (int j = 0; j <= fridge.getMaxItemsPerType(); j++) {
						data[i][j] = s.nextDouble();
						rowSum += data[i][j];
					}
					s.close();
					if (Math.round(rowSum*10000)/10000. != 1) {
						throw new InputMismatchException(
								"Row probabilities do not sum to 1.");
					}
				}
				probabilities.add(new Matrix(data));
			}
			modelLoaded = true;
		} catch (InputMismatchException e) {
			throw new IOException(String.format(
					"Invalid number format on line %d: %s", lineNo,
					e.getMessage()));
		} catch (NoSuchElementException e) {
			throw new IOException(String.format("Not enough tokens on line %d",
					lineNo));
		} catch (NullPointerException e) {
			throw new IOException(String.format(
					"Line %d expected, but file ended.", lineNo));
		} finally {
			input.close();
		}
	}
	
	/**
	 * Save output to file
	 * @param filename The file path to save to
	 * @param totalPenalty The total cost
	 * @param history List of all shopping orders starting at week 0
	 * @throws IOException
	 */
	public void saveOutput(String filename, double totalPenalty,
			List<List<Integer>> history) throws IOException {
		String ls = System.getProperty("line.separator");
		FileWriter output = new FileWriter(filename);
		output.write(String.format("%d %s", history.size(), ls));
		for (List<Integer> shopping : history) {
			for (int i : shopping) {
				output.write(String.format("%d %s", i, ls));
			}
		}
		output.write(String.format("%f", totalPenalty));
		output.close();
	}

	public boolean isModelLoaded() {
		return modelLoaded;
	}

	public int getNumWeeks() {
		return numWeeks;
	}

	public double getCost() {
		return cost;
	}

	public double getDiscountFactor() {
		return discountFactor;
	}

	public Fridge getFridge() {
		return fridge;
	}

	public List<Matrix> getProbabilities() {
		return new ArrayList<Matrix>(probabilities);
	}	
}
