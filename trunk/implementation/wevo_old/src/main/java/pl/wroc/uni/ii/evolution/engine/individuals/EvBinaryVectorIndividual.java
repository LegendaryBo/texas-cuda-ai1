package pl.wroc.uni.ii.evolution.engine.individuals;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class for binary individual with fixed length.<BR>
 * Each individual contains defined number of integer values, which can be set
 * to '0' or '1'<BR>
 * It's a more specified version of EvKnaryIndividual (max. gene value set to
 * 1), it contains some additional methods useful for binary vector individuals
 * 
 * @author Tomasz Kozakiewicz
 * @author Piotr Baraniak
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvBinaryVectorIndividual extends EvKnaryIndividual implements
		Serializable {

	private static final long serialVersionUID = -3883047179476641241L;

	private int[] boolTab;
	private int dimension;

	public float[] statystyki;

	public int[] licznikPassow;
	public int[] kartaWygranego = new int[7];

	public int[] wygranaStawka = new int[8];

	public int getGene(int index) {

		return (boolTab[index / 32] & (1 << index % 32)) >>> index % 32;

	}

	public int getDimension() {
		return dimension;
	}

	@Override
	public int[] getGenes() {
		return boolTab;
	}
	
	@Override
	public void setGene(int index, int value) {
		if (value == 0)
			boolTab[index / 32] = boolTab[index / 32]
					& ~(1 << index % 32);
		else
			boolTab[index / 32] = boolTab[index / 32] | (1 << index % 32);
		this.invalidate();
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(boolTab);
	}

	private EvBinaryVectorIndividual() {
		super(0, 0);
	}

	/**
	 * Create individual of specified length with all genes set to 0.<BR>
	 * Objective function is set to null by default.
	 * 
	 * @param d
	 *              fixed length of individual
	 */
	public EvBinaryVectorIndividual(int d) {
		// binary vector is k-nary individual with max_gene_value set to 1
		super(0, 1);
		dimension = d;
		boolTab = new int[1 + d / 32];

	}

	/**
	 * Creates new individual with genes given in a table.<BR>
	 * Every integer should have value 0 (false) or 1 (true).<BR>
	 * The table is cloned and further changes won't affect the individual.<br>
	 * 
	 * @param genes
	 *              of new individual. Every integer should be set to 0 or 1
	 */
	public EvBinaryVectorIndividual(int[] genes, int dimension) {

		super(0, 1);
		if (dimension == 0)
			throw new IllegalStateException("zle kurwa");

		this.dimension = dimension;
		boolTab = genes.clone();

	}

	public EvBinaryVectorIndividual(int[] is) {
		super(is, 1);
		this.dimension = is.length;
		boolTab = new int[1 + dimension / 32];
		for (int i=0; i < dimension; i++)
			setGene(i, is[i]);
	}

	/**
	 * String representation of individual containing only binary position
	 * specified in Integer ArrayList.<BR>
	 * 
	 * @param positions
	 * @return
	 */
	public String toString(ArrayList<Integer> positions) {

		if (positions.size() > getDimension()) {
			throw new IllegalArgumentException("Argument length to big");
		}

		StringBuffer output = new StringBuffer();
		for (int pos : positions) {
			if (pos > getDimension()) {
				throw new IllegalArgumentException(
						"One of ArrayList value "
								+ "is bigger than size of the individual");
			}

			if (getGene(pos) == 0) {
				output.append("0");
			} else {
				output.append("1");
			}

		}
		return output.toString();
	}

	@Override
	/*
	 * * Create exact copy of individual.<Br> Its clone genes, but it doesn't
	 * clone objective function (which is set to null)
	 */
	public EvBinaryVectorIndividual clone() {

		EvBinaryVectorIndividual b1 = new EvBinaryVectorIndividual(boolTab,
				dimension);
		for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
			b1.addObjectiveFunction(this.getObjectiveFunction(i));
			if (this.isEvaluated(i)) {
				b1.assignObjectiveFunctionValue(
						getObjectiveFunctionValue(i), i);
			}
		}
		b1.statystyki = statystyki;
		b1.licznikPassow = licznikPassow;
		b1.kartaWygranego = kartaWygranego;

		return b1;
	}

	/**
	 * Compares every pair of bit at the same position of current and given
	 * BinaryIndividual and returns number of bits that are equal.
	 * 
	 * @param individual
	 *              - Individual to be compared
	 * @return number of bits that are the same
	 */
	public int countIdenticalBits(EvBinaryVectorIndividual individual) {
		if (individual.getDimension() != getDimension()) {
			throw new IllegalStateException(
					"Number of bits must be the same");
		} else {
			int counter = 0;
			for (int i = 0; i < getDimension(); i++) {
				if (individual.getGene(i) == getGene(i)) {
					counter++;
				}
			}
			return counter;
		}
	}

	public int getJakisHashcode() {
		int wynik=13;
		for (int i=0; i < dimension; i++) {
			if (getGene(i) == 1)
				wynik = wynik*183 +191;
			else
				wynik = wynik*521 + 31;
		}
		return wynik;
	}
	
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		StringBuffer output = new StringBuffer();
		for (int i = 0; i < getDimension(); i++) {
			if (getGene(i) == 0) {
				output.append("0");
			} else {
				output.append("1");
			}
		}
		return output.toString();
	}

	// it shall be only possible to set max gene value to 1
	@Override
	public void setMaxGeneSize(int max_gene_value) {
		if (max_gene_value != 1) {
			throw new IllegalArgumentException(
					"Maximum gene value must be set to 1");
		}
		this.max_gene_value = 1;

	}

	@Override
	/*
	 * * Check if given object is equals to this individual.<BR> <BR> Returns
	 * true when:<BR> - <B>obj</b> is not null - <B>obj</b> is the same class
	 * of this individual - <B>obj</b> genes are same as genes of this
	 * individual
	 */
	public boolean equals(Object obj) {
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		if (hashCode() != obj.hashCode())
			return false;
		return true;
	}

	public int[] getBits(ArrayList<Integer> positions) {
		return null;
	}

	public void setBits(ArrayList<Integer> positions, int[] is) {

	}

}
