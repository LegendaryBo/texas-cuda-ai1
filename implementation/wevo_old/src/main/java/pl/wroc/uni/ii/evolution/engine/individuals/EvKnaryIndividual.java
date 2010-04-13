package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class implementing individual containing vector of integers as genes of
 * specified maximum value. <BR>
 * Each individual must have specified length, which can't be changed during
 * program run. Gene's lowest value is 0, maximum value is defined in
 * constructor.<BR>
 * <BR>
 * Example:<BR>
 * To create individual with 3 possible gene's values, set parameter
 * <B>max_gene_value</B> to 2. The will result in individual which may look
 * like "0,1,0,2,2,0,0,1,2,1".
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvKnaryIndividual extends EvIndividual {

  private static final long serialVersionUID = -3068145629255959376L;

  protected int max_gene_value;

  // main integer table containing genes' values
  protected int[] genes;

  private int length;


  /**
   * Create individual with specified number of genes and with specified maximum
   * value of those genes.<BR>
   * Genes can be set from 0, to </b>max_gene_value</b><BR>
   * By default all the genes are set to 0.
   * 
   * @param length - number of genes
   * @param max_gene_value
   */
  public EvKnaryIndividual(int length, int max_gene_value) {
    // initializing table of specified size with values set to 0
    initializeGenotype(length);

    this.max_gene_value = max_gene_value;
  }


  /**
   * Creates new individual from given table of genes.<br>
   * Individual length is the length of <B>genes</b> table.<BR>
   * !NOTE!<BR>
   * Changing values of <b>genes</b> tablce won't affect the individual<BR>
   * 
   * @param genes - table of genes, each integer should have value between 0 and
   *        <b>max_gene_value</b>
   * @param max_gene_value - maximum value of gene
   */
  public EvKnaryIndividual(int[] genes, int max_gene_value) {
    this.genes = genes.clone(); // cloning genes
    this.max_gene_value = max_gene_value;
  }


  /**
   * Create exact copy of individual.<Br>
   * Its clone genes, but it doesn't clone objective function (which is set to
   * null)
   */
  @Override
  public EvKnaryIndividual clone() {
    EvKnaryIndividual b1 = new EvKnaryIndividual(genes, max_gene_value);
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      b1.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        b1.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return b1;
  }


  /**
   * Get the value of gene indexed by given value.<Br>
   * <BR>
   * If index exceeds the size of individual, <B>IndexOutOfBoundsException</b>
   * is thrown
   * 
   * @param index of the gene (first gene is indexed by 0, last =
   *        getDimension()-1 )
   * @return gene value, from 0 (inclusive) to max. gene value (also inclusive)
   */
  public int getGene(int index) {
    return genes[index];
  }


  /**
   * Method returns whole genotype of the idividual represented by a table of
   * integers.<BR>
   * <BR>
   * !NOTE!<BR>
   * The table which is returned is NOT the same object that is inside
   * individual. It is cloned and changing it won't affect individual.
   * 
   * @return cloned table of individuals
   */
  public int[] getGenes() {
    return genes.clone();
  }


  /**
   * - -- Sets <b>value</b> to the gene specified by given <b>index</b><BR>
   * <BR>
   * If <B>value</b> exceeds max. genes' value, <b>IllegalArgumentException</b>
   * is thrown.
   * 
   * @param index - gene's index (leftmost gene's index = 0, rightmost index =
   *        length-1)
   * @param value to be set
   */
  public void setGene(int index, int value) {
    // check if given value is correct
    if (max_gene_value < value) {
      throw new IllegalArgumentException("Given value (" + value + ") "
          + "excedes max. value of the gene (" + max_gene_value + ")");
    }

    // notifies, that individual's objective function value may changes,
    // so it's need to be reevaluated
    this.invalidate();

    genes[index] = value;
  }


  @Override
  /**
   * Return hasCode of the individual. It returns hashhode of it's genes of
   * arrays
   */
  public int hashCode() {
    return Arrays.hashCode(genes);
  }


  /**
   * Gets dimension f this individual.
   * 
   * @return length of chromosome
   */
  public int getDimension() {
    return genes.length;
  }


  @Override
  /**
   * Check if given object is equals to this individual.<BR>
   * <BR>
   * Returns true when:<BR> - <B>obj</b> is not null - <B>obj</b> is the same
   * class of this individual - <B>obj</b> genes are same as genes of this
   * individual
   */
  public boolean equals(Object obj) {
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    if (!Arrays.equals(genes, ((EvKnaryIndividual) obj).genes))
      return false;
    return true;
  }


  /**
   * Sets the new length to the individual.<BR>
   * If genes table is null, it sets the table of genes of given size.<BR>
   * If new length is higher than current one, redundant genes aren't cut, thy
   * are still accessible.<BR>
   * If new length is higher than current genes table, genes are copied to a
   * table 2x larger/
   * 
   * @param length of the genotype
   */
  public void setLength(int length) {

    // check if new length is ok
    if (length < 1) {
      throw new IllegalArgumentException("Length of k-nary vector "
          + "individual must be higher than 0");
    }

    if (genes == null) {
      initializeGenotype(length);
    } else {
      if (length > this.length) {
        expandGenotype(length);
      }
      if (length < this.length) {
        shrinkGenotype(length);
      }
    }

    this.length = length;
  }


  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  public String toString() {
    StringBuffer output = new StringBuffer();
    for (int i = 0; i < getDimension(); i++) {
      output.append(getGene(i));
    }
    return output.toString();
  }


  /**
   * Sets the new maximum possible value to single gene to a new specified
   * value. <BR>
   * This method doesn't affect directly current genes, even if some of actual
   * genes are higher than the new maximum gene value, they are left as they
   * were.
   * 
   * @param max_gene_value - must be higher than 1
   */
  public void setMaxGeneSize(int max_gene_value) {
    if (max_gene_value < 1) {
      throw new IllegalArgumentException(
          "Maximum gene value must be higher than 0");
    }
    this.max_gene_value = max_gene_value;
  }


  // it adjust genes table so it can match new length
  private void expandGenotype(int length) {
    if (length <= genes.length) {
      this.length = length; // not necessary to adjust current table
    } else { // we need to expand current table

      int[] new_genotype; // replacement table for current one
      if (length > 2 * genes.length) { // multiple current table size
        new_genotype = new int[length];
      } else {
        new_genotype = new int[length]; // when multiple size table is
        // still to small
      }
      // rewrite genes to replacement table
      for (int i = 0; i < genes.length; i++) {
        new_genotype[i] = genes[i];
      }
      genes = new_genotype;
      this.length = length;
    }

  }


  // reduces length of individual, but it's not affect genes table
  private void shrinkGenotype(int length) {
    this.length = length;
  }


  // contruct a new genes table of given size
  private void initializeGenotype(int length) {
    genes = new int[length];
  }


  // TODO
  public int getMaxGeneValue() {
    return max_gene_value;
  }

}
