package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Abstract class for MessyIndividual. Represents individual that has multiple
 * alleles, it is a (gene, allele) pair vector of variable length.
 * MessyIndividual express a genotype (vector of allele vectors), each gene can
 * have multiple alleles, can be over or under specified. Extending classes has
 * to implement constructor and clone().
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @param <T> type of allele
 */

public abstract class EvMessyIndividual<T> extends EvIndividual {

  /** list of allele positions in the expressed genotype */
  protected ArrayList<Integer> genes;

  /** list of gene values */
  protected ArrayList<T> alleles;

  /** length of the expressed genotype */
  protected int genotype_length;

  /** genotype is calculated iff genoptype_calculated = true */
  protected boolean genotype_calculated = false;

  /** number of expressed (specified) genes in genotype (not underspecified) */
  protected int expressed_genes_number;

  /** expressed genotype, genotype[gene] contains all gene alleles */
  protected ArrayList<T>[] genotype;


  /**
   * Constructor creates new messy individual with specified genes and alleles.
   * 
   * @param expressed_genes number of genes expressed by individual's chromosome
   * @param genes list of genes (positions of alleles in expressed genotype),
   *        the length has to be the same to alleles, NOTE: it is not copied,
   *        use it later for reading only or call chromosome modifying methods
   *        after that.
   * @param alleles list of alleles (values of genes), the length has to be the
   *        same to genes, NOTE: it is not copied, use it later for reading only
   *        or call chromosome modifying methods after that.
   */
  public EvMessyIndividual(int genotype_length, ArrayList<Integer> genes,
      ArrayList<T> alleles) {
    if (genotype_length < 1)
      throw new IllegalArgumentException(
          "genotype_length must be a positive integer");

    this.genotype_length = genotype_length;
    this.alleles = alleles;
    this.genes = genes;
  }


  // ***************************************************************************
  // Chromosome's methods

  /**
   * Returns individual's chromosome length.
   * 
   * @return chromosome length
   */
  public int getChromosomeLength() {
    return genes.size();
  }


  /**
   * Gets genes list to effective direct access.
   * 
   * @return gene list list of genes, NOTE: it is not a copy, use it for reading
   *         only or call chromosome modifying methods after that.
   */
  public ArrayList<Integer> getGenes() {
    return genes;
  }


  /**
   * Gets alleles list to effective direct access.
   * 
   * @return allele list list of alleles, NOTE: it is not a copy, use it for
   *         reading only or call chromosome modifying methods after that.
   */
  public ArrayList<T> getAlleles() {
    return alleles;
  }


  /**
   * Gets genes within the given range to effective direct access.
   * 
   * @param range_from begin of the range to get genes
   * @param range_to end of the range to get genes
   * @return gene list list of genes, NOTE: it is not a copy, use it for reading
   *         only or call chromosome modifying methods after that.
   */
  public List<Integer> getGenes(int range_from, int range_to) {
    return genes.subList(range_from, range_to);
  }


  /**
   * Gets alleles within the given range to effective direct access.
   * 
   * @param range_from begin of the range to get genes
   * @param range_to end of the range to get genes
   * @return allele list list of alleles, NOTE: it is not a copy, use it for
   *         reading only or call chromosome modifying methods after that.
   */
  public List<T> getAlleles(int range_from, int range_to) {
    return alleles.subList(range_from, range_to);
  }


  /**
   * Gets clone of genes list.
   * 
   * @return gene list copy of list of genes
   */
  public ArrayList<Integer> getGenesList() {
    return new ArrayList<Integer>(genes);
  }


  /**
   * Gets clone of alleles list.
   * 
   * @return allele list copy of list of alleles
   */
  public ArrayList<T> getAllelesList() {
    return new ArrayList<T>(alleles);
  }


  /**
   * Gets clone of genes within the given range.
   * 
   * @param range_from begin of the range to get list
   * @param range_to end of the range to get list
   * @return allele list copy of list of alleles within specified range
   */
  public ArrayList<Integer> getGenesList(int range_from, int range_to) {
    return new ArrayList<Integer>(genes.subList(range_from, range_to));
  }


  /**
   * Gets clone of alleles within given range.
   * 
   * @param range_from begin of the range to get list
   * @param range_to end of the range to get list
   * @return allele list copy of list of alleles within specified range
   */
  public ArrayList<T> getAllelesList(int range_from, int range_to) {
    return new ArrayList<T>(alleles.subList(range_from, range_to));
  }


  /**
   * Sets new chromosome. This method invalidates the individual, also checks
   * length of genes and alleles lists. Can be used after manually modified
   * chromosome.
   */
  public void setChromosome() {
    if (alleles.size() != genes.size())
      throw new IllegalArgumentException(
          "Allele list and gene list must have same length");

    this.invalidate();
  }


  /**
   * Sets new chromosome.
   * 
   * @param genes new gene list to set, NOTE: it is not copied, use it later for
   *        reading only or call chromosome modifying methods after that.
   * @param alleles new allele list to set, NOTE: it is not copied, use it later
   *        for reading only or call chromosome modifying methods after that.
   */
  public void setChromosome(ArrayList<Integer> genes, ArrayList<T> alleles) {

    if (alleles.size() != genes.size())
      throw new IllegalArgumentException(
          "Allele list and gene list must have same length");

    this.invalidate();
    this.genes = genes;
    this.alleles = alleles;
  }


  /**
   * Adds new alleles to the chromosome.
   * 
   * @param position position in chromosome to insert
   * @param genes gene list to insert, the length has to be the same to alleles
   * @param alleles allele list to insert, the length has to be the same to
   *        genes
   */
  public void addAlleles(int position, ArrayList<Integer> gene_list,
      ArrayList<T> allele_list) {

    if (allele_list.size() != gene_list.size())
      throw new IllegalArgumentException(
          "Allele list and gene list must have the same length");

    this.invalidate();
    alleles.addAll(position, allele_list);
    genes.addAll(position, gene_list);
  }


  /**
   * Adds new alleles to the chromosome in the end.
   * 
   * @param genes gene list to insert, the length has to be the same to alleles
   * @param alleles allele list to insert, the length has to be the same to
   *        genes
   */
  public void addAlleles(ArrayList<Integer> gene_list, ArrayList<T> allele_list) {
    this.addAlleles(genes.size(), gene_list, allele_list);
  }


  /**
   * Removes allele from a given position.
   * 
   * @param position position of allele in chromosome to be removed
   */
  public void removeAllele(int position) {
    if (position < 0 || position >= genes.size())
      throw new IllegalArgumentException("Position must be in range");

    this.invalidate();
    alleles.remove(position);
    genes.remove(position);
  }


  /**
   * Gets gene from a given position in chromosome.
   * 
   * @param position position in chromosome to get gene
   * @return gene
   */
  public int getGene(int position) {
    if (position < 0 || position >= genes.size())
      throw new IllegalArgumentException("Position must be in range");

    return genes.get(position);
  }


  /**
   * Gets allele from a given position in chromosome.
   * 
   * @param position position in chromosome to get allele
   * @return allele
   */
  public T getAllele(int position) {
    if (position < 0 || position >= genes.size())
      throw new IllegalArgumentException("Position must be in range");

    return alleles.get(position);
  }


  /**
   * Sets allele and gene in a given position in chromosome.
   * 
   * @param position position in chromosome to set allele
   * @param gene new gene number to set
   * @param allele new allele to set
   */
  public void setAllele(int position, int gene, T allele) {
    if (position < 0 || position >= genes.size())
      throw new IllegalArgumentException("Position must be in range");

    this.invalidate();
    genes.set(position, gene);
    alleles.set(position, allele);
  }


  // ***************************************************************************
  // Genotype's methods

  /**
   * Returns the genotype length.
   * 
   * @return genotype length
   */
  public int getGenotypeLength() {
    return genotype_length;
  }


  /**
   * Calculate the genotype. There is lazy evaluation of the calculating.
   */
  @SuppressWarnings("unchecked")
  protected void calculateGenotype() {

    // Create new genotype - num_expressed_genes length array
    genotype = new ArrayList[genotype_length];
    for (int i = 0; i < genotype_length; i++)
      genotype[i] = new ArrayList<T>(1);

    // For each allel set to its gene
    ListIterator<T> alleles_iterator = alleles.listIterator();
    ListIterator<Integer> genes_iterator = genes.listIterator();

    while (genes_iterator.hasNext()) {
      int pos = genes_iterator.next().intValue();
      T allele = alleles_iterator.next();

      if (pos < 0 || pos >= genotype_length)
        throw new IllegalStateException("Gene " + pos + " is not in range");

      genotype[pos].add(allele);
    }

    // Calculate the number of expressed genes
    expressed_genes_number = 0;
    for (int i = 0; i < genotype_length; i++)
      if (genotype[i].size() > 0)
        expressed_genes_number++;

    genotype_calculated = true;
  }


  /**
   * Gets the expressed genotype - array of expressed_genes length,
   * genotype[gene] contains all gene alleles.
   * 
   * @return genotype NOTE: it is not a copy, use it for reading only or call
   *         chromosome modifying methods after that.
   */
  public ArrayList<T>[] getGenotype() {
    if (!genotype_calculated)
      calculateGenotype();
    return genotype;
  }


  /**
   * Gets the number of expressed genes, non empty genes in the individual's
   * genotype.
   * 
   * @return expressed genes number
   */
  public int getExpressedGenesNumber() {
    if (!genotype_calculated)
      calculateGenotype();
    return expressed_genes_number;
  }


  /**
   * Returns number of common expressed genes with the given individual, this
   * compares the gene lists of individuals and count common genes.
   * 
   * @param individual a messy individual to compare common genes with
   * @return common expressed genes number of common expressed genes
   */
  public int getCommonExpressedGenesNumber(EvMessyIndividual<T> individual) {

    if (genotype_length != individual.getGenotypeLength())
      throw new IllegalArgumentException(
          "Compared individuals have different expressed genes number");

    if (!genotype_calculated)
      calculateGenotype();

    ArrayList<T>[] individual_genotype = individual.getGenotype();

    // Count common expressed genes
    int common = 0;
    for (int i = 0; i < genotype_length; i++)
      if (genotype[i].size() != 0 && individual_genotype[i].size() != 0)
        common++;

    return common;
  }


  // ***************************************************************************
  // Individual's methods

  /**
   * Overrided invalidation, it is used also by lazy calculation of genotype.
   */
  @Override
  protected void invalidate() {
    super.invalidate();
    genotype_calculated = false;
  }


  /**
   * {@inheritDoc}
   */
  public String toString() {
    StringBuilder s = new StringBuilder();

    ListIterator<T> alleles_iterator = alleles.listIterator();
    ListIterator<Integer> genes_iterator = genes.listIterator();

    while (genes_iterator.hasNext()) {
      s.append("<" + genes_iterator.next() + "," + alleles_iterator.next()
          + ">");
    }

    return s.toString();
  }


  @Override
  public boolean equals(Object obj) {
    if (obj == null)
      return false;
    EvMessyIndividual individual = ((EvMessyIndividual) obj);
    // Check if the genotype length is the same
    if (individual.getGenotypeLength() != this.genotype_length)
      return false;
    // Checking if both chromosome are the same
    if (!individual.alleles.equals(alleles))
      return false;
    if (!individual.genes.equals(genes))
      return false;
    return true;
  }

}