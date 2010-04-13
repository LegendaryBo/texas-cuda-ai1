package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.ArrayList;
import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class for MessyIndividual Represents individual that can have multiple gene
 * values.
 * 
 * @author Marcin Golebiowski, Krzysztof Sroka
 */

public class EvSimplifiedMessyIndividual extends EvIndividual {

  public EvSimplifiedMessyIndividual best_found_without_empty_genes;

  private static final long serialVersionUID = -762518078515091478L;

  private ArrayList<Integer>[] chromosome;


  /**
   * Constructor. Creates new individual with full set of genes with values set
   * to 0.
   * 
   * @param chromosome_length Individual's chromosome length
   */
  @SuppressWarnings("unchecked")
  public EvSimplifiedMessyIndividual(int chromosome_length) {
    if (chromosome_length < 1) {
      throw new IllegalArgumentException(
          "Chromosome length should be bigger than zero");
    }
    chromosome = new ArrayList[chromosome_length];

    for (int i = 0; i < chromosome_length; i++) {
      chromosome[i] = new ArrayList<Integer>(0);
    }

  }


  /**
   * Returns individual's chromosome length.
   * 
   * @return chromosome length
   */
  public int getLength() {
    return chromosome.length;
  }


  /**
   * Modify specific gene by set gene to specified values
   * 
   * @param <code>position</code> gene's position in chromosome from
   *        <code> 0 </code>
   * @param <code>gene_values</code> gene's values to be placed at position
   */
  public void setGeneValues(int position, ArrayList<Integer> gene_values) {
    this.invalidate();
    this.removeGene(position);
    this.addGeneValues(position, gene_values);
  }


  /**
   * Modify specific gene by set gene's value to specified value.
   * 
   * @param <code>position</code> gene's position in chromosome from
   *        <code> 0 </code>
   * @param gene_value gene's value to be placed at position
   */
  public void setGeneValue(int position, int gene_value) {
    this.invalidate();
    this.removeGene(position);
    this.addGeneValue(position, gene_value);
  }


  /**
   * Modify specific gene by set nth gene's value to specified value.
   * 
   * @param position gene's position in chromosome from <code> 0 </code>
   * @param gene_value gene's value to be placed at position
   */
  public void setGeneValue(int position, int gene_value, int nth) {
    this.invalidate();
    chromosome[position].set(nth, gene_value);
  }


  /**
   * Modify specific gene by add specified value.
   * 
   * @param position gene's position in chromosome from <code> 0 </code>
   * @param gene_value gene's value to be placed at position
   */
  public void addGeneValue(int position, int gene_value) {
    chromosome[position].add(gene_value);
  }


  /**
   * Modify specific gene by add specified values
   * 
   * @param position gene's position in chromosome from <code> 0 </code>
   * @param gene_values gene's values to be placed at position
   */
  public void addGeneValues(int position, ArrayList<Integer> gen_values) {
    this.invalidate();
    if (gen_values == null || gen_values.size() == 0) {
      return;
    }
    chromosome[position].addAll(gen_values);
  }


  /**
   * Removes a gene from chromosome (sets it's value to empty list)
   * 
   * @param position gene's position from <code> 0 </code>
   */
  public void removeGene(int position) {
    this.invalidate();
    chromosome[position] = new ArrayList<Integer>();
  }


  /**
   * Returns gene values from chromosome. If gene is empty then it return
   * <code> emply list </code>
   * 
   * @param position gene's position from <code> 0 </code>
   * @return gene's value
   */
  public ArrayList<Integer> getGeneValues(int position) {
    return chromosome[position];
  }


  /**
   * Returns first gene value from chromosome.
   * 
   * @param position gene's position from <code> 0 </code>
   * @return gene gene's value
   */
  public int getGeneValue(int position) {
    return chromosome[position].get(0);
  }


  /**
   * Returns nth gene value from chromosome.
   * 
   * @param position gene's position from <code> 0 </code>
   * @param nth number of gene value to get
   * @return gene gene's value
   */
  public int getGeneValue(int position, int nth) {
    return chromosome[position].get(nth);
  }


  @Override
  public EvSimplifiedMessyIndividual clone() {
    EvSimplifiedMessyIndividual cloned =
        new EvSimplifiedMessyIndividual(chromosome.length);

    // clone every gens values
    for (int i = 0; i < chromosome.length; i++) {

      cloned.addGeneValues(i, this.getGeneValues(i));

    }
    cloned.best_found_without_empty_genes = this.best_found_without_empty_genes;
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      cloned.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        cloned.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return cloned;
  }


  /**
   * {@inheritDoc}
   */
  public String toString() {
    StringBuilder builded = new StringBuilder();
    int i = 0;
    for (ArrayList<Integer> gene_values : chromosome) {
      if (gene_values.size() == 0) {
        builded.append("<" + i + ",null>");
        i++;
        continue;
      }

      for (Integer value : gene_values) {
        builded.append("<" + i + "," + value + ">");
      }
      i++;
    }
    return builded.toString();

  }


  @Override
  public boolean equals(Object obj) {

    if (obj == null) {
      return false;
    }
    EvSimplifiedMessyIndividual individual =
        ((EvSimplifiedMessyIndividual) obj);
    // check if individuals are the same length
    if (individual.getLength() != this.chromosome.length) {
      return false;
    }

    // checking every gens values in both chromosomes if they are the same
    for (int i = 0; i < this.chromosome.length; i++) {
      individual = individual.sortGeneValues(i);
      EvSimplifiedMessyIndividual my = this.sortGeneValues(i);

      if (individual.getGeneValues(i) != null
          && !individual.getGeneValues(i).equals(my.getGeneValues(i))) {
        return false;
      }
      if (individual.getGeneValues(i) == null && my.getGeneValues(i) != null) {
        return false;
      }
    }

    return true;
  }


  /**
   * Sorts gene on given position. This individual remains unsorted.
   * 
   * @param position gene position
   * @return New MessyIndividual with the same genes, but stored in correct
   *         order.
   */
  public EvSimplifiedMessyIndividual sortGeneValues(int position) {

    EvSimplifiedMessyIndividual clone =
        (EvSimplifiedMessyIndividual) this.clone();

    ArrayList<Integer> gene_values_list = clone.getGeneValues(position);
    if (gene_values_list.size() == 0) {
      return clone;
    }

    Integer[] gene_values = new Integer[gene_values_list.size()];

    for (int i = 0; i < gene_values.length; i++) {
      gene_values[i] = gene_values_list.get(i);
    }

    Arrays.sort(gene_values);
    ArrayList<Integer> sorted_list = new ArrayList<Integer>();
    for (int val : gene_values) {
      sorted_list.add(val);
    }

    clone.addGeneValues(position, sorted_list);
    return clone;
  }


  /**
   * Remove from individual duplicated gene values at given position. This
   * individual remains unchanged.
   * 
   * @param position
   * @return generated invidual
   */

  public void removeDuplicateGeneValues(int position) {
    ArrayList<Integer> filtered_gene_values = new ArrayList<Integer>();

    if (this.getGeneValues(position).size() != 0) {
      ArrayList<Integer> gene_values_list = this.getGeneValues(position);
      Integer[] gene_values = new Integer[gene_values_list.size()];

      for (int i = 0; i < gene_values.length; i++) {
        gene_values[i] = gene_values_list.get(i);
      }

      Arrays.sort(gene_values);

      int prev_value = gene_values[0];
      filtered_gene_values.add(prev_value);
      for (int k = 1; k < gene_values.length; k++) {
        if (gene_values[k] != prev_value) {
          filtered_gene_values.add(gene_values[k]);
          prev_value = gene_values[k];
        }
      }

      this.setGeneValues(position, filtered_gene_values);
    }
  }


  /**
   * Remove from individual all duplicates.
   * 
   * @return new MessyIndividual instance without duplicated genes
   */
  public void removeAllDuplicateGeneValues() {
    for (int i = this.getLength() - 1; i >= 0; i--) {
      removeDuplicateGeneValues(i);
    }
  }
}
