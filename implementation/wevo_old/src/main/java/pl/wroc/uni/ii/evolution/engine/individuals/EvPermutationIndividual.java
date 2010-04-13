package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * EvPermutationIndividual
 * 
 * @author Donata Malecka, Piotr Baraniak
 * @author Karol "Asgaroth" Stosiek (karol.stosiek@gmail.com)
 */
public class EvPermutationIndividual extends EvIndividual {

  /**
   * 
   */
  private static final long serialVersionUID = 6050861148116245519L;

  private int[] chromosome;


  public EvPermutationIndividual(int d) {
    chromosome = new int[d];
  }


  public EvPermutationIndividual(int[] chromosome) {
    this.chromosome = chromosome.clone();
  }


  @Override
  public EvPermutationIndividual clone() {
    EvPermutationIndividual new_individual =
        new EvPermutationIndividual(chromosome.clone());
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      new_individual.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        new_individual.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return new_individual;
  }


  public int[] getChromosome() {
    return chromosome.clone();
  }


  public void setChromosome(int[] chromosome) {
    this.chromosome = chromosome.clone();
    invalidate();
  }


  public String toString() {
    String str = "(";
    for (int i = 0; i < chromosome.length; i++) {
      str += chromosome[i] + ", ";
    }
    str = str.substring(0, str.length() - 2) + ")";
    return str;
  }


  public int getGeneValue(int i) {
    return chromosome[i];
  }


  public void setGeneValue(int i, int value) {
    chromosome[i] = value;
    this.invalidate();
  }


  public int indexOf(int value) {
    if (value < 0 || value > chromosome.length - 1) {
      throw new IllegalArgumentException("Argument of indexOf must "
          + "be in range [0," + chromosome.length + "] (argument +" + "was: "
          + value + ").");
    }

    for (int i = 0; i < chromosome.length; i++) {
      if (chromosome[i] == value) {
        return i;
      }
    }

    throw new IllegalStateException("Value " + value + " not found in "
        + chromosome.toString() + "!");
  }


  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final EvPermutationIndividual other = (EvPermutationIndividual) obj;
    if (!Arrays.equals(chromosome, other.chromosome))
      return false;
    return true;
  }
}
