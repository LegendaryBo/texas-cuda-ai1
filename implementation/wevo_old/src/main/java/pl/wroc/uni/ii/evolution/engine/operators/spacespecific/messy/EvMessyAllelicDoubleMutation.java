package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Alleles mutation with given probability change randomly (uniformly) allele
 * for every gene in chromosome but it works only with messy individuals with
 * real alleles (alleles must be Double).
 * 
 * @author Rafal Paliwoda 'rp@message.pl'
 * @author Mateusz Malinowski 'm4linka@gmail.com'
 */
public class EvMessyAllelicDoubleMutation<T extends EvMessyIndividual<Double>>
    extends EvMutation<T> {

  private double mutation_probability;


  /**
   * Constructor
   * 
   * @param mutation_probability - probability of mutation
   * @param is_clone - true if we want clone individual (it works properly only
   *        with apply(EvPopulation) method)
   */
  public EvMessyAllelicDoubleMutation(double mutation_probability,
      boolean is_clone) {

    if (mutation_probability < 0.0 || mutation_probability > 1.0) {
      throw new IllegalArgumentException(
          "Mutation probabolity must be in range [0,1]");
    }
    this.mutation_probability = mutation_probability;
    super.setMutateClone(is_clone);
  }


  /**
   * Constructor
   * 
   * @param mutation_probability - probability of mutation
   */
  public EvMessyAllelicDoubleMutation(double mutation_probability) {
    this(mutation_probability, true);
  }


  @Override
  public T mutate(T individual) {
    if (mutation_probability == 0.0)
      return individual;

    int individual_len = individual.getChromosomeLength();

    for (int i = 0; i < individual_len; i++) {
      if (EvRandomizer.INSTANCE.nextDouble() <= mutation_probability) {
        int locus = individual.getGene(i);
        double allele;
        double new_allele;
        do {
          allele = individual.getAllele(i);
          new_allele = EvRandomizer.INSTANCE.nextDouble();
        } while (new_allele == allele);
        individual.setAllele(i, locus, new_allele);
      }
    }

    return individual;
  }

}
