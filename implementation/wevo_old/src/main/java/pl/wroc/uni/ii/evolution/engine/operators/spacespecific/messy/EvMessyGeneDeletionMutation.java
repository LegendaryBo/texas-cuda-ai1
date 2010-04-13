package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvMutation;

/**
 * This operator deletes randomly genes from all individuals in population
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvMessyGeneDeletionMutation<T extends EvMessyIndividual> extends
    EvMutation<T> {

  private int number_of_genes_to_delete; // number of genes to delete


  /**
   * Constructor
   * 
   * @param number_of_genes_to_delete - how many genes we want delete
   * @param is_clone - true if we want clone the individual (it works only with
   *        apply(EvPopulation) method)
   */
  public EvMessyGeneDeletionMutation(int number_of_genes_to_delete,
      boolean is_clone) {

    this.number_of_genes_to_delete = number_of_genes_to_delete;
    super.setMutateClone(is_clone);
  }


  /**
   * Constructor
   * 
   * @param number_of_genes_to_delete - how many genes we want delete
   */
  public EvMessyGeneDeletionMutation(int number_of_genes_to_delete) {
    this(number_of_genes_to_delete, true);
  }


  /**
   * Deletes some (randomly) genes from individual
   * 
   * @param individual - individual which we want cut
   * @return abbreviated individual
   */
  @Override
  public T mutate(T individual) {
    int individual_length = individual.getChromosomeLength();

    for (int i = 0; i < number_of_genes_to_delete; i++) {
      if (individual_length <= 1)
        break;

      // we choose arbitrarily the position to delete
      int pos = EvRandomizer.INSTANCE.nextInt(individual_length);
      individual.removeAllele(pos);
      --individual_length;
    }

    return individual;
  }

}