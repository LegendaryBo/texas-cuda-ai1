package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryBMDAProbability;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * BMDA alghorithm using bayesian network.
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 *
 */
public class EvBMDA extends EvAlgorithm<EvBinaryVectorIndividual> {

  /** type of selection. */
  private EvSelection<EvBinaryVectorIndividual> selection;
  
  /**
   * Construktor.
   * @param population_size population size
   * @param selection_ selection type
   */
  public EvBMDA(final int population_size,
      final EvSelection<EvBinaryVectorIndividual> selection_) {
    super(population_size);
    this.selection = selection_;
  }

  /**
   * Init the algorithm.
   */
  @Override
  public void init() {
    EvBinaryBMDAProbability bdmaOperator =
        new EvBinaryBMDAProbability(population_size);

    super.addOperatorToEnd(selection);
    super.addOperatorToEnd(bdmaOperator);

    super.init();
  }
}
