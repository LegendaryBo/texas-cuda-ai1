package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;


/**
 * Class implements MIMIC Algorithm. contains: - operators - solution space you
 * need to add: - objective function - termination condition
 * 
 * EXPERIMENTAL - USE AT OWN RISK !!
 * 
 * @version 0.1
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvDiscreteVectorMIMIC extends EvAlgorithm<EvKnaryIndividual> {
  
  /**
   * Number of individual, which stay alive.
   */
  private final int sel_k_best;

  /**
   * Dimension of each individual.
   */
  private final int dimension;
  
  /**
   * Table of possible gene values.
   */
  private final int[] geneValues;
  
  /**
   * Selection (first operator).
   */
  private EvKBestSelection<EvKnaryIndividual> selection_operator;

  /**
   * MIMIC operator. This operator generate new, better population.
   */
  private EvDiscreteVectorMIMICOperator mimic_operator;
  
  
  /**
   * Constructor.
   * 
   * @param dim dimension of individuals
   * @param gValues table of possible gene values
   * @param population_size size of population
   * @param sel_k number of individual, which stay alive after selection
   */
  public EvDiscreteVectorMIMIC(final int dim, final int[] gValues, 
      final int population_size, final int sel_k) {

    super(population_size);
    dimension = dim;
    sel_k_best = sel_k;
    geneValues = gValues;
  }
  
  /**
   * {@inheritDoc}
   */
  @Override
  public void init() {

    selection_operator = new EvKBestSelection<EvKnaryIndividual>(sel_k_best);
    mimic_operator = new EvDiscreteVectorMIMICOperator(dimension, geneValues, 
        population_size);

    addOperatorToEnd(selection_operator);
    addOperatorToEnd(mimic_operator);
    setSolutionSpace(mimic_operator.getSolutionSpace());

    super.init();
  }

}
