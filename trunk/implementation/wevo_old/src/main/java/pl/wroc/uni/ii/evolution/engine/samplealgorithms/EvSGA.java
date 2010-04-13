package pl.wroc.uni.ii.evolution.engine.samplealgorithms;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * An implementation of the popular SGA algorithm.
 * <p>
 * 
 * @author Marcin Golebiowski
 * @param <T> the type of individuals the algorithm works on
 */
public class EvSGA<T extends EvIndividual> extends EvAlgorithm<T> {

  private EvSelection<T> parent_selection;

  private EvOperator<T> children_creator;

  private EvReplacement<T> replacement;


  /**
   * @param population_size
   * @param parent_selection
   * @param children_creator
   * @param replacement
   */
  public EvSGA(int population_size, EvSelection<T> parent_selection,
      EvOperator<T> children_creator, EvReplacement<T> replacement) {
    super(population_size);
    this.parent_selection = parent_selection;
    this.children_creator = children_creator;
    this.replacement = replacement;
  }


  @Override
  public void init() {

    super
        .addOperatorToBeginning(new EvReplacementComposition<T>(
            new EvApplyOnSelectionComposition<T>(parent_selection,
                children_creator), replacement));

    super.init();
  }

}
